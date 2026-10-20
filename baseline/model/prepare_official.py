"""Prepare a reproducible N24News subset while preserving provided splits.

No model training or tokenizer required. Needs Pillow for selected-image decoding.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from baseline import allocate, write_json


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_file(task):
    path, stamp = task
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return stamp, h.hexdigest()


QUOTES = str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'", '–': '-', '—': '-'})


def canon(token):
    return ''.join(c for c in unicodedata.normalize('NFKD', token.translate(QUOTES))
                   if not unicodedata.combining(c)).casefold()


def matching_prefix_end(body, phrase):
    """Only remove full known metadata phrases at the start, never interior text."""
    if not isinstance(phrase, str):
        return None
    phrase = html.unescape(phrase).strip()
    if len(re.findall(r'\w+', phrase)) < 4:
        return None
    expected = [canon(m.group()) for m in re.finditer(r'\w+|[^\w\s]', phrase)]
    actual = list(re.finditer(r'\w+|[^\w\s]', body[:max(500, len(phrase) * 3)]))
    if len(actual) >= len(expected) and [canon(m.group()) for m in actual[:len(expected)]] == expected:
        return actual[len(expected) - 1].end()
    return None


def clean_body(record):
    raw = record.get('article')
    if not isinstance(raw, str):
        return '', []
    body = html.unescape(raw).strip()
    actions = []
    # Restrict all removal to leading known caption/headline, repeated at most 6 times.
    for _ in range(6):
        removed = False
        for field in ('caption', 'headline'):
            end = matching_prefix_end(body, record.get(field))
            if end is not None:
                body = body[end:].lstrip()
                actions.append('leading_' + field)
                # Some source records append photographer attribution to a caption.
                if field == 'caption':
                    match = re.match(r'^Credit\s*\.{3}[^\n]{0,300}(?:\n+|$)', body)
                    if match:
                        body = body[match.end():].lstrip()
                        actions.append('caption_credit')
                removed = True
                break
        if not removed:
            break
    return body, actions


class UnionFind:
    def __init__(self):
        self.parent = []

    def add(self):
        self.parent.append(len(self.parent))

    def find(self, n):
        while n != self.parent[n]:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def run(args):
    from PIL import Image
    root, out, cache_dir = Path(args.data_root).resolve(), Path(args.out_dir).resolve(), Path(args.cache_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('out-dir must be new/empty; do not overwrite frozen splits.')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'image_file_hashes.json'
    cache = json.loads(cache_file.read_text(encoding='utf-8')) if cache_file.exists() else {}
    report = {'seed': args.seed, 'source_root': str(root), 'source_counts': {}, 'source_class_counts': {},
              'source_file_sha256': {}, 'invalid_counts': {}, 'preprocessing': {},
              'duplicate_exclusions': {}, 'selected_counts': {}, 'selected_class_counts': {},
              'policy': {'splits': 'Preserve official train/dev/test; dev is named val in output.',
                         'body': 'HTML unescape + strip; remove only leading complete caption/headline matching >=4 words (accent/quote insensitive), at most 6 times; remove following Credit... line only after a removed caption.',
                         'dedupe': 'Union by article ID/URL, original/cleaned normalized body, image ID or image file SHA256. Drop ALL records in cross-split or conflicting-label groups. Keep first within-split duplicate.',
                         'sampling': 'Proportional per class; minimum 5 articles per class per selected split; seed fixed.',
                         'image_integrity': 'File SHA256 for entire source; fully decode selected images, replace failures within same class.',
                         'test_usage': 'Only integrity/duplicate/sampling checks; no model scoring or hyperparameter selection.'}}
    rows, ids, uf = [], {}, UnionFind()
    excluded_rows, examples = [], []
    for split, source in [('train', 'train'), ('val', 'dev'), ('test', 'test')]:
        path = root / 'news' / f'nytimes_{source}.json'
        file_bytes = path.read_bytes()
        report['source_file_sha256'][split] = hashlib.sha256(file_bytes).hexdigest()
        data = json.loads(file_bytes)
        del file_bytes
        tasks = {}
        for record in data:
            image = root / 'imgs' / f"{record.get('image_id')}.jpg"
            if image.is_file():
                stat = image.stat()
                stamp = f'{image}|{stat.st_size}|{stat.st_mtime_ns}'
                if stamp not in cache:
                    tasks[stamp] = image
        print(f'{split}: hashing {len(tasks)} source image files ({args.workers} workers)', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for count, (stamp, value) in enumerate(executor.map(hash_file, [(path, stamp) for stamp, path in tasks.items()]), 1):
                cache[stamp] = value
                if count % 5000 == 0:
                    write_json(cache_file, cache)
                    print(f'{split}: {count}/{len(tasks)} image files hashed', flush=True)
        write_json(cache_file, cache)
        report['source_counts'][split] = len(data)
        report['source_class_counts'][split] = dict(Counter(str(r.get('section')) for r in data))
        invalid, cleaning = Counter(), Counter()
        for i, record in enumerate(data):
            label = record.get('section')
            raw = record.get('article')
            body, actions = clean_body(record)
            image_id = record.get('image_id')
            reason = None
            if not isinstance(label, str) or not label.strip():
                reason = 'missing_label'
            elif not isinstance(raw, str) or not raw.strip():
                reason = 'missing_original_body'
            elif not body:
                reason = 'empty_after_cleaning'
            elif image_id is None or not str(image_id).strip():
                reason = 'missing_image_id'
            image = root / 'imgs' / f'{image_id}.jpg'
            if reason is None and not image.is_file():
                reason = 'missing_image_file'
            if reason:
                invalid[reason] += 1
                excluded_rows.append({'split': split, 'source_index': i, 'reason': reason})
                continue
            stat = image.stat()
            stamp = f'{image}|{stat.st_size}|{stat.st_mtime_ns}'
            if stamp not in cache:
                with image.open('rb') as f:
                    h = hashlib.sha256()
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(block)
                cache[stamp] = h.hexdigest()
            file_hash = cache[stamp]
            cleaned_hash = sha(re.sub(r'\s+', ' ', body).strip().casefold())
            raw_hash = sha(re.sub(r'\s+', ' ', raw).strip().casefold())
            article_id = str(record.get('article_id') or '').strip()
            url = str(record.get('article_url') or '').strip()
            keys = [('clean_body', cleaned_hash), ('raw_body', raw_hash), ('image_id', str(image_id)), ('image_file', file_hash)]
            if article_id:
                keys.append(('article_id', article_id))
            if url:
                keys.append(('url', url))
            n = len(rows)
            uf.add()
            for key in keys:
                if key in ids:
                    uf.union(n, ids[key])
                else:
                    ids[key] = n
            rows.append({'id': f'{split}:{i}:{article_id or image_id}', 'source_index': i,
                         'source_split': split, 'article_id': article_id, 'article_url': url,
                         'label': label.strip(), 'headline': record.get('headline', ''),
                         'body': body, 'image': str(image), 'body_hash': cleaned_hash,
                         'original_body_hash': raw_hash, 'image_file_hash': file_hash,
                         'image_id': str(image_id), 'cleaning_actions': actions})
            cleaning.update(actions)
            cleaning['articles_changed'] += int(body != raw.strip())
            if actions and len(examples) < 12 and split == 'train':
                examples.append({'source_index': i, 'headline': record.get('headline'), 'actions': actions,
                                 'before': raw[:650], 'after': body[:650]})
            if (i + 1) % 5000 == 0:
                print(f'{split}: {i+1}/{len(data)} audited', flush=True)
        report['invalid_counts'][split] = dict(invalid)
        report['preprocessing'][split] = dict(cleaning)
        write_json(cache_file, cache)
        print(f'{split}: {len(data)} source records processed', flush=True)
        del data
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    pools = {s: defaultdict(list) for s in ('train', 'val', 'test')}
    duplicate_counts = {s: Counter() for s in pools}
    duplicate_groups = Counter()
    for members in groups.values():
        splits = {rows[i]['source_split'] for i in members}
        labels = {rows[i]['label'] for i in members}
        if len(splits) > 1 or len(labels) > 1:
            reason = 'cross_split_duplicate_group' if len(splits) > 1 else 'conflicting_labels_group'
            duplicate_groups[reason] += 1
            rejected = members
        else:
            first = rows[members[0]]
            pools[first['source_split']][first['label']].append(first)
            rejected = members[1:]
            reason = 'within_split_duplicate'
        for i in rejected:
            row = rows[i]
            duplicate_counts[row['source_split']][reason] += 1
            excluded_rows.append({'split': row['source_split'], 'source_index': row['source_index'], 'reason': reason})
    report['duplicate_group_counts'] = dict(duplicate_groups)
    report['duplicate_exclusions'] = {s: dict(v) for s, v in duplicate_counts.items()}
    classes = sorted(pools['train'])
    if len(classes) != 24 or any(sorted(p) != classes for p in pools.values()):
        raise ValueError('All 24 classes must remain present in all splits.')
    rng = random.Random(args.seed)
    selected = {}
    invalid_selected = []
    for split, target in [('train', args.train_size), ('val', args.val_size), ('test', args.test_size)]:
        report.setdefault('eligible_class_counts', {})[split] = {k: len(v) for k, v in pools[split].items()}
        quotas = allocate(Counter({k: len(v) for k, v in pools[split].items()}), target)
        chosen = []
        for label in classes:
            candidates = pools[split][label]
            rng.shuffle(candidates)
            accepted = 0
            for row in candidates:
                if accepted == quotas[label]:
                    break
                try:
                    with Image.open(row['image']) as image:
                        image = image.convert('RGB')
                        image.load()
                        # Match existing baseline feature extraction's integrity check.
                        row['image_hash'] = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    invalid_selected.append({'split': split, 'id': row['id'], 'reason': type(error).__name__})
                    continue
                chosen.append(row)
                accepted += 1
            if accepted < quotas[label]:
                raise ValueError(f'Insufficient decodable images for {split}/{label}')
        rng.shuffle(chosen)
        selected[split] = chosen
        report['selected_counts'][split] = len(chosen)
        report['selected_class_counts'][split] = dict(Counter(r['label'] for r in chosen))
        print(f'{split}: selected and decoded {len(chosen)} images', flush=True)
    # Also enforce no decoded-pixel-identical images among the selected partitions.
    for field in ('id', 'article_id', 'article_url', 'body_hash', 'original_body_hash', 'image_id', 'image_file_hash', 'image_hash'):
        seen = {}
        for split, split_rows in selected.items():
            for row in split_rows:
                value = row[field]
                if not value:
                    continue
                if value in seen:
                    raise ValueError(f'Selected duplicate {field} in {seen[value]} and {split}; inspect before training.')
                seen[value] = split
    out.mkdir(parents=True, exist_ok=True)
    for split, split_rows in selected.items():
        write_json(out / f'{split}.json', split_rows)
    report['class_names'] = classes
    report['selected'] = sum(report['selected_counts'].values())
    report['split_counts'] = report['selected_counts']
    report['invalid_selected_images'] = invalid_selected
    report['verification'] = '24 classes in all splits; requested counts; fully decoded selected images; no selected duplicate IDs/URLs/body hashes/image file or pixel hashes.'
    report['arguments'] = vars(args)
    write_json(out / 'data_report.json', report)
    write_json(out / 'exclusions.json', excluded_rows)
    write_json(out / 'cleaning_examples.json', examples)
    write_json(out / 'selected_ids.json', {s: [r['id'] for r in v] for s, v in selected.items()})
    lines = ['# N24News 데이터 준비 결과', '',
             '| 분할 | 원본 기사 수 | 선정 기사 수 |', '|---|---:|---:|']
    for s in pools:
        lines.append(f"| {s} | {report['source_counts'][s]:,} | {report['selected_counts'][s]:,} |")
    lines += ['', f"실제 배포본 합계: {sum(report['source_counts'].values()):,}개. 논문 보고치 61,218개와 구분.",
              '', '## 제외 내역', '', '| 분할 | 본문 등 결측·빈 내용 | 중복·레이블 충돌 |', '|---|---:|---:|']
    for s in pools:
        lines.append(f"| {s} | {sum(report['invalid_counts'][s].values())} | {sum(report['duplicate_exclusions'][s].values())} |")
    lines += ['', '제외 후 남은 후보에서 표집한 수치이며, 단순히 원본에서 3,000개만 무작위로 뽑은 것이 아님.',
              '', '## 전처리', '', '기존 분할 유지, dev는 val로 이름만 변경. 원본 데이터는 수정하지 않음.',
              '본문 앞의 알려진 사진 설명·제목만 조건부 제거. HTML 엔티티 복원 및 바깥 공백 제거.',
              '중복 그룹이 분할을 넘으면 그룹 전체 제외. 같은 분할 내 중복은 첫 기사만 유지.',
              '이미지 파일 전체 해시 점검 및 선정 이미지 전체 디코딩 확인.',
              '', '## 클래스별 선정 수', '', '| 주제 | train | val | test |', '|---|---:|---:|---:|']
    for c in classes:
        lines.append(f"| {c} | {report['selected_class_counts']['train'][c]} | {report['selected_class_counts']['val'][c]} | {report['selected_class_counts']['test'][c]} |")
    lines += ['', '## 해석상 주의', '',
              '이번 3,000개 표본은 예비 실험용. 클래스별 표본이 작은 조건에서 차이를 일반화하지 말 것.',
              'Fashion & Style은 배포본의 실제 레이블을 그대로 유지함. 논문 표의 Fashion과 이름이 다름.',
              '중복 제거와 본문 정제는 보수적인 자동 규칙이며, 유사 이미지/유사 기사와 모든 캡션 잔존을 보장하지 못함.',
              'cleaning_examples.json으로 전처리 전후를 직접 확인할 것.',
              'test 파일은 모델 학습·튜닝에 사용하지 않음. 검증 세트 결과만 프로포절 예비 결과로 보고할 것.',
              '모델 학습 및 성능 측정은 아직 수행하지 않았음.',
              '', '## 다음 실행', '', 'baseline.py train --run-dir <이 폴더 경로> --tokens 32 --device cpu',
              '', '본문·이미지는 이미 manifest에 연결되어 있으므로 baseline.py prepare를 다시 실행하지 말 것.']
    (out / '데이터준비결과.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('source_counts', 'invalid_counts', 'duplicate_exclusions', 'selected_counts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--cache-dir', default='work/n24news_audit')
    p.add_argument('--train-size', type=int, default=2000)
    p.add_argument('--val-size', type=int, default=500)
    p.add_argument('--test-size', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=8, help='Parallel local image-file hashing')
    run(p.parse_args())
