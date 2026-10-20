"""Action Item 1: inspect one news article and export 8/32/128-token prefixes.

CPU only; downloads tokenizer files, not model weights. Run --help for options.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
from pathlib import Path

from baseline import image_path, records_from, write_json


def cut_body(body, tokenizer, lengths):
    """Slice token IDs directly. Decoded strings are for display, not re-training."""
    tokenizer.truncation_side = "right"
    ids = tokenizer.encode(body, add_special_tokens=False, truncation=False)
    cuts = []
    for length in lengths:
        prefix = ids[:length]
        model_ids = tokenizer.build_inputs_with_special_tokens(prefix)
        # Verify this is the same token sequence as baseline.py's truncation policy.
        reference = tokenizer(body, add_special_tokens=True, truncation=True,
                              max_length=length + tokenizer.num_special_tokens_to_add(pair=False))["input_ids"]
        if model_ids != reference:
            raise ValueError("Tokenizer prefix differs from baseline truncation. Check tokenizer settings.")
        cuts.append({"requested_body_tokens": length, "actual_body_tokens": len(prefix),
                     "shorter_than_requested": len(ids) < length,
                     "text": tokenizer.decode(prefix, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False),
                     "body_token_ids": prefix, "token_pieces": tokenizer.convert_ids_to_tokens(prefix),
                     "model_input_ids": model_ids, "model_input_tokens": len(model_ids),
                     "exceeds_tokenizer_input_limit": len(model_ids) > tokenizer.model_max_length})
    return len(ids), cuts


def export_article(record, index, tokenizer, args):
    from PIL import Image
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty/new --out-dir to avoid overwriting another preview.")
    body = record.get(args.body_field)
    headline = record.get(args.headline_field)
    label = record.get(args.label_field)
    for name, value in ((args.body_field, body), (args.headline_field, headline), (args.label_field, label)):
        if value is None or not str(value).strip():
            raise ValueError(f"Missing field '{name}' at index {index}. Check baseline.py inspect and field options.")
    if not isinstance(body, str):
        raise ValueError("Body must be a string.")
    body = body.strip()  # identical to baseline.py prepare
    path = image_path(record, args)
    image_bytes = path.read_bytes()
    with Image.open(path) as im:
        im.verify()
        mime = Image.MIME.get(im.format, "image/jpeg")
    total, cuts = cut_body(body, tokenizer, args.lengths)
    result = {"source_json": str(Path(args.json).resolve()), "record_index_zero_based": index,
              "headline": str(headline), "label": str(label), "image_path": str(path),
              "image_file_sha256": hashlib.sha256(image_bytes).hexdigest(),
              "tokenizer": args.text_model, "requested_revision": args.text_revision,
              "resolved_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
              "tokenizer_input_limit": tokenizer.model_max_length,
              "body_tokens_exclude_special_tokens": True, "truncation_side": "right",
              "body_normalization": "strip outer whitespace only; same as baseline prepare",
              "body_total_tokens": total, "body": body, "cuts": cuts}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "article.json", result)
    text_lines = ["Action Item 1: article prefix examples", f"Headline: {headline}",
                  f"Category: {label}", f"Original body tokens: {total}",
                  "Body token counts EXCLUDE special tokens.", ""]
    cards = []
    escape = lambda text: html.escape(str(text), quote=True)
    for cut in cuts:
        length = cut["requested_body_tokens"]
        note = "원문이 목표 길이보다 짧음" if cut["shorter_than_requested"] else "본문 앞부분"
        if cut["exceeds_tokenizer_input_limit"]:
            note += " · 특수 토큰 포함 시 모델 입력 한도 초과"
        text_lines += [f"[{length} body tokens / actual {cut['actual_body_tokens']} / "
                       f"with special tokens {cut['model_input_tokens']}]", cut["text"], ""]
        cards.append(f"<section><h2>{length}토큰</h2><small>실제 본문 {cut['actual_body_tokens']}토큰 · "
                     f"특수 토큰 포함 {cut['model_input_tokens']}토큰 · {escape(note)}</small>"
                     f"<p>{escape(cut['text'])}</p></section>")
    (out / "prefixes.txt").write_text("\n".join(text_lines), encoding="utf-8")
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    page = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Action Item 1 — 뉴스 본문 길이 비교</title>
<style>
body{font:16px/1.7 system-ui,sans-serif;background:#f3f5f8;color:#192632;margin:0}
main{max-width:1100px;margin:32px auto;padding:24px}h1{line-height:1.3}
header,section,details{background:white;padding:24px;border-radius:12px;margin:16px 0}
.intro{display:flex;gap:24px;align-items:flex-start}.intro img{width:42%;max-height:360px;object-fit:contain}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.grid section{margin:0}
p,pre{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#526475}
@media(max-width:760px){.grid{grid-template-columns:1fr}.intro{display:block}.intro img{width:100%}}
@media print{body{background:white}main{margin:0;padding:0}section{break-inside:avoid}}
</style><main><h1>같은 기사 본문의 길이별 비교</h1>
"""
    page += (f'<header class="intro"><img alt="기사 대표 이미지" src="{data_url}"><div>'
             f'<h2>{escape(headline)}</h2><p>주제: {escape(label)}<br>원문 본문: {total}토큰<br>'
             f'토크나이저: {escape(args.text_model)}</p></div></header>'
             '<p>아래 길이는 특수 토큰을 제외한 본문 토큰 수입니다. 같은 본문을 앞에서부터 자릅니다.</p>'
             f'<div class="grid">{"".join(cards)}</div>'
             f'<details><summary>전체 본문 보기</summary><p>{escape(body)}</p></details>'
             '<p><small>토큰 중간에서 단어가 끝날 수 있습니다. 표시 텍스트보다 article.json의 토큰 ID가 '
             '정확한 입력 기록입니다. 이 예시는 모델 성능 결과가 아닙니다.</small></p></main></html>')
    (out / "preview.html").write_text(page, encoding="utf-8")
    # Keep a local tokenizer snapshot so the same tokenization can be reproduced.
    tokenizer.save_pretrained(out / "tokenizer")
    print("\n".join(text_lines))
    print("HTML:", out / "preview.html")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out-dir", default="action1_preview")
    p.add_argument("--index", type=int, default=0, help="Zero-based article index")
    p.add_argument("--lengths", nargs="+", type=int, default=[8, 32, 128])
    p.add_argument("--records-key")
    p.add_argument("--body-field", default="body")
    p.add_argument("--headline-field", default="headline")
    p.add_argument("--label-field", default="category")
    p.add_argument("--image-id-field", default="image_id")
    p.add_argument("--image-field")
    p.add_argument("--text-model", default="FacebookAI/roberta-base")
    p.add_argument("--text-revision", default="main")
    p.add_argument("--local-files-only", action="store_true", help="Use a cached/local tokenizer only")
    return p


def main():
    args = parser().parse_args()
    if any(n < 1 for n in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        raise SystemExit("Lengths must be positive and distinct.")
    records = records_from(args.json, args.records_key)
    if not 0 <= args.index < len(records):
        raise SystemExit(f"--index must be in 0..{len(records)-1}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, revision=args.text_revision,
                                              local_files_only=args.local_files_only)
    export_article(records[args.index], args.index, tokenizer, args)


if __name__ == "__main__":
    main()
