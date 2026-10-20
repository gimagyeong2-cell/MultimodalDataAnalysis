"""N24News proposal baseline. See README.md for Korean instructions.

Frozen RoBERTa CLS + frozen ResNet-50 global pooled features; linear fusion.
No article crawling. Test data are untouched during model selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def records_from(path, records_key=None):
    data = read_json(path)
    if records_key:
        data = data[records_key]
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        keys = list(data)[:20] if isinstance(data, dict) else []
        raise ValueError(f"Expected a JSON list of records. Top-level keys: {keys}. "
                         "Use --records-key if the list is inside an object.")
    return data


def inspect_data(args):
    records = records_from(args.json, args.records_key)
    print(f"Records: {len(records)}")
    if records:
        print("First record keys:", list(records[0]))
        for key, value in records[0].items():
            print(f"  {key}: {str(value)[:180]}")


def image_path(record, args):
    root = Path(args.images).resolve()
    if args.image_field:
        value = record.get(args.image_field)
        if not value:
            raise ValueError("missing_image_path")
        path = Path(str(value))
        candidates = [path if path.is_absolute() else root / path]
    else:
        value = record.get(args.image_id_field)
        if value is None:
            raise ValueError("missing_image_id")
        path = root / str(value)
        candidates = [path] if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else [
            Path(str(path) + ext) for ext in (".jpg", ".jpeg", ".png", ".webp")]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ValueError("missing_image_file")


def allocate(counts, total):
    """Proportional class sampling, with >=5 rows/class for 3-way splits."""
    if total < 5 * len(counts):
        raise ValueError("Sample size must be at least 5 * number of classes.")
    if any(n < 5 for n in counts.values()):
        raise ValueError("Every class needs >=5 valid unique articles; inspect exclusions.")
    if total >= sum(counts.values()):
        return dict(counts)
    quota = {k: total * n / sum(counts.values()) for k, n in counts.items()}
    chosen = {k: min(n, max(5, int(quota[k]))) for k, n in counts.items()}
    while sum(chosen.values()) > total:
        k = max((k for k in chosen if chosen[k] > 5), key=lambda k: chosen[k] - quota[k])
        chosen[k] -= 1
    while sum(chosen.values()) < total:
        k = max((k for k in chosen if chosen[k] < counts[k]), key=lambda k: quota[k] - chosen[k])
        chosen[k] += 1
    return chosen


def prepare(args):
    from PIL import Image
    run = Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError("run-dir must be empty/new. Use a new directory for a new experiment.")
    records = records_from(args.json, args.records_key)
    raw_counts = Counter()
    valid, excluded = [], Counter()
    seen_body, seen_image, seen_article = set(), set(), set()
    image_hashes = {}
    for index, record in enumerate(records):
        label = record.get(args.label_field)
        if label is None or str(label).strip() == "":
            excluded["missing_label"] += 1
            continue
        label = str(label).strip()
        raw_counts[label] += 1
        body = record.get(args.body_field)
        if not isinstance(body, str) or not body.strip():
            excluded["missing_body"] += 1
            continue
        body_key = digest(re.sub(r"\s+", " ", body).strip().casefold())
        article_key = str(record.get("article_url") or record.get("id") or "").strip()
        try:
            path = image_path(record, args)
            # Hash decoded RGB pixels to catch identical images with different metadata.
            if str(path) not in image_hashes:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.load()
                    h = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    image_hashes[str(path)] = h
            image_key = image_hashes[str(path)]
        except ValueError as error:
            excluded[str(error)] += 1
            continue
        except (OSError, Image.DecompressionBombError):
            excluded["invalid_image"] += 1
            continue
        duplicate = (body_key in seen_body or image_key in seen_image or
                     (article_key and article_key in seen_article))
        # Record even excluded duplicates, so chains of duplicate identities cannot leak.
        seen_body.add(body_key)
        seen_image.add(image_key)
        if article_key:
            seen_article.add(article_key)
        if duplicate:
            excluded["duplicate_body_image_or_article"] += 1
            continue
        valid.append({"id": f"row-{index}", "label": label, "body": body.strip(),
                      "image": str(path), "body_hash": body_key, "image_hash": image_key})
    counts = Counter(x["label"] for x in valid)
    if len(counts) != args.expected_classes or set(counts) != set(raw_counts):
        raise ValueError(f"Expected {args.expected_classes} complete classes; valid={dict(counts)}, "
                         f"raw={dict(raw_counts)}, excluded={dict(excluded)}. Check field names.")
    total = min(args.sample_size, len(valid)) if args.sample_size else len(valid)
    selected_counts = allocate(counts, total)
    rng = random.Random(args.seed)
    grouped = defaultdict(list)
    for row in valid:
        grouped[row["label"]].append(row)
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(grouped):
        rows = grouped[label]
        rng.shuffle(rows)
        rows = rows[:selected_counts[label]]
        n_val = max(1, round(len(rows) * .2))
        n_test = max(1, round(len(rows) * .2))
        splits["val"].extend(rows[:n_val])
        splits["test"].extend(rows[n_val:n_val + n_test])
        splits["train"].extend(rows[n_val + n_test:])
    run.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        write_json(run / f"{name}.json", rows)
    report = {"source_json": str(Path(args.json).resolve()), "source_records": len(records),
              "raw_class_counts": dict(raw_counts), "valid_unique_records": len(valid),
              "exclusions": dict(excluded), "selected": total, "seed": args.seed,
              "split_policy": "per-class approximately 60/20/20; exact duplicate removal before sampling",
              "class_names": sorted(counts), "arguments": vars(args),
              "split_counts": {k: len(v) for k, v in splits.items()},
              "class_counts": {k: dict(Counter(r["label"] for r in v)) for k, v in splits.items()}}
    report["arguments"].pop("func", None)
    write_json(run / "data_report.json", report)
    print(json.dumps({k: report[k] for k in ("source_records", "valid_unique_records", "exclusions", "split_counts")}, indent=2))


def feature_key(rows, settings, modality):
    payload = {"rows": [(r["id"], r["body_hash"], r["image_hash"]) for r in rows],
               "settings": settings, "modality": modality}
    return digest(json.dumps(payload, sort_keys=True))[:24]


def features(rows, run, settings, device_name, batch_size):
    import numpy as np
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    from transformers import AutoTokenizer, AutoModel
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    cache = run / "features"
    cache.mkdir(exist_ok=True)
    outputs = []
    for modality in ("text", "image"):
        target = cache / f"{modality}-{feature_key(rows, settings, modality)}.npy"
        if target.exists():
            print("Cached:", target.name)
            outputs.append(np.load(target, allow_pickle=False))
            continue
        if modality == "text":
            tokenizer = AutoTokenizer.from_pretrained(settings["text_model"], revision=settings["text_revision"])
            encoder = AutoModel.from_pretrained(settings["text_model"], revision=settings["text_revision"])
            total_length = settings["tokens"] + tokenizer.num_special_tokens_to_add(pair=False)
            max_length = min(tokenizer.model_max_length, encoder.config.max_position_embeddings - tokenizer.pad_token_id - 1)
            if total_length > max_length:
                raise ValueError(f"Requested {settings['tokens']} body tokens + special tokens = {total_length}, "
                                 f"but encoder supports {max_length}. Use <= {max_length - 2} body tokens.")
        else:
            weights = ResNet50_Weights.IMAGENET1K_V2
            encoder = resnet50(weights=weights)
            encoder.fc = torch.nn.Identity()
            transform = weights.transforms()
        encoder.requires_grad_(False).eval().to(device)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                if modality == "text":
                    encoded = tokenizer([r["body"] for r in batch], truncation=True,
                                        max_length=total_length, padding=True, return_tensors="pt")
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    values = encoder(**encoded).last_hidden_state[:, 0, :]
                else:
                    tensors = []
                    for row in batch:
                        # Refuse silently changed data rather than evaluating on stale inputs.
                        with Image.open(row["image"]) as im:
                            im = im.convert("RGB")
                            current = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                            if current != row["image_hash"]:
                                raise ValueError(f"Image changed since prepare: {row['image']}")
                            tensors.append(transform(im))
                    values = encoder(torch.stack(tensors).to(device))
                batches.append(values.float().cpu().numpy())
                print(f"{modality}: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
        array = np.concatenate(batches)
        np.save(target, array)
        outputs.append(array)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return outputs


def metrics(y, predicted, classes):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    labels = list(range(len(classes)))
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "per_class": classification_report(y, predicted, labels=labels, target_names=classes,
                                               output_dict=True, zero_division=0)}


def environments():
    from importlib.metadata import version
    return {"python": platform.python_version(), **{p: version(p) for p in
            ("torch", "torchvision", "transformers", "scikit-learn", "numpy", "Pillow", "joblib")}}


def save_predictions(run, name, rows, predictions, classes):
    with (run / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true_label", *predictions.keys()])
        for i, row in enumerate(rows):
            writer.writerow([row["id"], row["label"], *[classes[int(p[i])] for p in predictions.values()]])


def train(args):
    import numpy as np
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from huggingface_hub import model_info
    run = Path(args.run_dir).resolve()
    if (run / "models.joblib").exists():
        raise ValueError("Training already saved here. Prepare a new run directory for a new experiment.")
    report = read_json(run / "data_report.json")
    train_rows, val_rows = read_json(run / "train.json"), read_json(run / "val.json")
    classes = report["class_names"]
    mapping = {c: i for i, c in enumerate(classes)}
    # Resolve mutable model reference once, then record/use exact commit for reproducibility.
    revision = model_info(args.text_model, revision=args.text_revision).sha
    settings = {"text_model": args.text_model, "text_revision": revision, "tokens": args.tokens,
                "text_pooling": "CLS", "image_model": "resnet50", "image_weights": "IMAGENET1K_V2",
                "encoders_frozen": True}
    write_json(run / "model_settings.json", settings)
    text, image = features(train_rows + val_rows, run, settings, args.device, args.batch_size)
    n = len(train_rows)
    y_train = np.array([mapping[r["label"]] for r in train_rows])
    y_val = np.array([mapping[r["label"]] for r in val_rows])
    inputs = {"text_only": text, "image_only": image, "early_fusion": np.concatenate([text, image], axis=1)}
    models, scores, predictions, grid = {}, {}, {}, {}
    for name, x in inputs.items():
        best_score, best = -1., None
        grid[name] = []
        for c in args.c_grid:
            pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=args.max_iter,
                                     solver="lbfgs", random_state=report["seed"]))
            pipeline.fit(x[:n], y_train)
            pred = pipeline.predict(x[n:])
            score = metrics(y_val, pred, classes)["macro_f1"]
            grid[name].append({"C": c, "val_macro_f1": score})
            if score > best_score:
                best_score, best = score, pipeline
        models[name] = best
        predictions[name] = best.predict(x[n:])
        scores[name] = metrics(y_val, predictions[name], classes)
    majority = DummyClassifier(strategy="most_frequent").fit(np.zeros((n, 1)), y_train)
    models["majority"] = majority
    predictions["majority"] = majority.predict(np.zeros((len(val_rows), 1)))
    scores["majority"] = metrics(y_val, predictions["majority"], classes)
    result = {"evaluation_split": "validation (used for C selection; preliminary results)",
              "units": "0 to 1; delta_pp = 100 * difference", "settings": settings,
              "split_counts": report["split_counts"], "scores": scores, "C_selection": grid,
              "delta_macro_f1": scores["early_fusion"]["macro_f1"] - scores["text_only"]["macro_f1"]}
    result["delta_pp"] = 100 * result["delta_macro_f1"]
    joblib.dump({"models": models, "classes": classes, "settings": settings}, run / "models.joblib")
    write_json(run / "validation_results.json", result)
    write_json(run / "environment.json", environments())
    save_predictions(run, "validation_predictions.csv", val_rows, predictions, classes)
    lines = ["# Validation preliminary results", "", "| Model | Macro-F1 | Accuracy |",
             "|---|---:|---:|"]
    lines += [f"| {name} | {v['macro_f1']:.4f} | {v['accuracy']:.4f} |" for name, v in scores.items()]
    lines += ["", f"Image contribution: {result['delta_pp']:+.2f} Macro-F1 points (0-100 scale).",
              "", "These are validation scores used for selecting C, not final test scores."]
    (run / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def evaluate_test(args):
    import numpy as np
    import joblib
    run = Path(args.run_dir).resolve()
    if (run / "test_results.json").exists():
        raise ValueError("Test results already exist. Do not repeatedly tune on this holdout.")
    # Only load joblib artifacts produced locally by this script.
    bundle = joblib.load(run / "models.joblib")
    rows = read_json(run / "test.json")
    classes = bundle["classes"]
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[r["label"]] for r in rows])
    text, image = features(rows, run, bundle["settings"], args.device, args.batch_size)
    inputs = {"text_only": text, "image_only": image, "early_fusion": np.concatenate([text, image], axis=1),
              "majority": np.zeros((len(rows), 1))}
    predictions = {name: model.predict(inputs[name]) for name, model in bundle["models"].items()}
    scores = {name: metrics(y, p, classes) for name, p in predictions.items()}
    delta = scores["early_fusion"]["macro_f1"] - scores["text_only"]["macro_f1"]
    write_json(run / "test_results.json", {"evaluation_split": "test", "n": len(rows),
               "scores": scores, "delta_macro_f1": delta, "delta_pp": 100 * delta})
    save_predictions(run, "test_predictions.csv", rows, predictions, classes)
    print(json.dumps({name: {k: v for k, v in s.items() if k != "per_class"} for name, s in scores.items()}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Show JSON schema before choosing field names")
    inspect.add_argument("--json", required=True)
    inspect.add_argument("--records-key")
    inspect.set_defaults(func=inspect_data)
    prep = sub.add_parser("prepare", help="Validate, deduplicate, sample and freeze splits")
    prep.add_argument("--json", required=True)
    prep.add_argument("--images", required=True)
    prep.add_argument("--run-dir", required=True)
    prep.add_argument("--records-key")
    prep.add_argument("--body-field", default="body")
    prep.add_argument("--label-field", default="category")
    prep.add_argument("--image-id-field", default="image_id")
    prep.add_argument("--image-field", help="Explicit image path field, relative to --images or absolute")
    prep.add_argument("--sample-size", type=int, default=3000, help="0 uses all valid articles")
    prep.add_argument("--expected-classes", type=int, default=24)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare)
    training = sub.add_parser("train", help="Train/select using train+val only")
    training.add_argument("--run-dir", required=True)
    training.add_argument("--tokens", type=int, default=32, help="Body tokens EXCLUDING special tokens")
    training.add_argument("--text-model", default="FacebookAI/roberta-base")
    training.add_argument("--text-revision", default="main")
    training.add_argument("--c-grid", nargs="+", type=float, default=[.01, .1, 1.])
    training.add_argument("--max-iter", type=int, default=2000)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    training.set_defaults(func=train)
    test = sub.add_parser("evaluate-test", help="Final evaluation, only after settings are frozen")
    test.add_argument("--run-dir", required=True)
    test.add_argument("--batch-size", type=int, default=16)
    test.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    test.set_defaults(func=evaluate_test)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if getattr(args, "sample_size", 0) < 0 or getattr(args, "tokens", 1) < 1 or getattr(args, "batch_size", 1) < 1:
        raise SystemExit("Invalid negative/zero size argument")
    args.func(args)
