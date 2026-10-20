"""Download-free checks for split integrity and data validation.

Run: python -m unittest -v test_baseline.py
"""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import baseline


class PrepareTests(unittest.TestCase):
    def test_deduplication_shared_cohort_and_reproducible_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            rows = []
            for category in range(3):
                for number in range(20):
                    i = category * 20 + number
                    Image.new("RGB", (4, 4), (i, i + 1, i + 2)).save(images / f"{i}.png")
                    rows.append({"category": str(category), "body": f"Unique body {i}", "image_id": i})
            rows.append(dict(rows[0]))  # duplicate body and image
            rows.append({"category": "0", "body": "", "image_id": 1})
            rows.append({"category": "1", "body": "Unavailable image", "image_id": 999})
            source = root / "data.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            def run(name):
                args = baseline.parser().parse_args(["prepare", "--json", str(source),
                    "--images", str(images), "--run-dir", str(root / name),
                    "--expected-classes", "3", "--sample-size", "45"])
                args.func(args)
                return {s: baseline.read_json(root / name / f"{s}.json") for s in ("train", "val", "test")}
            a, b = run("run1"), run("run2")
            self.assertEqual(a, b)
            self.assertEqual(sum(map(len, a.values())), 45)
            for field in ("id", "body_hash", "image_hash"):
                sets = [{r[field] for r in a[s]} for s in a]
                for i in range(3):
                    for j in range(i):
                        self.assertTrue(sets[i].isdisjoint(sets[j]))
            for rows in a.values():
                self.assertEqual({r["label"] for r in rows}, {"0", "1", "2"})
            report = baseline.read_json(root / "run1" / "data_report.json")
            self.assertEqual(report["valid_unique_records"], 60)
            self.assertEqual(report["exclusions"], {"duplicate_body_image_or_article": 1,
                             "missing_body": 1, "missing_image_file": 1})

    def test_small_sample_rejected(self):
        with self.assertRaises(ValueError):
            baseline.allocate({"a": 20, "b": 20}, 9)

    def test_allocation_totals_and_minimum(self):
        for total in (15, 16, 30, 50, 100):
            counts = {"a": 80, "b": 15, "c": 5}
            result = baseline.allocate(counts, total)
            self.assertEqual(sum(result.values()), total)
            self.assertTrue(all(5 <= result[k] <= counts[k] for k in counts))

    def test_nested_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.json"
            path.write_text('{"data": [{"body": "hello"}]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                baseline.records_from(path)
            self.assertEqual(baseline.records_from(path, "data"), [{"body": "hello"}])


if __name__ == "__main__":
    unittest.main()
