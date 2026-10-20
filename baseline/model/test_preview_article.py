"""Export checks with a fake tokenizer; no model/tokenizer downloads."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from preview_article import cut_body, export_article, parser


class FakeTokenizer:
    model_max_length = 512
    init_kwargs = {}

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def build_inputs_with_special_tokens(self, ids):
        return [0] + ids + [2]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, max_length, **kwargs):
        return {"input_ids": self.build_inputs_with_special_tokens(self.encode(text)[:max_length - 2])}

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]

    def save_pretrained(self, path):
        Path(path).mkdir()
        (Path(path) / "fixture.txt").write_text("fake tokenizer for tests only", encoding="utf-8")


class PreviewTests(unittest.TestCase):
    def test_prefix_ids_special_tokens_and_short_input(self):
        n, cuts = cut_body("a" * 40, FakeTokenizer(), [8, 32, 128])
        self.assertEqual(n, 40)
        self.assertEqual([c["actual_body_tokens"] for c in cuts], [8, 32, 40])
        self.assertEqual([c["model_input_tokens"] for c in cuts], [10, 34, 42])
        self.assertEqual(cuts[0]["body_token_ids"], cuts[1]["body_token_ids"][:8])
        self.assertTrue(cuts[2]["shorter_than_requested"])

    def test_html_escape_and_export_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (8, 8), "blue").save(root / "1.png")
            args = parser().parse_args(["--json", str(root / "example.json"),
                "--images", str(root), "--out-dir", str(root / "preview")])
            record = {"headline": '<script>alert("x")</script>', "category": "Test",
                      "body": "  A < B & C. " * 20, "image_id": 1}
            result = export_article(record, 0, FakeTokenizer(), args)
            out = root / "preview"
            page = (out / "preview.html").read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>", page)
            self.assertIn("data:image/png;base64,", page)
            self.assertEqual(result["body"], record["body"].strip())
            for name in ("article.json", "prefixes.txt", "tokenizer/fixture.txt"):
                self.assertTrue((out / name).exists())
            with self.assertRaises(ValueError):
                export_article(record, 0, FakeTokenizer(), args)


if __name__ == "__main__":
    unittest.main()
