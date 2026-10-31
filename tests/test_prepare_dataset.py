import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_dataset import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_selects_exact_reproducible_subset_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            for index in range(503):
                image_path = source_dir / "image_{:03d}.png".format(index)
                Image.new("RGB", (4, 4), (index % 256, 0, 0)).save(image_path)
            (source_dir / "broken.jpg").write_bytes(b"not an image")

            first_output = root / "subset"
            first_manifest = prepare_dataset(
                source_dir, first_output, num_images=500, random_seed=42)
            copied_images = [
                path for path in first_output.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            self.assertEqual(len(copied_images), 500)
            self.assertEqual(
                len({entry["filename"] for entry in first_manifest["images"]}),
                500)
            with (first_output / "manifest.json").open(encoding="utf-8") as file:
                self.assertEqual(json.load(file), first_manifest)

            repeated_manifest = prepare_dataset(
                source_dir, first_output, num_images=500, random_seed=42)
            self.assertEqual(repeated_manifest, first_manifest)

            second_output = root / "second_subset"
            second_manifest = prepare_dataset(
                source_dir, second_output, num_images=500, random_seed=42)
            self.assertEqual(second_manifest, first_manifest)

    def test_rejects_nonempty_mismatched_output(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            for index in range(2):
                Image.new("RGB", (4, 4)).save(
                    source_dir / "image_{}.png".format(index))

            output_dir = root / "subset"
            output_dir.mkdir()
            (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_dataset(
                    source_dir, output_dir, num_images=1, random_seed=42)
            self.assertEqual(
                (output_dir / "keep.txt").read_text(encoding="utf-8"),
                "keep")


if __name__ == "__main__":
    unittest.main()
