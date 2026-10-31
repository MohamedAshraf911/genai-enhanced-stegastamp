import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "coco" / "val2017"
OUTPUT_DIR = PROJECT_ROOT / "data" / "coco" / "subset_500"
NUM_IMAGES = 500
RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MANIFEST_NAME = "manifest.json"


def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError):
        return False


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(source_dir, selected, num_images, random_seed):
    return {
        "source_dir": source_dir.as_posix(),
        "num_images": num_images,
        "random_seed": random_seed,
        "images": [
            {
                "filename": path.name,
                "sha256": sha256_file(path),
            }
            for path in selected
        ],
    }


def existing_subset_matches(output_dir, manifest):
    manifest_path = output_dir / MANIFEST_NAME
    if not output_dir.is_dir() or not manifest_path.is_file():
        return False

    try:
        with manifest_path.open(encoding="utf-8") as file:
            existing_manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    if existing_manifest != manifest:
        return False

    expected_files = {
        entry["filename"]: entry["sha256"]
        for entry in manifest["images"]
    }
    actual_entries = {
        path.name
        for path in output_dir.iterdir()
    }
    if actual_entries != set(expected_files) | {MANIFEST_NAME}:
        return False

    return all(
        sha256_file(output_dir / filename) == expected_hash
        for filename, expected_hash in expected_files.items()
    )


def prepare_dataset(
    source_dir=SOURCE_DIR,
    output_dir=OUTPUT_DIR,
    num_images=NUM_IMAGES,
    random_seed=RANDOM_SEED,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(
            "Dataset directory not found: {}".format(source_dir.resolve()))
    if num_images <= 0:
        raise ValueError("num_images must be greater than zero")

    candidates = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and is_valid_image(path)
    )
    if len(candidates) < num_images:
        raise ValueError(
            "Found only {} valid images in {}; need {}.".format(
                len(candidates), source_dir.resolve(), num_images))

    selected = random.Random(random_seed).sample(candidates, num_images)
    manifest = create_manifest(
        source_dir, selected, num_images, random_seed)

    if output_dir.exists():
        if existing_subset_matches(output_dir, manifest):
            print("Subset already matches manifest: {}".format(output_dir))
            return manifest
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                "Output directory is not empty or does not match the "
                "expected subset: {}".format(output_dir.resolve()))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".{}-".format(output_dir.name),
        dir=output_dir.parent,
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        for source_path in selected:
            shutil.copy2(source_path, staging_dir / source_path.name)

        manifest_path = staging_dir / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")

        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.replace(output_dir)

    print(
        "Copied {} images to {} (seed {}).".format(
            num_images, output_dir, random_seed))
    print("Manifest: {}".format(output_dir / MANIFEST_NAME))
    return manifest


def main():
    prepare_dataset()


if __name__ == "__main__":
    main()
