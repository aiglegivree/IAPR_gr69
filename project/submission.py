"""Submission csv generation for the UNO project."""

import argparse
import csv
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "iapr-26-uno-vision-challenge"
SEGMENTATION_DIR = PROJECT_DIR / "segmentation"

for path in (PROJECT_DIR, SEGMENTATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clustering import prediction_row_for_image
from cnn import Classifier, DEFAULT_MODEL_PATH

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

SEGMENTATION_FILE = SEGMENTATION_DIR / "segmentation_utils.py"
SEGMENTATION_SPEC = importlib.util.spec_from_file_location("uno_segmentation_module", SEGMENTATION_FILE)
segmentation = importlib.util.module_from_spec(SEGMENTATION_SPEC)
SEGMENTATION_SPEC.loader.exec_module(segmentation)


DEFAULT_TEST_IMAGES_DIR = DATA_DIR / "test_images"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "submission_pipeline.csv"

SUBMISSION_COLUMNS = [
    "image_id",
    "center_card",
    "active_player",
    "player_1_cards",
    "player_2_cards",
    "player_3_cards",
    "player_4_cards",
]
EMPTY_ROW = {
    "center_card": "EMPTY",
    "active_player": "EMPTY",
    "player_1_cards": "EMPTY",
    "player_2_cards": "EMPTY",
    "player_3_cards": "EMPTY",
    "player_4_cards": "EMPTY",
}


def progress(iterable, total, desc):
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc)
        return
    for index, item in enumerate(iterable, start=1):
        if index == 1 or index == total or index % 10 == 0:
            print(f"{desc}: {index}/{total}")
        yield item


def configure_segmentation(backend):
    if backend == "auto":
        backend = segmentation.configure_acceleration()
    segmentation.set_acceleration_backend(backend)
    return backend


def empty_submission_row(image_id):
    return {"image_id": image_id, **EMPTY_ROW}


def generate_submission(
    image_dir=DEFAULT_TEST_IMAGES_DIR,
    output_path=DEFAULT_OUTPUT_CSV,
    model_path=DEFAULT_MODEL_PATH,
    limit=None,
    workers=8,
    classifier_device="auto",
    classifier_cpu=False,
    segmentation_backend="cpu",
    eps=525,
    buffer=200,
):
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = configure_segmentation(segmentation_backend)
    classifier = Classifier(model_path, device=classifier_device, cpu=classifier_cpu)

    print(f"Classifier device: {classifier.device}")
    print(f"Checkpoint validation accuracy: {classifier.checkpoint.get('val_acc', 'unknown')}")
    print(f"Segmentation backend: {backend}")

    image_ids = [path.stem for path in sorted(image_dir.glob("*.jpg"))]
    if limit is not None:
        image_ids = image_ids[:limit]

    def process(image_id):
        try:
            row = prediction_row_for_image(image_dir / f"{image_id}.jpg", classifier, eps=eps, buffer=buffer)
            return row, None
        except Exception as exc:
            return empty_submission_row(image_id), (image_id, repr(exc))

    if workers <= 1:
        results = [process(image_id) for image_id in progress(image_ids, len(image_ids), "submission")]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(progress(executor.map(process, image_ids), len(image_ids), "submission"))

    rows = []
    failures = []
    for row, failure in results:
        rows.append(row)
        if failure is not None:
            failures.append(failure)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")
    if failures:
        print(f"Failures: {len(failures)}")
        print(failures[:10])

    return rows, failures


def parse_submission_args():
    parser = argparse.ArgumentParser(description="Generate the UNO submission CSV.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_TEST_IMAGES_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--segmentation-backend", choices=("cpu", "cuda", "opencl", "auto"), default="cpu")
    parser.add_argument("--eps", type=int, default=525)
    parser.add_argument("--buffer", type=int, default=200)
    return parser.parse_args()


def run_submission_cli():
    args = parse_submission_args()
    generate_submission(
        image_dir=args.image_dir,
        output_path=args.output,
        model_path=args.model,
        limit=args.limit,
        workers=args.workers,
        classifier_device=args.device,
        classifier_cpu=args.cpu,
        segmentation_backend=args.segmentation_backend,
        eps=args.eps,
        buffer=args.buffer,
    )
