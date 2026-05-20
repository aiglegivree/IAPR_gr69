"""Post-classification game logic for one UNO image.

This file contains :
- segmentation-based patch extraction
- token detection
- center card selection
- player attribution
- duplicate hand-card collapse
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SRC_DIR = Path(__file__).resolve().parent

for path in (SRC_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cards_to_players_utils import cluster_2_player_mapping, clustering as cluster_points
from cnn import Classifier
import segmentation_utils as segmentation

CENTER_SYMBOL_COUNT = 2
DETECTION_PAIR_DISTANCE = 475
DETECTION_PAIR_DISTANCE_TOLERANCE = 60

PLAYER_ID_TO_NAME = {1: "p1", 2: "p2", 3: "p3", 4: "p4"}
PLAYER_ID_TO_COLUMN = {
    1: "player_1_cards",
    2: "player_2_cards",
    3: "player_3_cards",
    4: "player_4_cards",
}
COLOR_PREFIX = {"blue": "b", "green": "g", "red": "r", "yellow": "y", "black": "black"}


def card_string(color, symbol):
    """Convert a color and a CNN symbol prediction into the final card label."""

    if symbol in {"wild", "draw_4"}:
        return symbol
    prefix = COLOR_PREFIX.get(color, color)
    return symbol if prefix == "black" else f"{prefix}_{symbol}"


def active_player_from_token(token_center, width, height, buffer=200):
    """Infer the active player from the detected token center."""

    if token_center is None:
        return "EMPTY"

    anchors = np.array(
        [
            [width // 2, height - buffer],
            [width - buffer, height // 2],
            [width // 2, buffer],
            [buffer, height // 2],
        ],
        dtype=np.float32,
    )
    distances = np.linalg.norm(anchors - np.asarray(token_center, dtype=np.float32), axis=1)
    return PLAYER_ID_TO_NAME[int(np.argmin(distances)) + 1]


def sort_player_cards(records, player_id):
    """Sort one player's detected cards in a stable reading order."""

    if player_id in (1, 3):
        return sorted(records, key=lambda row: (row["center"][0], row["center"][1]))
    return sorted(records, key=lambda row: (row["center"][1], row["center"][0]))


def collapse_duplicate_detections(records):
    """Remove repeated detections of the same card based on center distance."""

    grouped = defaultdict(list)
    for order, record in enumerate(records):
        grouped[record["card"]].append((order, record))

    kept = []
    for card_records in grouped.values():
        removed = set()
        used = set()
        pairs = []

        for left in range(len(card_records)):
            for right in range(left + 1, len(card_records)):
                left_record = card_records[left][1]
                right_record = card_records[right][1]
                distance = np.linalg.norm(
                    np.asarray(left_record["center"], dtype=np.float32)
                    - np.asarray(right_record["center"], dtype=np.float32)
                )
                delta = abs(distance - DETECTION_PAIR_DISTANCE)
                if delta <= DETECTION_PAIR_DISTANCE_TOLERANCE:
                    pairs.append((delta, left, right))

        for _, left, right in sorted(pairs):
            if left in used or right in used:
                continue
            used.update((left, right))
            removed.add(right)

        for index, (order, record) in enumerate(card_records):
            if index not in removed:
                kept.append((order, record["card"]))

    kept.sort(key=lambda item: item[0])
    return [card for _, card in kept]


def detect_and_classify(image_rgb, classifier):
    """Detect symbol patches, then classify each patch with the CNN."""

    detected = segmentation.detect_symbols(Image.fromarray(image_rgb))
    predictions = classifier.classify_patches([patch for patch, _, _ in detected])
    records = []

    for (patch, color, center), (symbol, confidence) in zip(detected, predictions):
        records.append(
            {
                "patch": patch,
                "color": color,
                "symbol": symbol,
                "card": card_string(color, symbol),
                "center": tuple(float(value) for value in center),
                "confidence": confidence,
            }
        )

    return records


def assign_players(records, width, height, eps=525, buffer=200):
    """Assign each detected card to the center pile or to one player."""

    if not records:
        return records

    image_center = np.array([width / 2, height / 2], dtype=np.float32)
    ordered = sorted(
        range(len(records)),
        key=lambda index: np.linalg.norm(np.asarray(records[index]["center"], dtype=np.float32) - image_center),
    )
    center_indices = set(ordered[: min(CENTER_SYMBOL_COUNT, len(records))])

    for index in center_indices:
        records[index]["player_id"] = 0

    hand_records = [record for index, record in enumerate(records) if index not in center_indices]
    if not hand_records:
        return records

    points = np.array([record["center"] for record in hand_records], dtype=np.float32)
    cluster_centers, cluster_labels = cluster_points(points, eps=eps, min_samples=1)
    player_ids = cluster_2_player_mapping(cluster_centers, cluster_labels, width, height, buffer=buffer)

    for record, player_id in zip(hand_records, player_ids):
        record["player_id"] = int(player_id) if player_id is not None else None

    return records


def prediction_row_for_image(image_path, classifier, eps=525, buffer=200):
    """Run the full post-CNN pipeline for one image and return one csv row."""

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image)
    height, width = image_rgb.shape[:2]

    records = detect_and_classify(image_rgb, classifier)
    assign_players(records, width, height, eps=eps, buffer=buffer)

    grouped = defaultdict(list)
    for record in records:
        player_id = record.get("player_id")
        if player_id is not None:
            grouped[player_id].append(record)

    center_records = sorted(grouped.get(0, []), key=lambda record: record["confidence"], reverse=True)
    center_card = center_records[0]["card"] if center_records else "EMPTY"
    token_center = segmentation.get_token_center(image)

    row = {
        "image_id": Path(image_path).stem,
        "center_card": center_card,
        "active_player": active_player_from_token(token_center, width, height, buffer=buffer),
        "player_1_cards": "EMPTY",
        "player_2_cards": "EMPTY",
        "player_3_cards": "EMPTY",
        "player_4_cards": "EMPTY",
    }

    for player_id, column in PLAYER_ID_TO_COLUMN.items():
        player_records = sort_player_cards(grouped.get(player_id, []), player_id)
        cards = collapse_duplicate_detections(player_records)
        row[column] = ";".join(cards) if cards else "EMPTY"

    return row
