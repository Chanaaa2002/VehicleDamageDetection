from typing import List, Dict, Optional


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calculate IoU between two boxes.

    Box format:
    [x1, y1, x2, y2]

    IoU means how much two boxes overlap.
    """

    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


def find_damaged_part(
    damage_box: List[float],
    part_detections: List[Dict],
    min_iou: float = 0.05
) -> Optional[Dict]:
    """
    Find which vehicle part overlaps most with the damage box.

    damage_box example:
    [100, 120, 250, 300]

    part_detections example:
    [
        {"part": "front_bumper", "confidence": 0.91, "box": [80, 100, 300, 330]},
        {"part": "front_light", "confidence": 0.88, "box": [310, 100, 400, 180]}
    ]
    """

    best_part = None
    best_iou = 0.0

    for part in part_detections:
        part_box = part["box"]
        iou = calculate_iou(damage_box, part_box)

        if iou > best_iou:
            best_iou = iou
            best_part = part

    if best_part is None or best_iou < min_iou:
        return None

    return {
        "part": best_part["part"],
        "part_confidence": best_part["confidence"],
        "overlap_iou": round(best_iou, 4)
    }


def fuse_damage_and_part(
    damage_detection: Dict,
    part_detections: List[Dict]
) -> Dict:
    """
    Final fusion output.

    damage_detection example:
    {
        "damage_type": "dent",
        "confidence": 0.76,
        "box": [100, 120, 250, 300]
    }
    """

    damaged_part = find_damaged_part(
        damage_box=damage_detection["box"],
        part_detections=part_detections
    )

    if damaged_part is None:
        part_name = "unknown_part"
        part_confidence = 0.0
        overlap_iou = 0.0
        warning = "Damaged vehicle part could not be confidently identified."
    else:
        part_name = damaged_part["part"]
        part_confidence = damaged_part["part_confidence"]
        overlap_iou = damaged_part["overlap_iou"]
        warning = None

    return {
        "damage_type": damage_detection["damage_type"],
        "damage_confidence": damage_detection["confidence"],
        "damaged_part": part_name,
        "part_confidence": part_confidence,
        "overlap_iou": overlap_iou,
        "warning": warning
    }