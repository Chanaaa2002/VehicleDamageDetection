from pathlib import Path
from typing import List, Dict, Optional

from ultralytics import YOLO


# Main project folder
PROJECT_ROOT = Path(r"C:\Users\User\Desktop\Vehicle_Damage_Detection")

# Final selected trained models
DAMAGE_MODEL_PATH = PROJECT_ROOT / "models" / "detection" / "cardd_coco_best.pt"
CARPARTS_MODEL_PATH = PROJECT_ROOT / "models" / "segmentation" / "carparts_best.pt"


def load_models():
    """
    Load final trained YOLO models.

    Damage model:
    - Detects damage types such as dent, scratch, crack, glass shatter, lamp broken, tire flat.

    Carparts model:
    - Detects vehicle parts such as bumper, door, glass, hood, light, mirror, wheel.
    """

    if not DAMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Damage model not found: {DAMAGE_MODEL_PATH}")

    if not CARPARTS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Carparts model not found: {CARPARTS_MODEL_PATH}")

    damage_model = YOLO(str(DAMAGE_MODEL_PATH))
    carparts_model = YOLO(str(CARPARTS_MODEL_PATH))

    return damage_model, carparts_model


def extract_yolo_detections(result, model_names) -> List[Dict]:
    """
    Convert YOLO output into a simple list.

    YOLO gives output in its own format.
    This function converts it into this format:

    [
        {
            "class_name": "dent",
            "confidence": 0.76,
            "box": [x1, y1, x2, y2]
        }
    ]
    """

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class_name": model_names[class_id],
            "confidence": round(confidence, 4),
            "box": [round(value, 2) for value in xyxy]
        })

    return detections


def predict_damage_and_parts(image_path: str) -> Dict:
    """
    Run damage model and carparts model on one image.
    """

    damage_model, carparts_model = load_models()

    # Run damage detection model
    damage_result = damage_model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )[0]

    # Run carparts segmentation model
    carparts_result = carparts_model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )[0]

    damage_detections = extract_yolo_detections(
        damage_result,
        damage_model.names
    )

    carpart_detections = extract_yolo_detections(
        carparts_result,
        carparts_model.names
    )

    return {
        "damage_detections": damage_detections,
        "carpart_detections": carpart_detections
    }


def get_best_damage(damage_detections: List[Dict]) -> Optional[Dict]:
    """
    Select the highest-confidence damage prediction.

    If the model detects multiple damages, this function picks the strongest one first.
    Later we can handle multiple damages.
    """

    if not damage_detections:
        return None

    best = max(damage_detections, key=lambda item: item["confidence"])

    return {
        "damage_type": best["class_name"],
        "confidence": best["confidence"],
        "box": best["box"]
    }


def convert_parts_for_fusion(carpart_detections: List[Dict]) -> List[Dict]:
    """
    Convert carpart predictions into the format needed by fusion_logic.py.
    """

    converted = []

    for item in carpart_detections:
        converted.append({
            "part": item["class_name"],
            "confidence": item["confidence"],
            "box": item["box"]
        })

    return converted