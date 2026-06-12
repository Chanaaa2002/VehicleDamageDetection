from pathlib import Path
from ultralytics import YOLO
from fusion_logic import fuse_predictions
from repair_rules import decide_repair_action
from cost_estimator import estimate_repair_cost


# Main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Temporary vehicle details for testing cost estimation.
# In the final UI, the user will enter brand, model, and year.
# Important: The system does NOT detect brand/model from image yet.
TEST_VEHICLE = {
    "brand": "Honda",
    "model": "Vezel",
    "year": 2015,
}

# Archive5 label meaning was visually inferred from sample images.
# Use archive5 only as supporting damage-type evidence, not main decision.
ARCHIVE5_LABEL_MAP = {
    "1": "crack",
    "2": "scratch",
    "3": "tire flat",
    "4": "dent",
    "5": "glass shatter",
    "6": "lamp broken",
}

MODEL_PATHS = {
    # Optional damage/whole support
    "damage_classifier": PROJECT_ROOT / "models" / "classification" / "archive1_best.pt",

    # Severity model
    "severity_classifier": PROJECT_ROOT / "models" / "classification" / "archive2_best.pt",

    # Archive5 support model
    "archive5_damage_type_support": PROJECT_ROOT / "models" / "classification" / "archive5_best.pt",

    # Main damage detector
    "damage_detector": PROJECT_ROOT / "models" / "detection" / "cardd_coco_best.pt",

    # Damage area segmentation
    "damage_segmenter": PROJECT_ROOT / "models" / "segmentation" / "cardd_sod_best.pt",

    # Vehicle part segmentation
    "carpart_segmenter": PROJECT_ROOT / "models" / "segmentation" / "carparts_best.pt",

    # Weak damaged-part support model
    "damaged_part_support": PROJECT_ROOT / "models" / "detection" / "damaged_parts_best.pt",
}


def load_models():
    """
    Loads all selected trained models.
    This does not train anything.
    It only opens the saved .pt files.
    """
    models = {}

    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        models[name] = YOLO(str(path))

    return models


def predict_classification(model, image_path):
    """
    Runs a classification model.

    Example outputs:
    archive1 -> 00-damage / 01-whole
    archive2 -> 01-minor / 02-moderate / 03-severe
    archive5 -> 1 / 2 / 3 / 4 / 5 / 6
    """
    result = model.predict(source=str(image_path), verbose=False)[0]

    class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    class_name = str(model.names[class_id])

    return {
        "class_name": class_name,
        "confidence": round(confidence, 4),
    }


def predict_detection_or_segmentation(model, image_path, conf=0.10, imgsz=640):
    """
    Runs detection or segmentation models.

    Detection gives:
    class name + confidence + box

    Segmentation gives:
    class name + confidence + box + mask availability
    """
    result = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        verbose=False
    )[0]

    predictions = []

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "predictions": [],
            "has_masks": False,
            "mask_count": 0,
        }

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        predictions.append({
            "class_name": model.names[class_id],
            "confidence": round(confidence, 4),
            "box": [round(v, 2) for v in xyxy],
        })

    has_masks = result.masks is not None
    mask_count = len(result.masks.data) if has_masks else 0

    return {
        "predictions": predictions,
        "has_masks": has_masks,
        "mask_count": mask_count,
    }


def convert_archive5_prediction(raw_prediction):
    """
    Converts archive5 numeric class into readable damage type.

    Example:
    class 4 -> dent
    class 6 -> lamp broken
    """
    numeric_label = raw_prediction["class_name"]
    readable_label = ARCHIVE5_LABEL_MAP.get(numeric_label, "unknown")

    return {
        "class_name": readable_label,
        "original_label": numeric_label,
        "confidence": raw_prediction["confidence"],
        "note": "Archive5 label mapping is visually inferred and used only as support."
    }


def run_all_models(image_path, models=None):
    """
    Runs all selected models on one image.

    This function only collects model outputs.
    It does not make the final decision.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if models is None:
        models = load_models()

    outputs = {}

    outputs["damage_classifier"] = predict_classification(
        models["damage_classifier"],
        image_path
    )

    outputs["severity_classifier"] = predict_classification(
        models["severity_classifier"],
        image_path
    )

    raw_archive5 = predict_classification(
        models["archive5_damage_type_support"],
        image_path
    )

    outputs["archive5_damage_type_support"] = convert_archive5_prediction(raw_archive5)

    outputs["damage_detector"] = predict_detection_or_segmentation(
        models["damage_detector"],
        image_path
    )

    outputs["damage_segmenter"] = predict_detection_or_segmentation(
        models["damage_segmenter"],
        image_path
    )

    outputs["carpart_segmenter"] = predict_detection_or_segmentation(
        models["carpart_segmenter"],
        image_path
    )

    outputs["damaged_part_support"] = predict_detection_or_segmentation(
        models["damaged_part_support"],
        image_path
    )

    return outputs


def get_best_prediction(output):
    """
    Gets highest confidence detection/segmentation prediction.
    """
    predictions = output.get("predictions", [])

    if not predictions:
        return None

    return max(predictions, key=lambda x: x["confidence"])


def clean_display_text(text):
    """
    Final cleanup before printing text in terminal/UI.
    This fixes small missing-space problems in reason text.
    """
    text = str(text)

    # Strong direct fixes
    text = text.replace("includehidden", "include hidden")
    text = text.replace("Includehidden", "Include hidden")
    text = text.replace("INCLUDEHIDDEN", "INCLUDE HIDDEN")
    text = text.replace("may includehidden", "may include hidden")
    text = text.replace("may Includehidden", "may include hidden")

    fixes = {
        "Partprediction": "Part prediction",
        "partprediction": "Part prediction",
        "PARTPREDICTION": "Part prediction",

        "damagemay": "damage may",
        "Damagemay": "Damage may",
        "DAMAGEMAY": "Damage may",

        "forsafety": "for safety",
        "Forsafety": "For safety",
        "FORSAFETY": "For safety",

        "recommendedbecause": "recommended because",
        "Recommendedbecause": "Recommended because",

        "sourcesmay": "sources may",
        "model sourcesmay": "model sources may",

        "publiclistings": "public listings",
        "public marketreference": "public market reference",

        "include hidden or": "include hidden or",
    }

    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)

    return " ".join(text.split())


def clean_output_for_display(value):
    """
    Cleans strings inside dictionaries/lists before printing.
    This does not change the model logic.
    It only makes terminal output readable.
    """
    if isinstance(value, dict):
        return {
            key: clean_output_for_display(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            clean_output_for_display(item)
            for item in value
        ]

    if isinstance(value, str):
        return clean_display_text(value)

    return value


def print_cost_summary(cost_result, repair_result):
    """
    Prints cost estimation in a readable way.

    Shows:
    - vehicle details
    - repair action
    - replacement option
    - part condition
    - whether replacement is required
    - primary estimate
    - possible full repair estimate
    """
    cost_result = clean_output_for_display(cost_result)
    repair_result = clean_output_for_display(repair_result)

    vehicle = cost_result.get("vehicle", {})
    repair_decision = cost_result.get("repair_decision", {})
    primary_estimate = cost_result.get("primary_estimate", {})
    full_estimate = cost_result.get("possible_full_repair_estimate", {})

    print("\nCOST ESTIMATION:")
    print(
        "Vehicle:",
        vehicle.get("brand", "Unknown"),
        vehicle.get("model", "Unknown"),
        vehicle.get("year", "Unknown")
    )

    print("Currency:", cost_result.get("currency", "LKR"))
    print("Damage context:", cost_result.get("damage_context", "unknown"))
    print("Primary damaged part:", cost_result.get("primary_damaged_part", "unknown"))
    print("Supported vehicle:", cost_result.get("supported_vehicle", False))

    print("\nRepair decision:")
    print("Recommended action:", repair_decision.get("recommended_action", repair_result.get("recommended_action", "unknown")))
    print("Repair category:", repair_decision.get("repair_category", repair_result.get("repair_category", "unknown")))
    print("Replacement option:", repair_decision.get("replacement_option", repair_result.get("replacement_option", "unknown")))
    print("Inspection required:", repair_decision.get("inspection_required", repair_result.get("inspection_required", True)))

    print("\nPrimary estimate:")
    print("Total range:", primary_estimate.get("total_range", "Rs. 0 - Rs. 0"))

    if primary_estimate.get("matched_parts"):
        print("Matched primary part:")
        for part in primary_estimate["matched_parts"]:
            print("-", part.get("detected_part", "unknown"), "=>", part.get("part", "unknown"))
            print("  Condition used:", part.get("condition", "unknown"))
            print("  Replacement required:", part.get("replacement_required", "unknown"))
            print("  Part action:", part.get("part_action", "unknown"))
            print("  Recommended part condition:", part.get("recommended_part_condition", "unknown"))
            print("  Estimate:", part.get("total_range", "Rs. 0 - Rs. 0"))

    if primary_estimate.get("missing_parts"):
        print("Missing primary parts:", primary_estimate["missing_parts"])

    print("\nPossible full repair estimate:")
    print("Total range:", full_estimate.get("total_range", "Rs. 0 - Rs. 0"))

    if full_estimate.get("matched_parts"):
        print("Matched possible affected parts:")
        for part in full_estimate["matched_parts"]:
            print("-", part.get("detected_part", "unknown"), "=>", part.get("part", "unknown"))
            print("  Condition used:", part.get("condition", "unknown"))
            print("  Replacement required:", part.get("replacement_required", "unknown"))
            print("  Part action:", part.get("part_action", "unknown"))
            print("  Recommended part condition:", part.get("recommended_part_condition", "unknown"))
            print("  Estimate:", part.get("total_range", "Rs. 0 - Rs. 0"))

    if full_estimate.get("missing_parts"):
        print("Missing possible affected parts:", full_estimate["missing_parts"])

    print("\nCost notes:")
    for note in cost_result.get("cost_estimation_note", []):
        print("-", note)


def print_simple_summary(image_name, results):
    """
    Prints readable model outputs, final fused result, repair recommendation,
    and Sri Lankan cost estimation.
    """
    print("\n===================================================")
    print("IMAGE:", image_name)
    print("===================================================")

    print("Damage / Whole:", results["damage_classifier"])
    print("Severity:", results["severity_classifier"])
    print("Archive5 damage-type support:", results["archive5_damage_type_support"])

    print("Best damage detection:", get_best_prediction(results["damage_detector"]))
    print("Best damage segmentation:", get_best_prediction(results["damage_segmenter"]))
    print("Best car-part prediction:", get_best_prediction(results["carpart_segmenter"]))
    print("Best damaged-part support:", get_best_prediction(results["damaged_part_support"]))

    print("Damage mask available:", results["damage_segmenter"]["has_masks"])
    print("Car-part mask available:", results["carpart_segmenter"]["has_masks"])

    # Step 1: Fusion logic combines all AI model outputs.
    fused_result = fuse_predictions(results)

    print("\nFINAL FUSED RESULT:")
    print(clean_output_for_display(fused_result))

    # Step 2: Repair rules decide what action should be recommended.
    repair_result = decide_repair_action(fused_result)

    print("\nREPAIR RECOMMENDATION:")
    print(clean_output_for_display(repair_result))

    # Step 3: Estimate repair cost using Sri Lankan price reference CSV.
    cost_result = estimate_repair_cost(
        fused_result=fused_result,
        repair_result=repair_result,
        vehicle_brand=TEST_VEHICLE["brand"],
        vehicle_model=TEST_VEHICLE["model"],
        vehicle_year=TEST_VEHICLE["year"]
    )

    print_cost_summary(cost_result, repair_result)


if __name__ == "__main__":
    TEST_IMAGES_FOLDER = PROJECT_ROOT / "test_images"

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    image_paths = [
        p for p in TEST_IMAGES_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    image_paths = sorted(image_paths)

    print("Test images folder:", TEST_IMAGES_FOLDER)
    print("Images found:", len(image_paths))

    if not image_paths:
        print("No images found. Add images into test_images folder.")
    else:
        models = load_models()

        for image_path in image_paths:
            results = run_all_models(image_path, models=models)
            print_simple_summary(image_path.name, results)