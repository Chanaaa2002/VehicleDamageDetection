"""
api_server.py

FastAPI backend for the Vehicle Damage Detection system.

This API connects the React frontend with:
- YOLO model inference
- general object validation
- clean vehicle validation
- vehicle damage input validation
- fusion logic
- repair recommendation
- Sri Lankan repair cost estimation

Endpoints:
GET  /
GET  /health
GET  /options
POST /analyze
POST /estimate-confirmed
"""

import sys
import uuid
import shutil
from pathlib import Path
from typing import List, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Make sure src folder imports work correctly.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from model_inference import load_models, run_all_models, get_best_prediction
from fusion_logic import fuse_predictions
from repair_rules import decide_repair_action
from cost_estimator import estimate_repair_cost


app = FastAPI(
    title="Vehicle Damage Detection API",
    description="AI-powered vehicle damage detection, validation, repair recommendation, user confirmation, and Sri Lankan repair cost estimation API.",
    version="1.4.0"
)


# Allow React frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = PROJECT_ROOT / "uploaded_images"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MODELS = None
GENERAL_OBJECT_MODEL = None


SUPPORTED_VEHICLES = [
    {"brand": "Honda", "model": "Vezel", "default_year": 2015},
    {"brand": "Toyota", "model": "Aqua", "default_year": 2015},
    {"brand": "Toyota", "model": "Vitz", "default_year": 2016},
    {"brand": "Suzuki", "model": "Alto", "default_year": 2018},
    {"brand": "Honda", "model": "Fit", "default_year": 2015},
    {"brand": "Toyota", "model": "Prius", "default_year": 2013},
    {"brand": "Suzuki", "model": "Wagon R", "default_year": 2017},
]

DAMAGE_TYPES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]

SEVERITY_LEVELS = [
    "minor",
    "moderate",
    "severe",
]

DAMAGE_PARTS = [
    "front bumper",
    "rear bumper",
    "windshield",
    "lamp/light",
    "rear lamp/light",
    "headlight/front lamp",
    "wheel/tire",
    "hood/front shell",
    "tailgate",
    "fender/body panel",
    "front body panel",
    "rear body panel",
    "body shell",
    "internal hidden components",
]


class ConfirmedEstimateRequest(BaseModel):
    """
    Request body for confirmed cost estimation.

    React frontend sends this after the user confirms or edits
    the AI prediction.
    """

    brand: str
    model: str
    year: int

    damage_type: str
    severity: str
    damaged_part: str

    possible_affected_parts: List[str] = Field(default_factory=list)

    force_manual_inspection: bool = False


def get_loaded_models():
    """
    Loads vehicle-damage YOLO models only once.
    First request may take some time.
    After that, the same models are reused.
    """
    global MODELS

    if MODELS is None:
        print("Loading vehicle damage models...")
        MODELS = load_models()
        print("Vehicle damage models loaded successfully.")

    return MODELS


def get_general_object_model():
    """
    Loads a small general YOLO model for safety validation.

    This model is used only to detect obvious non-vehicle images such as
    person, dog, laptop, chair, etc. It also helps detect full vehicle photos.
    """
    global GENERAL_OBJECT_MODEL

    if GENERAL_OBJECT_MODEL is None:
        try:
            from ultralytics import YOLO

            print("Loading general object validation model...")
            GENERAL_OBJECT_MODEL = YOLO("yolov8n.pt")
            print("General object validation model loaded successfully.")
        except Exception as error:
            print(f"General object validation model could not be loaded: {error}")
            GENERAL_OBJECT_MODEL = False

    return GENERAL_OBJECT_MODEL


def make_json_safe(value: Any):
    """
    Converts Python objects into JSON-safe values.
    This prevents JSON response errors from numpy/path/custom objects.
    """
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

    except Exception:
        pass

    return value


def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Saves uploaded image into uploaded_images folder.
    """
    original_name = upload_file.filename or "uploaded_image.jpg"
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {extension}. Allowed: jpg, jpeg, png, webp"
        )

    unique_name = f"{uuid.uuid4().hex}{extension}"
    save_path = UPLOAD_DIR / unique_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return save_path


def build_model_output_summary(raw_results):
    """
    Creates a small readable summary of raw model outputs.
    Useful for frontend display/debugging.
    """
    return {
        "damage_classifier": raw_results.get("damage_classifier"),
        "severity_classifier": raw_results.get("severity_classifier"),
        "archive5_damage_type_support": raw_results.get("archive5_damage_type_support"),
        "best_damage_detection": get_best_prediction(raw_results.get("damage_detector", {})),
        "best_damage_segmentation": get_best_prediction(raw_results.get("damage_segmenter", {})),
        "best_car_part_prediction": get_best_prediction(raw_results.get("carpart_segmenter", {})),
        "best_damaged_part_support": get_best_prediction(raw_results.get("damaged_part_support", {})),
        "damage_mask_available": raw_results.get("damage_segmenter", {}).get("has_masks", False),
        "carpart_mask_available": raw_results.get("carpart_segmenter", {}).get("has_masks", False),
    }


def get_confidence(prediction):
    """
    Safely reads confidence from a prediction dictionary.
    """
    if not prediction:
        return 0.0

    try:
        return float(prediction.get("confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def validate_general_vehicle_context(image_path: Path):
    """
    Uses a general COCO object detector to:
    1. Reject obvious non-vehicle images.
    2. Detect whether the uploaded image is a whole-vehicle view.

    This helps with:
    - person / dog / laptop / room rejection
    - clean whole-vehicle photos that should not go to repair-cost estimation
    """

    model = get_general_object_model()

    if model is False:
        return {
            "valid": True,
            "reason_code": "general_validator_unavailable",
            "message": "General object validator unavailable. Continuing with custom validation.",
            "details": {}
        }

    try:
        results = model(str(image_path), conf=0.25, verbose=False)
    except Exception as error:
        return {
            "valid": True,
            "reason_code": "general_validator_error",
            "message": f"General object validator error: {error}. Continuing with custom validation.",
            "details": {}
        }

    vehicle_classes = {
        "car",
        "motorcycle",
        "bus",
        "truck",
    }

    obvious_non_vehicle_classes = {
        "person",
        "bicycle",
        "dog",
        "cat",
        "bird",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "handbag",
        "suitcase",
        "bottle",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
    }

    detected_objects = []
    vehicle_confidence = 0.0
    non_vehicle_confidence = 0.0
    strongest_non_vehicle = None
    largest_vehicle_area_ratio = 0.0

    for result in results:
        names = result.names

        if result.boxes is None:
            continue

        image_height, image_width = result.orig_shape[:2]
        image_area = max(float(image_width * image_height), 1.0)

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(names[class_id]).lower()
            confidence = float(box.conf[0])

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            box_width = max(float(x2 - x1), 0.0)
            box_height = max(float(y2 - y1), 0.0)
            area_ratio = (box_width * box_height) / image_area

            detected_objects.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "area_ratio": round(area_ratio, 4),
            })

            if class_name in vehicle_classes:
                vehicle_confidence = max(vehicle_confidence, confidence)
                largest_vehicle_area_ratio = max(largest_vehicle_area_ratio, area_ratio)

            if class_name in obvious_non_vehicle_classes:
                if confidence > non_vehicle_confidence:
                    non_vehicle_confidence = confidence
                    strongest_non_vehicle = class_name

    whole_vehicle_view = (
        vehicle_confidence >= 0.35
        and largest_vehicle_area_ratio >= 0.18
    )

    if vehicle_confidence >= 0.35:
        return {
            "valid": True,
            "reason_code": "vehicle_context_detected",
            "message": "Vehicle context detected.",
            "details": {
                "vehicle_confidence": round(vehicle_confidence, 4),
                "vehicle_area_ratio": round(largest_vehicle_area_ratio, 4),
                "whole_vehicle_view": whole_vehicle_view,
                "non_vehicle_confidence": round(non_vehicle_confidence, 4),
                "strongest_non_vehicle": strongest_non_vehicle,
                "detected_objects": detected_objects[:10],
            }
        }

    if vehicle_confidence < 0.35 and non_vehicle_confidence >= 0.45:
        return {
            "valid": False,
            "reason_code": "obvious_non_vehicle_image",
            "message": (
                f"This image appears to contain '{strongest_non_vehicle}', not a vehicle damage area. "
                "Please upload a clear photo of a damaged vehicle."
            ),
            "details": {
                "vehicle_confidence": round(vehicle_confidence, 4),
                "vehicle_area_ratio": round(largest_vehicle_area_ratio, 4),
                "whole_vehicle_view": whole_vehicle_view,
                "non_vehicle_confidence": round(non_vehicle_confidence, 4),
                "strongest_non_vehicle": strongest_non_vehicle,
                "detected_objects": detected_objects[:10],
            }
        }

    return {
        "valid": True,
        "reason_code": "no_obvious_non_vehicle_object",
        "message": "No obvious non-vehicle object detected. Continuing with custom validation.",
        "details": {
            "vehicle_confidence": round(vehicle_confidence, 4),
            "vehicle_area_ratio": round(largest_vehicle_area_ratio, 4),
            "whole_vehicle_view": whole_vehicle_view,
            "non_vehicle_confidence": round(non_vehicle_confidence, 4),
            "strongest_non_vehicle": strongest_non_vehicle,
            "detected_objects": detected_objects[:10],
        }
    }


def validate_vehicle_damage_input(raw_results, general_validation=None):
    """
    Checks whether the uploaded image is suitable for vehicle damage analysis.

    Strong rule:
    - Non-vehicle images are rejected.
    - Clean whole-vehicle images are rejected.
    - A vehicle photo is accepted only when damage evidence is reliable.

    This prevents full clean vehicle photos from being forced into
    glass shatter / dent / scratch predictions.
    """

    general_details = (general_validation or {}).get("details", {})
    whole_vehicle_view = bool(general_details.get("whole_vehicle_view", False))
    vehicle_area_ratio = float(general_details.get("vehicle_area_ratio", 0.0) or 0.0)
    general_vehicle_confidence = float(general_details.get("vehicle_confidence", 0.0) or 0.0)

    damage_classifier = raw_results.get("damage_classifier", {})
    damage_class_name = str(damage_classifier.get("class_name", "")).lower()
    damage_class_conf = get_confidence(damage_classifier)

    severity_classifier = raw_results.get("severity_classifier", {})
    severity_name = str(severity_classifier.get("class_name", "")).lower()
    severity_conf = get_confidence(severity_classifier)

    archive5_support = raw_results.get("archive5_damage_type_support", {})
    archive5_name = str(archive5_support.get("class_name", "")).lower()
    archive5_conf = get_confidence(archive5_support)

    best_damage_detection = get_best_prediction(raw_results.get("damage_detector", {}))
    best_damage_segmentation = get_best_prediction(raw_results.get("damage_segmenter", {}))
    best_car_part = get_best_prediction(raw_results.get("carpart_segmenter", {}))
    best_damaged_part_support = get_best_prediction(raw_results.get("damaged_part_support", {}))

    damage_detection_conf = get_confidence(best_damage_detection)
    damage_segmentation_conf = get_confidence(best_damage_segmentation)
    car_part_conf = get_confidence(best_car_part)
    damaged_part_support_conf = get_confidence(best_damaged_part_support)

    best_damage_name = str(best_damage_detection.get("class_name", "")).lower() if best_damage_detection else ""
    best_segment_name = str(best_damage_segmentation.get("class_name", "")).lower() if best_damage_segmentation else ""
    best_car_part_name = str(best_car_part.get("class_name", "")).lower() if best_car_part else ""
    best_damaged_part_name = str(best_damaged_part_support.get("class_name", "")).lower() if best_damaged_part_support else ""

    vehicle_signal_confidence = max(car_part_conf, damaged_part_support_conf, general_vehicle_confidence)
    damage_signal_confidence = max(damage_detection_conf, damage_segmentation_conf)

    vehicle_signal = (
        car_part_conf >= 0.45
        or damaged_part_support_conf >= 0.35
        or general_vehicle_confidence >= 0.35
    )

    classifier_says_whole = (
        "whole" in damage_class_name
        or "undamaged" in damage_class_name
        or "no damage" in damage_class_name
        or "normal" in damage_class_name
    )

    classifier_says_damaged = (
        "damage" in damage_class_name
        or "damaged" in damage_class_name
    ) and not classifier_says_whole

    strong_damage_detector = damage_detection_conf >= 0.75
    strong_damage_segmenter = damage_segmentation_conf >= 0.75

    medium_damage_detector = damage_detection_conf >= 0.60
    medium_damage_segmenter = damage_segmentation_conf >= 0.60
    medium_archive5_damage = archive5_conf >= 0.75 and archive5_name not in ["", "unknown"]

    damage_evidence_count = 0

    if medium_damage_detector:
        damage_evidence_count += 1

    if medium_damage_segmenter:
        damage_evidence_count += 1

    if classifier_says_damaged and damage_class_conf >= 0.85:
        damage_evidence_count += 1

    if medium_archive5_damage:
        damage_evidence_count += 1

    reliable_damage_signal = (
        strong_damage_detector
        or strong_damage_segmenter
        or damage_evidence_count >= 2
    )

    glass_or_windshield_false_positive_risk = (
        whole_vehicle_view
        and (
            "glass" in best_damage_name
            or "glass" in archive5_name
            or "windshield" in best_car_part_name
            or "windshield" in best_damaged_part_name
            or "front_glass" in best_car_part_name
        )
        and damage_detection_conf < 0.90
        and damage_segmentation_conf < 0.90
    )

    weak_minor_damage_on_full_vehicle = (
        whole_vehicle_view
        and ("minor" in severity_name or severity_conf < 0.70)
        and damage_detection_conf < 0.90
        and damage_segmentation_conf < 0.90
    )

    validation_details = {
        "vehicle_signal_confidence": round(vehicle_signal_confidence, 4),
        "damage_signal_confidence": round(damage_signal_confidence, 4),
        "damage_evidence_count": damage_evidence_count,
        "reliable_damage_signal": reliable_damage_signal,
        "whole_vehicle_view": whole_vehicle_view,
        "vehicle_area_ratio": round(vehicle_area_ratio, 4),
        "general_vehicle_confidence": round(general_vehicle_confidence, 4),
        "glass_or_windshield_false_positive_risk": glass_or_windshield_false_positive_risk,
        "weak_minor_damage_on_full_vehicle": weak_minor_damage_on_full_vehicle,
        "damage_classifier": {
            "class_name": damage_class_name,
            "confidence": round(damage_class_conf, 4),
        },
        "severity_classifier": {
            "class_name": severity_name,
            "confidence": round(severity_conf, 4),
        },
        "archive5_damage_type_support": {
            "class_name": archive5_name,
            "confidence": round(archive5_conf, 4),
        },
        "best_damage_detection": {
            "class_name": best_damage_name,
            "confidence": round(damage_detection_conf, 4),
        },
        "best_damage_segmentation": {
            "class_name": best_segment_name,
            "confidence": round(damage_segmentation_conf, 4),
        },
        "best_car_part_prediction": {
            "class_name": best_car_part_name,
            "confidence": round(car_part_conf, 4),
        },
        "best_damaged_part_support": {
            "class_name": best_damaged_part_name,
            "confidence": round(damaged_part_support_conf, 4),
        }
    }

    if not vehicle_signal and not reliable_damage_signal:
        return {
            "valid": False,
            "reason_code": "not_vehicle_damage_image",
            "message": "This image does not appear to be a vehicle damage image. Please upload a clear photo of the damaged vehicle.",
            "details": validation_details,
        }

    if not vehicle_signal and reliable_damage_signal:
        return {
            "valid": False,
            "reason_code": "vehicle_part_not_confirmed",
            "message": "Damage-like patterns were detected, but the system could not confirm that this is a vehicle. Please upload a clearer vehicle damage photo.",
            "details": validation_details,
        }

    if vehicle_signal and not reliable_damage_signal:
        return {
            "valid": False,
            "reason_code": "no_clear_damage_detected",
            "message": "A vehicle is visible, but no clear damage was detected. Please upload a closer photo of the damaged area.",
            "details": validation_details,
        }

    if (
        whole_vehicle_view
        and (
            glass_or_windshield_false_positive_risk
            or weak_minor_damage_on_full_vehicle
        )
    ):
        return {
            "valid": False,
            "reason_code": "vehicle_no_damage",
            "message": "A vehicle is visible, but no clear external damage was detected. Please upload a closer photo of the damaged area.",
            "details": validation_details,
        }

    if (
        classifier_says_whole
        and damage_class_conf >= 0.75
        and not strong_damage_detector
        and not strong_damage_segmenter
    ):
        return {
            "valid": False,
            "reason_code": "vehicle_no_damage",
            "message": "The image appears to show a vehicle, but no reliable damage was identified.",
            "details": validation_details,
        }

    return {
        "valid": True,
        "reason_code": "valid_vehicle_damage_image",
        "message": "Vehicle damage image accepted for analysis.",
        "details": validation_details,
    }


def build_confirmation_status(fused_result, repair_result):
    """
    Tells frontend whether user confirmation is required.

    Confirmation is required when:
    - model confidence is low
    - part prediction is uncertain
    - severity is uncertain
    - multiple parts may affect cost
    - manual inspection is recommended
    """

    confirmation_reasons = []

    if fused_result.get("confirmation_required", False):
        confirmation_reasons.extend(fused_result.get("confirmation_reasons", []))

    if fused_result.get("severity_uncertain", False):
        confirmation_reasons.append("Severity confidence is low.")

    if fused_result.get("part_confidence_level") in ["low", "medium"]:
        confirmation_reasons.append("Damaged part prediction may need confirmation.")

    if repair_result.get("inspection_required", False):
        confirmation_reasons.append("Repair recommendation requires inspection before final quote.")

    possible_parts = fused_result.get("possible_affected_parts", [])

    if len(possible_parts) > 1:
        confirmation_reasons.append("Multiple affected parts may change the final cost.")

    unique_reasons = []
    for reason in confirmation_reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    return {
        "required": len(unique_reasons) > 0,
        "reasons": unique_reasons,
        "message": (
            "Please confirm or edit the AI result before final cost estimation."
            if unique_reasons
            else "AI result is confident, but user can still edit before final cost estimation."
        )
    }


def analyze_single_image(image_path: Path, original_filename: str, brand: str, model: str, year: int):
    """
    Runs full AI pipeline for one image.

    Safety flow:
    1. Run general object validation
    2. Reject obvious non-vehicle images
    3. Run vehicle-damage models
    4. Validate vehicle/damage evidence
    5. Continue to fusion, repair rules, and cost estimation only if valid
    """

    general_validation = validate_general_vehicle_context(image_path)

    if not general_validation["valid"]:
        return make_json_safe({
            "filename": original_filename,
            "saved_image_path": str(image_path),
            "input_validation": general_validation,
            "general_validation": general_validation,
            "model_output_summary": {},
            "error": general_validation["message"],
            "next_step": general_validation["reason_code"],
        })

    models = get_loaded_models()

    raw_results = run_all_models(image_path, models=models)

    input_validation = validate_vehicle_damage_input(raw_results, general_validation)

    if not input_validation["valid"]:
        return make_json_safe({
            "filename": original_filename,
            "saved_image_path": str(image_path),
            "input_validation": input_validation,
            "general_validation": general_validation,
            "model_output_summary": build_model_output_summary(raw_results),
            "error": input_validation["message"],
            "next_step": input_validation["reason_code"],
        })

    fused_result = fuse_predictions(raw_results)

    repair_result = decide_repair_action(fused_result)

    cost_result = estimate_repair_cost(
        fused_result=fused_result,
        repair_result=repair_result,
        vehicle_brand=brand,
        vehicle_model=model,
        vehicle_year=year
    )

    confirmation_status = build_confirmation_status(
        fused_result=fused_result,
        repair_result=repair_result
    )

    response = {
        "filename": original_filename,
        "saved_image_path": str(image_path),
        "input_validation": input_validation,
        "general_validation": general_validation,
        "model_output_summary": build_model_output_summary(raw_results),

        "final_fused_result": fused_result,
        "repair_recommendation": repair_result,
        "cost_estimation": cost_result,

        "confirmation_status": confirmation_status,
        "next_step": (
            "confirm_result_before_final_estimate"
            if confirmation_status["required"]
            else "user_can_accept_or_edit_before_final_estimate"
        )
    }

    return make_json_safe(response)


def build_user_confirmed_fused_result(request: ConfirmedEstimateRequest):
    """
    Builds a fused_result-style dictionary from user-confirmed data.

    This lets us reuse repair_rules.py and cost_estimator.py
    without changing those modules.
    """

    possible_parts = request.possible_affected_parts.copy()

    if request.damaged_part and request.damaged_part not in possible_parts:
        possible_parts.insert(0, request.damaged_part)

    fused_result = {
        "damage_detected": True,

        "damage_type": request.damage_type,
        "damage_confidence": 1.0,
        "damage_source": "user_confirmed",

        "archive5_support": "user_confirmed",
        "archive5_confidence": 1.0,

        "damage_notes": [
            "Damage details were confirmed or edited by the user before final cost estimation."
        ],

        "severity": request.severity,
        "severity_confidence": 1.0,
        "severity_uncertain": False,

        "damaged_part": request.damaged_part,
        "part_confidence_level": "user_confirmed",
        "part_source": "user_confirmed",

        "part_notes": [
            "Damaged part was confirmed or edited by the user before final cost estimation."
        ],

        "possible_affected_parts": possible_parts,

        "confirmation_required": request.force_manual_inspection,
        "confirmation_reasons": (
            ["User requested manual inspection before final cost confirmation."]
            if request.force_manual_inspection
            else []
        ),

        "damage_mask_available": False,
        "carpart_mask_available": False,
    }

    return fused_result


@app.get("/")
def root():
    return {
        "message": "Vehicle Damage Detection API is running.",
        "endpoints": {
            "health": "/health",
            "options": "/options",
            "analyze": "/analyze",
            "estimate_confirmed": "/estimate-confirmed"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running."
    }


@app.get("/options")
def get_options():
    """
    Gives frontend dropdown options.
    """
    return {
        "vehicles": SUPPORTED_VEHICLES,
        "damage_types": DAMAGE_TYPES,
        "severity_levels": SEVERITY_LEVELS,
        "damage_parts": DAMAGE_PARTS,
    }


@app.post("/analyze")
async def analyze_vehicle_damage(
    brand: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Analyze one or multiple vehicle damage images.

    React frontend sends:
    - brand
    - model
    - year
    - files

    This returns:
    - AI prediction
    - preliminary repair recommendation
    - preliminary cost estimate
    - confirmation status

    If the image is not suitable, it returns:
    - error message
    - input_validation details
    - no cost estimation
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one image."
        )

    if year < 1990 or year > 2035:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid vehicle year."
        )

    results = []

    for upload_file in files:
        saved_path = save_upload_file(upload_file)

        try:
            result = analyze_single_image(
                image_path=saved_path,
                original_filename=upload_file.filename or saved_path.name,
                brand=brand,
                model=model,
                year=year
            )

            results.append(result)

        except Exception as error:
            results.append({
                "filename": upload_file.filename,
                "error": str(error)
            })

    return make_json_safe({
        "vehicle": {
            "brand": brand,
            "model": model,
            "year": year
        },
        "image_count": len(files),
        "results": results
    })


@app.post("/estimate-confirmed")
def estimate_confirmed_damage(request: ConfirmedEstimateRequest):
    """
    Calculates repair recommendation and cost estimation using
    user-confirmed or user-edited damage details.

    This endpoint is used after the AI result is shown to the user.

    Frontend flow:
    1. User uploads image
    2. /analyze returns AI prediction
    3. User confirms or edits damage details
    4. /estimate-confirmed returns final cost estimate
    """

    if request.year < 1990 or request.year > 2035:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid vehicle year."
        )

    if not request.damage_type:
        raise HTTPException(
            status_code=400,
            detail="Damage type is required."
        )

    if not request.severity:
        raise HTTPException(
            status_code=400,
            detail="Severity is required."
        )

    if not request.damaged_part:
        raise HTTPException(
            status_code=400,
            detail="Damaged part is required."
        )

    confirmed_fused_result = build_user_confirmed_fused_result(request)

    repair_result = decide_repair_action(confirmed_fused_result)

    if request.force_manual_inspection:
        repair_result["inspection_required"] = True

    cost_result = estimate_repair_cost(
        fused_result=confirmed_fused_result,
        repair_result=repair_result,
        vehicle_brand=request.brand,
        vehicle_model=request.model,
        vehicle_year=request.year
    )

    return make_json_safe({
        "vehicle": {
            "brand": request.brand,
            "model": request.model,
            "year": request.year,
        },
        "confirmed_fused_result": confirmed_fused_result,
        "repair_recommendation": repair_result,
        "cost_estimation": cost_result,
        "confirmation_status": {
            "required": False,
            "message": "Final estimate generated using user-confirmed damage details."
        }
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )