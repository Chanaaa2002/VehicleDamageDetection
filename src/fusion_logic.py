"""
fusion_logic.py

This file combines all model outputs from model_inference.py.

model_inference.py = runs the AI models
fusion_logic.py    = combines model outputs and creates final decision

This version supports:
1. Primary damaged part
2. Possible affected parts
3. User confirmation for multi-part accident damage
"""


def get_best_prediction(model_output):
    """
    For detection/segmentation models, there can be many predictions.
    This gets the highest confidence one.
    """
    predictions = model_output.get("predictions", [])

    if not predictions:
        return None

    return max(predictions, key=lambda item: item["confidence"])


def clean_label(label):
    """
    Makes model labels easier to read.

    Example:
    front_bumper_damage -> front bumper damage
    03-severe -> 03 severe
    """
    if label is None:
        return "unknown"

    label = str(label).lower()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    return label.strip()


def clean_severity(label):
    """
    Converts archive2 output into simple severity.

    Example:
    01-minor -> minor
    02-moderate -> moderate
    03-severe -> severe
    """
    label = clean_label(label)

    if "minor" in label:
        return "minor"

    if "moderate" in label:
        return "moderate"

    if "severe" in label:
        return "severe"

    return "unknown"


def unique_list(items):
    """
    Removes duplicates while keeping order.
    """
    cleaned = []

    for item in items:
        item = clean_label(item)

        if item != "unknown" and item not in cleaned:
            cleaned.append(item)

    return cleaned


def is_glass_part(part_name):
    """
    Checks whether the detected damaged part is glass-related.
    """
    part_name = clean_label(part_name)

    glass_keywords = [
        "windshield",
        "front glass",
        "back glass",
        "rear glass",
        "glass"
    ]

    return any(keyword in part_name for keyword in glass_keywords)


def is_wheel_or_tire_part(part_name):
    """
    Checks whether the detected damaged part is wheel/tire-related.
    """
    part_name = clean_label(part_name)

    wheel_keywords = [
        "wheel",
        "tire",
        "tyre"
    ]

    return any(keyword in part_name for keyword in wheel_keywords)


def extract_part_from_damaged_part_support(label):
    """
    Converts damaged_parts model output into simple part name.

    Example:
    front_bumper_damage -> front bumper
    rear_bumper_damage  -> rear bumper
    windshield_damage   -> windshield
    wheel_tire_damage   -> wheel/tire
    """
    label = clean_label(label)

    if "front bumper" in label:
        return "front bumper"

    if "rear bumper" in label or "back bumper" in label:
        return "rear bumper"

    if "hood" in label or "bonnet" in label:
        return "hood"

    if "windshield" in label or "front glass" in label:
        return "windshield"

    if "back glass" in label or "rear glass" in label:
        return "back glass"

    if "wheel" in label or "tire" in label or "tyre" in label:
        return "wheel/tire"

    if "door" in label:
        return "door"

    if "lamp" in label or "light" in label or "headlight" in label:
        return "lamp/light"

    return "unknown"


def normalize_carpart_label(label):
    """
    Converts carparts model output into simple part name.

    Example:
    back_bumper -> rear bumper
    front_glass -> windshield
    wheel       -> wheel/tire
    """
    label = clean_label(label)

    if "front bumper" in label:
        return "front bumper"

    if "back bumper" in label or "rear bumper" in label:
        return "rear bumper"

    if "front glass" in label or "windshield" in label:
        return "windshield"

    if "back glass" in label or "rear glass" in label:
        return "back glass"

    if "tailgate" in label or "trunk" in label:
        return "tailgate"

    if "wheel" in label or "tire" in label or "tyre" in label:
        return "wheel/tire"

    if "door" in label:
        return "door"

    if "hood" in label or "bonnet" in label:
        return "hood"

    if "lamp" in label or "light" in label or "headlight" in label:
        return "lamp/light"

    return label if label else "unknown"


def choose_damaged_part(results):
    """
    Chooses the primary damaged part.

    Sources:
    1. carparts_best.pt
    2. damaged_parts_best.pt

    If they disagree or confidence is low, user confirmation is required.
    """
    carpart_prediction = get_best_prediction(results["carpart_segmenter"])
    support_prediction = get_best_prediction(results["damaged_part_support"])

    carpart_name = "unknown"
    carpart_conf = 0.0

    support_part = "unknown"
    support_conf = 0.0

    notes = []

    if carpart_prediction is not None:
        carpart_name = normalize_carpart_label(carpart_prediction["class_name"])
        carpart_conf = carpart_prediction["confidence"]
        notes.append(f"Car-part model suggests '{carpart_name}'.")

    if support_prediction is not None:
        support_part = extract_part_from_damaged_part_support(
            support_prediction["class_name"]
        )
        support_conf = support_prediction["confidence"]
        notes.append(f"Damaged-part support model suggests '{support_part}'.")

    # Case 1: both models agree
    if (
        carpart_name != "unknown"
        and support_part != "unknown"
        and carpart_name == support_part
    ):
        return {
            "damaged_part": carpart_name,
            "part_confidence_level": "high",
            "confirmation_required": False,
            "part_source": "carpart_and_support_agree",
            "carpart_suggestion": carpart_name,
            "support_part_suggestion": support_part,
            "part_notes": notes,
        }

    # Case 2: damaged-part support is reasonably strong
    if support_part != "unknown" and support_conf >= 0.50:
        return {
            "damaged_part": support_part,
            "part_confidence_level": "medium",
            "confirmation_required": True,
            "part_source": "damaged_part_support_model",
            "carpart_suggestion": carpart_name,
            "support_part_suggestion": support_part,
            "part_notes": notes + [
                "User confirmation is recommended because model sources may disagree."
            ],
        }

    # Case 3: damaged-part support is weak but useful
    if support_part != "unknown" and support_conf >= 0.10:
        return {
            "damaged_part": support_part,
            "part_confidence_level": "low",
            "confirmation_required": True,
            "part_source": "weak_damaged_part_support",
            "carpart_suggestion": carpart_name,
            "support_part_suggestion": support_part,
            "part_notes": notes + [
                "Support confidence is low, so user must confirm the damaged part."
            ],
        }

    # Case 4: only carpart model found something
    if carpart_name != "unknown" and carpart_conf >= 0.50:
        return {
            "damaged_part": carpart_name,
            "part_confidence_level": "low",
            "confirmation_required": True,
            "part_source": "carpart_segmentation_only",
            "carpart_suggestion": carpart_name,
            "support_part_suggestion": support_part,
            "part_notes": notes + [
                "Only car-part segmentation suggested the part, so user confirmation is required."
            ],
        }

    # Case 5: no useful part
    return {
        "damaged_part": "unknown",
        "part_confidence_level": "unknown",
        "confirmation_required": True,
        "part_source": "no_reliable_part_prediction",
        "carpart_suggestion": carpart_name,
        "support_part_suggestion": support_part,
        "part_notes": notes + [
            "No reliable damaged part was found. User must select the part manually."
        ],
    }


def choose_damage_type(results, part_info):
    """
    Chooses the final damage type.

    Main source:
    cardd_coco_best.pt

    Support source:
    archive5_best.pt

    Extra rule:
    If part evidence strongly supports glass damage, and archive5 strongly says glass shatter,
    the final damage type is corrected to glass shatter.
    """
    best_damage = get_best_prediction(results["damage_detector"])
    archive5 = results.get("archive5_damage_type_support", {})

    archive5_type = clean_label(archive5.get("class_name", "unknown"))
    archive5_conf = archive5.get("confidence", 0.0)

    damaged_part = clean_label(part_info.get("damaged_part", "unknown"))

    if best_damage is not None:
        main_type = clean_label(best_damage["class_name"])
        main_conf = best_damage["confidence"]
    else:
        main_type = "unknown"
        main_conf = 0.0

    notes = []

    if archive5_type != "unknown":
        if archive5_type == main_type:
            notes.append("Archive5 agrees with main damage detector.")
        else:
            notes.append(
                f"Archive5 suggests '{archive5_type}', but main detector suggests '{main_type}'."
            )

    # General rule:
    # If the part is glass and archive5 strongly says glass shatter,
    # trust the glass-shatter explanation.
    if (
        is_glass_part(damaged_part)
        and archive5_type == "glass shatter"
        and archive5_conf >= 0.85
    ):
        return {
            "damage_type": "glass shatter",
            "damage_confidence": archive5_conf,
            "damage_source": "archive5_part_consistency_override",
            "archive5_support": archive5_type,
            "archive5_confidence": archive5_conf,
            "damage_notes": notes + [
                "Damage type was corrected because the damaged part is glass and archive5 strongly detected glass shatter."
            ],
        }

    # Safety rule:
    # If main detector says tire flat, but the part is not wheel/tire,
    # and archive5 gives a strong different answer, use archive5.
    if (
        main_type == "tire flat"
        and not is_wheel_or_tire_part(damaged_part)
        and archive5_type != "unknown"
        and archive5_conf >= 0.85
    ):
        return {
            "damage_type": archive5_type,
            "damage_confidence": archive5_conf,
            "damage_source": "archive5_conflict_override",
            "archive5_support": archive5_type,
            "archive5_confidence": archive5_conf,
            "damage_notes": notes + [
                "Main detector predicted tire flat, but the damaged part is not wheel/tire. Strong archive5 support was used instead."
            ],
        }

    # Main rule:
    # If cardd_coco is confident, trust it.
    if main_conf >= 0.50:
        return {
            "damage_type": main_type,
            "damage_confidence": main_conf,
            "damage_source": "cardd_coco_best.pt",
            "archive5_support": archive5_type,
            "archive5_confidence": archive5_conf,
            "damage_notes": notes,
        }

    # If cardd_coco is weak but archive5 is strong, use archive5 as backup.
    if archive5_conf >= 0.70:
        return {
            "damage_type": archive5_type,
            "damage_confidence": archive5_conf,
            "damage_source": "archive5_support_fallback",
            "archive5_support": archive5_type,
            "archive5_confidence": archive5_conf,
            "damage_notes": notes + [
                "Main damage detector was weak, so archive5 was used as support fallback."
            ],
        }

    return {
        "damage_type": main_type,
        "damage_confidence": main_conf,
        "damage_source": "low_confidence",
        "archive5_support": archive5_type,
        "archive5_confidence": archive5_conf,
        "damage_notes": notes + [
            "Damage type confidence is low."
        ],
    }


def choose_severity(results):
    """
    Chooses final severity from archive2_best.pt.
    If confidence is below 0.60, mark it as uncertain.
    """
    severity_output = results["severity_classifier"]

    severity = clean_severity(severity_output["class_name"])
    confidence = severity_output["confidence"]

    return {
        "severity": severity,
        "severity_confidence": confidence,
        "severity_uncertain": confidence < 0.60,
    }


def infer_possible_affected_parts(part_info, damage_info, severity_info):
    """
    Infers possible additional affected parts.

    This does not mean all these parts are definitely damaged.
    It means these parts should be shown to user/mechanic for confirmation.

    This version separates front accident and rear accident logic.
    """
    damaged_part = clean_label(part_info.get("damaged_part", "unknown"))
    carpart_suggestion = clean_label(part_info.get("carpart_suggestion", "unknown"))
    support_part_suggestion = clean_label(part_info.get("support_part_suggestion", "unknown"))

    damage_type = clean_label(damage_info.get("damage_type", "unknown"))
    archive5_support = clean_label(damage_info.get("archive5_support", "unknown"))
    severity = clean_label(severity_info.get("severity", "unknown"))

    possible_parts = [
        damaged_part,
        carpart_suggestion,
        support_part_suggestion,
    ]

    all_part_text = " ".join(possible_parts)

    front_context = (
        "front" in all_part_text
        or "windshield" in all_part_text
        or "hood" in all_part_text
        or "bonnet" in all_part_text
    )

    rear_context = (
        "rear" in all_part_text
        or "back" in all_part_text
        or "tailgate" in all_part_text
        or "trunk" in all_part_text
    )

    # Front glass / windshield situation
    if (
        "glass shatter" in damage_type
        or "glass shatter" in archive5_support
        or damaged_part == "windshield"
    ):
        possible_parts.extend([
            "windshield",
            "front bumper",
            "lamp/light",
            "hood/front shell",
            "front body panel"
        ])

    # Front bumper situation
    if damaged_part == "front bumper":
        possible_parts.extend([
            "front bumper",
            "lamp/light",
            "hood/front shell",
            "front body panel"
        ])

    # Rear bumper situation
    if damaged_part == "rear bumper":
        possible_parts.extend([
            "rear bumper",
            "tailgate",
            "lamp/light",
            "rear body panel"
        ])

    # Lamp/light situation
    # Important:
    # Do not add front bumper for rear accident.
    # Do not add rear bumper for front accident.
    if "lamp broken" in damage_type or damaged_part == "lamp/light":
        if rear_context and not front_context:
            possible_parts.extend([
                "rear bumper",
                "rear lamp/light",
                "tailgate",
                "rear body panel"
            ])
        elif front_context and not rear_context:
            possible_parts.extend([
                "front bumper",
                "headlight/front lamp",
                "hood/front shell",
                "front body panel"
            ])
        else:
            possible_parts.extend([
                "lamp/light",
                "body panel"
            ])

    # Wheel/tire situation
    if "tire flat" in damage_type or damaged_part == "wheel/tire":
        possible_parts.extend([
            "wheel/tire",
            "fender/body panel"
        ])

    # Severe accidents may affect extra parts.
    if severity == "severe":
        possible_parts.extend([
            "body shell",
            "internal hidden components"
        ])

    return unique_list(possible_parts)


def fuse_predictions(results):
    """
    Main fusion function.

    Input:
    results from model_inference.py

    Output:
    one final fused result
    """
    damage_classifier = results["damage_classifier"]

    # Choose damaged part first because damage type may depend on part consistency.
    part_info = choose_damaged_part(results)

    # Choose damage type using both damage detector and part evidence.
    damage_info = choose_damage_type(results, part_info)

    # Choose severity.
    severity_info = choose_severity(results)

    damage_segmenter = results["damage_segmenter"]
    carpart_segmenter = results["carpart_segmenter"]

    # Important:
    # Do NOT use archive1 as a hard gate.
    # It predicted whole for some damaged images.
    damage_detected = (
        damage_classifier["class_name"] == "00-damage"
        or damage_info["damage_confidence"] >= 0.50
        or damage_segmenter["has_masks"]
    )

    possible_affected_parts = infer_possible_affected_parts(
        part_info,
        damage_info,
        severity_info
    )

    # Confirmation is required if:
    # 1. part model was uncertain
    # 2. severity is uncertain
    # 3. severity is severe
    # 4. multiple parts may be affected
    confirmation_required = (
        part_info["confirmation_required"]
        or severity_info["severity_uncertain"]
        or severity_info["severity"] == "severe"
        or len(possible_affected_parts) > 1
    )

    confirmation_reasons = []

    if part_info["confirmation_required"]:
        confirmation_reasons.append("Part prediction confidence is uncertain.")

    if severity_info["severity_uncertain"]:
        confirmation_reasons.append("Severity confidence is low.")

    if severity_info["severity"] == "severe":
        confirmation_reasons.append("Severe accident damage may include hidden or multiple damaged parts.")

    if len(possible_affected_parts) > 1:
        confirmation_reasons.append("Multiple affected parts may need confirmation before cost estimation.")

    final_result = {
        "damage_detected": damage_detected,

        "damage_type": damage_info["damage_type"],
        "damage_confidence": damage_info["damage_confidence"],
        "damage_source": damage_info["damage_source"],
        "archive5_support": damage_info["archive5_support"],
        "archive5_confidence": damage_info["archive5_confidence"],
        "damage_notes": damage_info["damage_notes"],

        "severity": severity_info["severity"],
        "severity_confidence": severity_info["severity_confidence"],
        "severity_uncertain": severity_info["severity_uncertain"],

        # Primary damaged part
        "damaged_part": part_info["damaged_part"],
        "part_confidence_level": part_info["part_confidence_level"],
        "part_source": part_info["part_source"],
        "part_notes": part_info["part_notes"],

        # New multi-part support
        "possible_affected_parts": possible_affected_parts,
        "confirmation_required": confirmation_required,
        "confirmation_reasons": confirmation_reasons,

        "damage_mask_available": damage_segmenter["has_masks"],
        "carpart_mask_available": carpart_segmenter["has_masks"],
    }

    return final_result