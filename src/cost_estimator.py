"""
cost_estimator.py

This file estimates Sri Lankan vehicle repair cost ranges.

It uses:
- vehicle brand
- vehicle model
- vehicle year
- damaged part
- possible affected parts
- price_reference.csv
- repair recommendation from repair_rules.py

Important:
This does not give an exact price.
It gives an estimated LKR range because real repair cost depends on:
- part condition
- genuine / reconditioned / used part
- labour cost
- paint work
- garage location
- hidden damage
"""

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE_CSV = PROJECT_ROOT / "data" / "price_reference.csv"


def clean_text(value):
    """
    Converts text into simple lowercase format for matching.
    """
    if value is None:
        return ""

    return str(value).lower().replace("_", " ").replace("-", " ").strip()


def format_lkr(amount):
    """
    Converts number into readable Sri Lankan rupee format.
    Example: 85000 -> Rs. 85,000
    """
    try:
        amount = int(round(float(amount)))
        return f"Rs. {amount:,}"
    except Exception:
        return "Rs. 0"


def load_price_reference(csv_path=DEFAULT_PRICE_CSV):
    """
    Loads price_reference.csv into a list of dictionaries.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Price reference file not found: {csv_path}")

    rows = []

    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def detect_damage_area_context(parts):
    """
    Detects whether the possible affected parts are mainly front-side or rear-side.

    This helps avoid mistakes like:
    rear bumper damage + lamp/light => front headlight

    If the damage is rear-side, lamp/light should match rear lamp/light.
    If the damage is front-side, lamp/light should match headlight/front lamp.
    """
    text = " ".join(clean_text(part) for part in parts)

    rear_context = (
        "rear" in text
        or "tailgate" in text
        or "dickey" in text
        or "back" in text
        or "trunk" in text
    )

    front_context = (
        "front" in text
        or "windshield" in text
        or "hood" in text
        or "bonnet" in text
        or "front glass" in text
    )

    if rear_context and not front_context:
        return "rear"

    if front_context and not rear_context:
        return "front"

    if rear_context and front_context:
        return "mixed"

    return "unknown"


def get_part_aliases(part, damage_context="unknown"):
    """
    Creates possible matching names for a detected part.

    damage_context helps decide whether generic lamp/light should mean:
    - front headlight
    - rear lamp
    """
    part = clean_text(part)

    if part == "lamp/light":
        if damage_context == "rear":
            return [
                "rear lamp/light",
                "headlight/front lamp",
                "lamp/light",
            ]

        if damage_context == "front":
            return [
                "headlight/front lamp",
                "rear lamp/light",
                "lamp/light",
            ]

        return [
            "headlight/front lamp",
            "rear lamp/light",
            "lamp/light",
        ]

    alias_map = {
        "headlight": [
            "headlight/front lamp",
        ],
        "front lamp": [
            "headlight/front lamp",
        ],
        "headlight/front lamp": [
            "headlight/front lamp",
        ],
        "rear lamp": [
            "rear lamp/light",
        ],
        "tail light": [
            "rear lamp/light",
        ],
        "rear lamp/light": [
            "rear lamp/light",
        ],
        "windshield": [
            "windshield",
            "front glass",
        ],
        "front glass": [
            "windshield",
            "front glass",
        ],
        "wheel": [
            "wheel/tire",
        ],
        "tire": [
            "wheel/tire",
        ],
        "tyre": [
            "wheel/tire",
        ],
        "wheel/tire": [
            "wheel/tire",
        ],
        "front bumper": [
            "front bumper",
        ],
        "rear bumper": [
            "rear bumper",
        ],
        "hood": [
            "hood/front shell",
        ],
        "bonnet": [
            "hood/front shell",
        ],
        "hood/front shell": [
            "hood/front shell",
        ],
        "tailgate": [
            "tailgate",
        ],
        "dickey": [
            "tailgate",
        ],
        "fender/body panel": [
            "fender/body panel",
        ],
        "front body panel": [
            "front body panel",
            "fender/body panel",
        ],
        "rear body panel": [
            "rear body panel",
            "fender/body panel",
        ],
        "body panel": [
            "front body panel",
            "rear body panel",
            "fender/body panel",
        ],
        "body shell": [
            "front body panel",
            "rear body panel",
        ],
        "internal hidden components": [],
    }

    if part in alias_map:
        return alias_map[part]

    return [part]


def match_price_rows(price_rows, brand, model, year, part, damage_context="unknown"):
    """
    Finds matching CSV rows for brand + model + year + part.

    Important:
    This version respects alias priority.

    Example:
    If damage_context = rear and part = lamp/light,
    it will search rear lamp/light first before headlight/front lamp.
    """
    brand = clean_text(brand)
    model = clean_text(model)

    try:
        year = int(year)
    except Exception:
        year = None

    part_aliases = get_part_aliases(part, damage_context=damage_context)
    part_aliases = [clean_text(alias) for alias in part_aliases]

    matches = []

    # Alias loop first keeps priority order.
    for alias in part_aliases:
        for row in price_rows:
            row_brand = clean_text(row.get("brand"))
            row_model = clean_text(row.get("model"))
            row_part = clean_text(row.get("part"))

            try:
                year_from = int(row.get("year_from", 0))
                year_to = int(row.get("year_to", 9999))
            except Exception:
                year_from = 0
                year_to = 9999

            brand_match = row_brand == brand
            model_match = row_model == model
            year_match = year is not None and year_from <= year <= year_to
            part_match = row_part == alias

            if brand_match and model_match and year_match and part_match:
                matches.append(row)

    return matches


def choose_best_price_row(matches):
    """
    Chooses the first matching row.
    Since match_price_rows keeps alias priority,
    the first match is the best match.
    """
    if not matches:
        return None

    return matches[0]


def get_part_repair_guidance(part_name, detected_part, condition, fused_result, repair_result):
    """
    Gives user-friendly repair guidance for each matched part.

    This tells the user:
    - whether replacement is required
    - whether repair/paint is enough
    - whether used/reconditioned/new part is used for estimate
    """
    part_name = clean_text(part_name)
    detected_part = clean_text(detected_part)
    condition = clean_text(condition)

    damage_type = clean_text(fused_result.get("damage_type", "unknown"))
    severity = clean_text(fused_result.get("severity", "unknown"))

    recommended_action = clean_text(repair_result.get("recommended_action", "unknown"))
    repair_category = clean_text(repair_result.get("repair_category", "unknown"))
    replacement_option = clean_text(repair_result.get("replacement_option", "unknown"))

    part_text = f"{part_name} {detected_part}"

    # Wheel/tire safety comes first.
    if "wheel" in part_text or "tire" in part_text or "tyre" in part_text:
        return {
            "replacement_required": None,
            "part_action": "Wheel or tyre damage needs safety inspection before use. Replacement is needed only if the wheel or tyre is confirmed damaged.",
            "recommended_part_condition": "Used/new wheel or new tyre may be required depending on inspection.",
        }

    # Glass damage usually needs replacement.
    if "glass" in damage_type or "glass" in repair_category or "windshield" in part_text:
        if "windshield" in part_text or "front glass" in part_text:
            return {
                "replacement_required": True,
                "part_action": "Glass damage usually requires replacement for safety.",
                "recommended_part_condition": "New/replacement glass is recommended.",
            }

    # Lamp/light damage usually needs lamp replacement.
    if "lamp broken" in damage_type or "light replacement" in repair_category:
        if (
            "lamp" in part_text
            or "light" in part_text
            or "headlight" in part_text
        ):
            return {
                "replacement_required": True,
                "part_action": "Broken lamp/light usually requires replacing the light unit.",
                "recommended_part_condition": "Used/reconditioned or new replacement part depending on availability.",
            }

        return {
            "replacement_required": None,
            "part_action": "This nearby part may need inspection, repair, or paint work depending on the actual damage.",
            "recommended_part_condition": condition if condition else "Market reference condition.",
        }

    # Scratch usually does not need replacement.
    if "scratch" in damage_type:
        return {
            "replacement_required": False,
            "part_action": "Repair/paint/polish is usually enough for scratch damage.",
            "recommended_part_condition": "No replacement part normally required unless the panel is badly damaged.",
        }

    # Minor dent usually can be repaired without replacement.
    if "minor dent" in repair_category or recommended_action == "repair without replacement":
        return {
            "replacement_required": False,
            "part_action": "Repair without replacing the part is usually possible.",
            "recommended_part_condition": "Repair/paint work. Used or reconditioned part only if the panel cannot be repaired.",
        }

    # Severe or major damage needs inspection.
    if "manual inspection" in recommended_action or "major damage" in repair_category or severity == "severe":
        return {
            "replacement_required": None,
            "part_action": "Manual inspection is required. Repair or replacement depends on structural condition.",
            "recommended_part_condition": "Used/reconditioned or new part may be required after inspection.",
        }

    # General replacement rule.
    if "replacement required" in replacement_option or recommended_action == "replace part":
        return {
            "replacement_required": True,
            "part_action": "Part replacement is recommended by the repair rule.",
            "recommended_part_condition": condition if condition else "Used/reconditioned or new replacement part.",
        }

    return {
        "replacement_required": None,
        "part_action": "Repair or replacement depends on mechanic confirmation.",
        "recommended_part_condition": condition if condition else "Market reference condition.",
    }


def calculate_part_estimate(row, detected_part, fused_result=None, repair_result=None):
    """
    Calculates total estimate for one part.
    Total = part price + labour price
    """
    if fused_result is None:
        fused_result = {}

    if repair_result is None:
        repair_result = {}

    part_min = float(row.get("part_min_lkr", 0))
    part_max = float(row.get("part_max_lkr", 0))
    labour_min = float(row.get("labour_min_lkr", 0))
    labour_max = float(row.get("labour_max_lkr", 0))

    total_min = part_min + labour_min
    total_max = part_max + labour_max

    part_name = row.get("part", "unknown")
    condition = row.get("condition", "unknown")

    guidance = get_part_repair_guidance(
        part_name=part_name,
        detected_part=detected_part,
        condition=condition,
        fused_result=fused_result,
        repair_result=repair_result
    )

    return {
        "part": part_name,
        "detected_part": detected_part,
        "condition": condition,
        "part_min_lkr": int(part_min),
        "part_max_lkr": int(part_max),
        "labour_min_lkr": int(labour_min),
        "labour_max_lkr": int(labour_max),
        "total_min_lkr": int(total_min),
        "total_max_lkr": int(total_max),
        "total_range": f"{format_lkr(total_min)} - {format_lkr(total_max)}",
        "source_name": row.get("source_name", ""),
        "source_type": row.get("source_type", ""),
        "notes": row.get("notes", ""),
        "replacement_required": guidance["replacement_required"],
        "part_action": guidance["part_action"],
        "recommended_part_condition": guidance["recommended_part_condition"],
    }


def estimate_parts_cost(
    price_rows,
    brand,
    model,
    year,
    parts,
    fused_result=None,
    repair_result=None,
    damage_context="unknown"
):
    """
    Estimates cost for multiple parts.
    """
    if fused_result is None:
        fused_result = {}

    if repair_result is None:
        repair_result = {}

    matched_parts = []
    missing_parts = []

    total_min = 0
    total_max = 0

    already_added_csv_parts = set()

    for part in parts:
        part = clean_text(part)

        if not part or part == "unknown":
            continue

        matches = match_price_rows(
            price_rows=price_rows,
            brand=brand,
            model=model,
            year=year,
            part=part,
            damage_context=damage_context
        )

        best_row = choose_best_price_row(matches)

        if best_row is None:
            missing_parts.append(part)
            continue

        csv_part_name = clean_text(best_row.get("part"))

        # Prevent adding same CSV part twice.
        if csv_part_name in already_added_csv_parts:
            continue

        already_added_csv_parts.add(csv_part_name)

        estimate = calculate_part_estimate(
            row=best_row,
            detected_part=part,
            fused_result=fused_result,
            repair_result=repair_result
        )

        matched_parts.append(estimate)

        total_min += estimate["total_min_lkr"]
        total_max += estimate["total_max_lkr"]

    return {
        "total_min_lkr": int(total_min),
        "total_max_lkr": int(total_max),
        "total_range": f"{format_lkr(total_min)} - {format_lkr(total_max)}",
        "matched_parts": matched_parts,
        "missing_parts": missing_parts,
    }


def estimate_repair_cost(
    fused_result,
    repair_result,
    vehicle_brand,
    vehicle_model,
    vehicle_year,
    csv_path=DEFAULT_PRICE_CSV
):
    """
    Main function used by the system.

    Inputs:
    - fused_result from fusion_logic.py
    - repair_result from repair_rules.py
    - vehicle brand/model/year from user

    Output:
    - primary estimate
    - possible full repair estimate
    - matched parts
    - missing parts
    - warning notes
    """

    price_rows = load_price_reference(csv_path)

    damaged_part = clean_text(fused_result.get("damaged_part", "unknown"))

    possible_parts = repair_result.get("cost_parts_to_confirm", [])
    possible_parts = [clean_text(part) for part in possible_parts]

    all_context_parts = possible_parts.copy()

    if damaged_part and damaged_part != "unknown":
        all_context_parts.append(damaged_part)

    damage_context = detect_damage_area_context(all_context_parts)

    # Primary estimate = main damaged part only.
    primary_parts = []

    if damaged_part and damaged_part != "unknown":
        primary_parts.append(damaged_part)

    primary_estimate = estimate_parts_cost(
        price_rows=price_rows,
        brand=vehicle_brand,
        model=vehicle_model,
        year=vehicle_year,
        parts=primary_parts,
        fused_result=fused_result,
        repair_result=repair_result,
        damage_context=damage_context
    )

    # Full possible estimate = all possible affected parts.
    full_estimate = estimate_parts_cost(
        price_rows=price_rows,
        brand=vehicle_brand,
        model=vehicle_model,
        year=vehicle_year,
        parts=possible_parts,
        fused_result=fused_result,
        repair_result=repair_result,
        damage_context=damage_context
    )

    supported_vehicle = (
        len(primary_estimate["matched_parts"]) > 0
        or len(full_estimate["matched_parts"]) > 0
    )

    notes = [
        "This is an estimated Sri Lankan market price range, not a final quotation.",
        "Final repair cost may change after mechanic inspection.",
        "Hidden/internal damage, paint quality, part condition, and garage labour rates can change the final cost.",
        "The system estimates a price range instead of an exact value because part prices and labour charges vary."
    ]

    if repair_result.get("inspection_required", True):
        notes.append("Manual inspection is required before confirming the final repair cost.")

    if not supported_vehicle:
        notes.append("No matching vehicle price data found in price_reference.csv.")

    if primary_estimate["missing_parts"]:
        notes.append("Primary damaged part price was not found for this vehicle.")

    if full_estimate["missing_parts"]:
        notes.append("Some possible affected parts do not have price data and need manual quotation.")

    return {
        "vehicle": {
            "brand": vehicle_brand,
            "model": vehicle_model,
            "year": vehicle_year,
        },
        "supported_vehicle": supported_vehicle,
        "currency": "LKR",
        "damage_context": damage_context,
        "primary_damaged_part": damaged_part,
        "repair_decision": {
            "recommended_action": repair_result.get("recommended_action", "unknown"),
            "repair_category": repair_result.get("repair_category", "unknown"),
            "replacement_option": repair_result.get("replacement_option", "unknown"),
            "inspection_required": repair_result.get("inspection_required", True),
        },
        "primary_estimate": primary_estimate,
        "possible_full_repair_estimate": full_estimate,
        "inspection_required": repair_result.get("inspection_required", True),
        "cost_estimation_note": notes,
    }


if __name__ == "__main__":
    """
    Small test for cost_estimator.py only.
    This does not run YOLO models.

    This test uses rear-side lamp damage to confirm that:
    lamp/light => rear lamp/light
    """

    sample_fused_result = {
        "damage_type": "lamp broken",
        "severity": "moderate",
        "damaged_part": "rear bumper"
    }

    sample_repair_result = {
        "recommended_action": "replace_part",
        "repair_category": "light_replacement",
        "replacement_option": "replacement_required",
        "inspection_required": True,
        "cost_parts_to_confirm": [
            "rear bumper",
            "tailgate",
            "lamp/light",
            "rear body panel",
            "rear lamp/light"
        ]
    }

    estimate = estimate_repair_cost(
        fused_result=sample_fused_result,
        repair_result=sample_repair_result,
        vehicle_brand="Honda",
        vehicle_model="Vezel",
        vehicle_year=2015
    )

    print(estimate)