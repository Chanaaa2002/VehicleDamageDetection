"""
repair_rules.py

This file decides the repair recommendation after fusion.
"""


def clean_text(value):
    if value is None:
        return "unknown"

    return str(value).lower().replace("_", " ").replace("-", " ").strip()


def unique_clean_list(items):
    cleaned = []

    for item in items:
        text = clean_text(item)

        if text and text != "unknown" and text not in cleaned:
            cleaned.append(text)

    return cleaned


def clean_reason_text(text):
    """
    Final cleanup for reason text.
    This fixes missing-space issues before showing output.
    """
    text = str(text)

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

        "glass related": "glass-related",
        "Glass related": "Glass-related",
    }

    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)

    text = " ".join(text.split())

    if text and not text.endswith("."):
        text += "."

    return text


def build_reason(fused_result, extra_reasons):
    """
    Builds final clean reason text.
    """
    reasons = []

    confirmation_reasons = fused_result.get("confirmation_reasons", [])

    for reason in confirmation_reasons:
        if reason:
            reasons.append(str(reason).strip())

    if fused_result.get("confirmation_required", True) and not reasons:
        part_confidence_level = clean_text(
            fused_result.get("part_confidence_level", "unknown")
        )

        if part_confidence_level in ["low", "medium", "unknown"]:
            reasons.append("Part prediction confidence is uncertain.")
        else:
            reasons.append("User confirmation is recommended before final cost estimation.")

    for reason in extra_reasons:
        if reason:
            reasons.append(str(reason).strip())

    # Remove duplicates
    final_reasons = []
    seen = set()

    for reason in reasons:
        reason = clean_reason_text(reason)

        if reason and reason not in seen:
            final_reasons.append(reason)
            seen.add(reason)

    final_text = " ".join(final_reasons)

    # Final safety cleanup after joining
    final_text = clean_reason_text(final_text)

    return final_text


def decide_repair_action(fused_result):
    damage_type = clean_text(fused_result.get("damage_type", "unknown"))
    severity = clean_text(fused_result.get("severity", "unknown"))
    damaged_part = clean_text(fused_result.get("damaged_part", "unknown"))

    confirmation_required = fused_result.get("confirmation_required", True)
    severity_uncertain = fused_result.get("severity_uncertain", False)

    possible_affected_parts = fused_result.get("possible_affected_parts", [])
    possible_affected_parts = unique_clean_list(possible_affected_parts)

    can_estimate_cost = damaged_part != "unknown" or len(possible_affected_parts) > 0

    glass_parts = ["windshield", "front glass", "back glass", "rear glass"]
    light_parts = [
        "lamp/light",
        "headlight/front lamp",
        "rear lamp/light",
        "front light",
        "rear light",
        "headlight",
        "tail light",
    ]
    wheel_parts = ["wheel/tire", "wheel", "tire", "tyre"]

    if severity == "severe":
        reason = build_reason(
            fused_result,
            ["Severe damage may include hidden internal or structural damage."]
        )

        return {
            "recommended_action": "manual_inspection",
            "repair_category": "major_damage",
            "replacement_option": "possible_replacement",
            "inspection_required": True,
            "can_estimate_cost": can_estimate_cost,
            "cost_parts_to_confirm": possible_affected_parts,
            "reason": reason,
        }

    if damaged_part in glass_parts or "glass shatter" in damage_type:
        reason = build_reason(
            fused_result,
            ["Glass-related damage normally requires replacement for safety."]
        )

        return {
            "recommended_action": "replace_part",
            "repair_category": "glass_replacement",
            "replacement_option": "replacement_required",
            "inspection_required": confirmation_required or severity_uncertain,
            "can_estimate_cost": can_estimate_cost,
            "cost_parts_to_confirm": possible_affected_parts,
            "reason": reason,
        }

    if damaged_part in light_parts or "lamp broken" in damage_type:
        reason = build_reason(
            fused_result,
            ["Broken lamp or light damage usually requires part replacement."]
        )

        return {
            "recommended_action": "replace_part",
            "repair_category": "light_replacement",
            "replacement_option": "replacement_required",
            "inspection_required": confirmation_required or severity_uncertain,
            "can_estimate_cost": can_estimate_cost,
            "cost_parts_to_confirm": possible_affected_parts,
            "reason": reason,
        }

    if damaged_part in wheel_parts or "tire flat" in damage_type:
        reason = build_reason(
            fused_result,
            ["Wheel or tire damage should be inspected for safety before use."]
        )

        return {
            "recommended_action": "manual_inspection",
            "repair_category": "wheel_or_tire_damage",
            "replacement_option": "possible_replacement",
            "inspection_required": True,
            "can_estimate_cost": can_estimate_cost,
            "cost_parts_to_confirm": possible_affected_parts,
            "reason": reason,
        }

    if "scratch" in damage_type:
        if severity == "minor":
            reason = build_reason(
                fused_result,
                ["Minor scratch damage can usually be polished or repainted."]
            )

            return {
                "recommended_action": "polish_or_repaint",
                "repair_category": "minor_paint_damage",
                "replacement_option": "not_required",
                "inspection_required": confirmation_required or severity_uncertain,
                "can_estimate_cost": can_estimate_cost,
                "cost_parts_to_confirm": possible_affected_parts,
                "reason": reason,
            }

        reason = build_reason(
            fused_result,
            ["Moderate scratch damage may require repainting or panel repair."]
        )

        return {
            "recommended_action": "repaint_or_repair",
            "repair_category": "paint_or_panel_damage",
            "replacement_option": "not_required",
            "inspection_required": confirmation_required or severity_uncertain,
            "can_estimate_cost": can_estimate_cost,
            "cost_parts_to_confirm": possible_affected_parts,
            "reason": reason,
        }

    if "dent" in damage_type:
        if severity == "minor":
            reason = build_reason(
                fused_result,
                ["Minor dent damage may be repaired without replacing the part."]
            )

            return {
                "recommended_action": "repair_without_replacement",
                "repair_category": "minor_dent_repair",
                "replacement_option": "not_required",
                "inspection_required": confirmation_required or severity_uncertain,
                "can_estimate_cost": can_estimate_cost,
                "cost_parts_to_confirm": possible_affected_parts,
                "reason": reason,
            }

        reason = build_reason(
            fused_result,
            ["Moderate dent damage may need panel beating, repainting, or replacement depending on part condition."]
        )

        return {
            "recommended_action": "repair_or_replace",
            "repair_category": "dent_repair",
            "replacement_option": "possible_replacement",
            "inspection_required": confirmation_required or severity_uncertain,
            "can_estimate_cost": can_estimate_cost,
            "cost_parts_to_confirm": possible_affected_parts,
            "reason": reason,
        }

    if "crack" in damage_type:
        reason = build_reason(
            fused_result,
            ["Crack damage can spread and may require replacement depending on the affected part."]
        )

        return {
            "recommended_action": "repair_or_replace",
            "repair_category": "crack_damage",
            "replacement_option": "possible_replacement",
            "inspection_required": True,
            "can_estimate_cost": can_estimate_cost,
            "cost_parts_to_confirm": possible_affected_parts,
            "reason": reason,
        }

    reason = build_reason(
        fused_result,
        ["Damage type or part is uncertain, so manual inspection is recommended."]
    )

    return {
        "recommended_action": "manual_inspection",
        "repair_category": "uncertain_damage",
        "replacement_option": "unknown",
        "inspection_required": True,
        "can_estimate_cost": can_estimate_cost,
        "cost_parts_to_confirm": possible_affected_parts,
        "reason": reason,
    }