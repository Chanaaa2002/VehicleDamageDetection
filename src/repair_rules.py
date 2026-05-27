def decide_repair_action(damage_type: str, damaged_part: str, severity: str) -> dict:
    """
    Decide whether the part can be repaired or should be replaced.

    severity:
    minor / moderate / severe
    """

    damage_type = damage_type.lower()
    damaged_part = damaged_part.lower()
    severity = severity.lower()

    replace_damage_types = ["glass shatter", "lamp broken", "tire flat"]

    if damage_type in replace_damage_types:
        return {
            "action": "replace_part",
            "replacement_option": "new_or_reconditioned",
            "reason": f"{damage_type} usually needs part replacement."
        }

    if "mirror" in damaged_part and severity in ["moderate", "severe"]:
        return {
            "action": "replace_part",
            "replacement_option": "new_or_reconditioned",
            "reason": "Mirror damage usually requires replacement."
        }

    if "light" in damaged_part and severity in ["moderate", "severe"]:
        return {
            "action": "replace_part",
            "replacement_option": "new_or_reconditioned",
            "reason": "Light damage usually requires replacement."
        }

    if damage_type == "scratch":
        if severity == "minor":
            return {
                "action": "repair_without_replacement",
                "replacement_option": "not_required",
                "reason": "Minor scratch can usually be polished or repainted."
            }
        else:
            return {
                "action": "repair_or_repaint",
                "replacement_option": "not_required",
                "reason": "Scratch damage usually needs repair or repainting."
            }

    if damage_type == "dent":
        if severity in ["minor", "moderate"]:
            return {
                "action": "repair_without_replacement",
                "replacement_option": "not_required",
                "reason": "Dent may be repairable without replacing the part."
            }
        else:
            return {
                "action": "manual_inspection_required",
                "replacement_option": "possible_replacement",
                "reason": "Severe dent may need replacement depending on structural damage."
            }

    if damage_type == "crack":
        if severity == "minor":
            return {
                "action": "repair_possible",
                "replacement_option": "not_required_or_reconditioned",
                "reason": "Small crack may be repairable depending on part material."
            }
        else:
            return {
                "action": "replace_part",
                "replacement_option": "new_or_reconditioned",
                "reason": "Moderate or severe crack may require replacement."
            }

    return {
        "action": "manual_inspection_required",
        "replacement_option": "unknown",
        "reason": "Damage type or part needs manual review."
    }