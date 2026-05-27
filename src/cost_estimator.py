import csv
from pathlib import Path
from typing import Optional, Dict


PRICE_FILE = Path("data/price_reference.csv")


def load_price_reference() -> list:
    """
    Load price reference CSV.
    """
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_FILE}")

    with open(PRICE_FILE, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def find_price_row(
    brand: str,
    model: str,
    year: str,
    part: str
) -> Optional[Dict]:
    """
    Find matching price row for vehicle and part.
    """

    rows = load_price_reference()

    brand = brand.lower()
    model = model.lower()
    year = str(year).lower()
    part = part.lower()

    for row in rows:
        if (
            row["brand"].lower() == brand
            and row["model"].lower() == model
            and str(row["year"]).lower() == year
            and row["part"].lower() == part
        ):
            return row

    return None


def estimate_cost(
    brand: str,
    model: str,
    year: str,
    damaged_part: str,
    repair_action: str
) -> Dict:
    """
    Estimate cost based on vehicle part and repair action.
    """

    price_row = find_price_row(brand, model, year, damaged_part)

    if price_row is None:
        return {
            "cost_available": False,
            "message": "No price reference found for this vehicle/part.",
            "estimated_cost": None
        }

    if repair_action in ["repair_without_replacement", "repair_or_repaint", "repair_possible"]:
        min_cost = price_row["repair_min"]
        max_cost = price_row["repair_max"]
        cost_type = "repair_cost"

    elif repair_action == "replace_part":
        min_cost = price_row["recondition_min"]
        max_cost = price_row["new_max"]
        cost_type = "replacement_cost_range"

    else:
        min_cost = min(
            int(price_row["repair_min"]),
            int(price_row["recondition_min"]),
            int(price_row["new_min"])
        )
        max_cost = max(
            int(price_row["repair_max"]),
            int(price_row["recondition_max"]),
            int(price_row["new_max"])
        )
        cost_type = "manual_review_range"

    return {
        "cost_available": True,
        "cost_type": cost_type,
        "estimated_cost": f"LKR {min_cost} - {max_cost}",
        "note": "This is an estimated range. Final price should be confirmed by a repair shop."
    }