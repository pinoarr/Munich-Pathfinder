from .config import (
    CRIME_DENSITY_THRESHOLD,
    CRIME_PROX_THRESHOLD,
    DEFAULT_DIST,
    GREEN_PROX_THRESHOLD,
    LIT_NEGATIVE,
    LIT_POSITIVE,
    SAFE_PROX_THRESHOLD,
    WATER_PROX_THRESHOLD,
)
from .environment import is_bike_friendly
from .utils import as_float


def init_segment_stats():
    return {
        "length": 0.0,
        "segments": 0,
        "scenic_sum": 0.0,
        "safe_sum": 0.0,
        "crime_density_sum": 0.0,
        "bad_area_sum": 0.0,
        "green_length": 0.0,
        "water_length": 0.0,
        "bike_length": 0.0,
        "lit_length": 0.0,
        "dark_length": 0.0,
        "safe_anchor_length": 0.0,
        "crime_hot_length": 0.0,
    }


def accumulate_segment_stats(stats, attrs, segment_length):
    scenic = max(0.0, min(1.0, float(attrs.get("scenic_score", 0.0))))
    safe = max(0.0, min(1.0, float(attrs.get("safe_score", 0.0))))
    dist_green = as_float(attrs.get("dist_green", DEFAULT_DIST), DEFAULT_DIST)
    dist_water = as_float(attrs.get("dist_water", DEFAULT_DIST), DEFAULT_DIST)
    dist_safe = as_float(attrs.get("dist_safe_poi", DEFAULT_DIST), DEFAULT_DIST)
    dist_crime = as_float(attrs.get("dist_crime_hotspot", DEFAULT_DIST), DEFAULT_DIST)
    crime_density = max(0.0, float(attrs.get("crime_density", 0.0)))
    bad_area = max(0.0, float(attrs.get("bad_area_score", 0.0)))
    lit_value = str(attrs.get("lit", "")).lower()

    stats["length"] += segment_length
    stats["segments"] += 1
    stats["scenic_sum"] += scenic * segment_length
    stats["safe_sum"] += safe * segment_length
    stats["crime_density_sum"] += crime_density * segment_length
    stats["bad_area_sum"] += bad_area * segment_length

    if dist_green < GREEN_PROX_THRESHOLD:
        stats["green_length"] += segment_length
    if dist_water < WATER_PROX_THRESHOLD:
        stats["water_length"] += segment_length
    if dist_safe < SAFE_PROX_THRESHOLD:
        stats["safe_anchor_length"] += segment_length
    if dist_crime < CRIME_PROX_THRESHOLD or crime_density >= CRIME_DENSITY_THRESHOLD:
        stats["crime_hot_length"] += segment_length
    if is_bike_friendly(attrs):
        stats["bike_length"] += segment_length
    if lit_value in LIT_POSITIVE:
        stats["lit_length"] += segment_length
    elif lit_value in LIT_NEGATIVE:
        stats["dark_length"] += segment_length


def finalize_segment_stats(raw_stats):
    total = raw_stats["length"]
    if total <= 0:
        return {
            "length_m": 0.0,
            "segments": 0,
            "avg_scenic": 0.0,
            "avg_safe_score": 0.0,
            "avg_crime_density": 0.0,
            "avg_bad_area": 0.0,
            "green_share": 0.0,
            "water_share": 0.0,
            "bike_share": 0.0,
            "lit_share": 0.0,
            "dark_share": 0.0,
            "crime_hot_share": 0.0,
            "safe_anchor_share": 0.0,
        }

    def share(value):
        return value / total if total > 0 else 0.0

    return {
        "length_m": total,
        "segments": raw_stats["segments"],
        "avg_scenic": raw_stats["scenic_sum"] / total,
        "avg_safe_score": raw_stats["safe_sum"] / total,
        "avg_crime_density": raw_stats["crime_density_sum"] / total,
        "avg_bad_area": raw_stats["bad_area_sum"] / total,
        "green_share": share(raw_stats["green_length"]),
        "water_share": share(raw_stats["water_length"]),
        "bike_share": share(raw_stats["bike_length"]),
        "lit_share": share(raw_stats["lit_length"]),
        "dark_share": share(raw_stats["dark_length"]),
        "crime_hot_share": share(raw_stats["crime_hot_length"]),
        "safe_anchor_share": share(raw_stats["safe_anchor_length"]),
    }
