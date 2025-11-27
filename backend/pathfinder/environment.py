import json
import math
import random
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .config import (
    BIKE_CRASH_DATA_PATH,
    BIKE_CRASH_INFLUENCE_RADIUS,
    BIKE_CRASH_SIGMA,
    BUSY_PUBLIC_TAGS,
    CALM_STREET_TAGS,
    CAR_ROAD_TAGS,
    CRIME_DATA_PATH,
    CRIME_DENSITY_NORMALIZER,
    CRIME_INFLUENCE_RADIUS,
    CRIME_NEAR_DECAY,
    CRIME_SIGMA,
    DEFAULT_DIST,
    GREEN_LANDUSE,
    GREEN_LEISURE,
    LIGHTING_TAGS,
    LOW_SPEED_TAGS,
    PROPERTY_CRIME_TYPES,
    REQUIRED_EDGE_KEYS,
    ROUGH_SURFACES,
    SAFE_POI_TAGS,
    SCENIC_PLACE,
    SIGMA_BUSY,
    SIGMA_GREEN,
    SIGMA_LIGHT,
    SIGMA_SAFE_POI,
    SIGMA_WATER,
    SMOOTH_SURFACES,
    TRAFFIC_DATA_PATH,
    TRAFFIC_SIGMA,
    VIOLENT_CRIME_TYPES,
)
from .utils import as_float, as_int, to_tags, truthy


def tree_geometry_from_candidate(tree_info, candidate):
    if candidate is None or tree_info is None:
        return None
    if isinstance(candidate, (int, np.integer)):
        idx = int(candidate)
        if 0 <= idx < len(tree_info["geoms"]):
            return tree_info["geoms"][idx]
        return None
    if isinstance(candidate, BaseGeometry):
        return candidate
    return None


def resolve_tree_geometry(tree_info, target_geom):
    if tree_info is None:
        return None
    candidate = tree_info["tree"].nearest(target_geom)
    return tree_geometry_from_candidate(tree_info, candidate)


def crime_weight(crime_type: str) -> float:
    label = str(crime_type or "").strip().lower()
    if not label:
        return 1.0
    if label in VIOLENT_CRIME_TYPES:
        return 2.4
    if label in PROPERTY_CRIME_TYPES:
        return 1.3
    if "droge" in label or "waffe" in label:
        return 1.6
    if "brand" in label:
        return 1.2
    return 1.0


def load_crime_points(target_crs):
    if not CRIME_DATA_PATH.exists():
        print(f"Warning: crime dataset missing at {CRIME_DATA_PATH}")
        return None
    try:
        entries = json.loads(CRIME_DATA_PATH.read_text())
    except Exception as exc:
        print(f"Warning: failed to parse crime dataset ({exc})")
        return None

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    geoms = []
    weights = []

    for entry in entries:
        try:
            lon = float(entry.get("lng"))
            lat = float(entry.get("lat"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(lon) or not math.isfinite(lat):
            continue
        x, y = transformer.transform(lon, lat)
        geoms.append(Point(x, y))
        weights.append(crime_weight(entry.get("type")))

    if not geoms:
        print("Warning: crime dataset had no usable points")
        return None

    tree = STRtree(geoms)
    geom_weights = {id(geom): weight for geom, weight in zip(geoms, weights)}
    return {"tree": tree, "geoms": geoms, "weights": geom_weights}


def load_point_dataset(path: Path, target_crs):
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text())
    except Exception as exc:
        print(f"Warning: failed to parse {path.name}: {exc}")
        return []
    geoms = []
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    for entry in entries:
        try:
            lon = float(entry.get("lng") or entry.get("lon") or entry.get("longitude"))
            lat = float(entry.get("lat") or entry.get("latitude"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(lon) or not math.isfinite(lat):
            continue
        x, y = transformer.transform(lon, lat)
        geoms.append(Point(x, y))
    return geoms


def mock_points_from_bounds(graph_proj: nx.MultiDiGraph, count: int):
    xs = [data["x"] for _, data in graph_proj.nodes(data=True)]
    ys = [data["y"] for _, data in graph_proj.nodes(data=True)]
    if not xs or not ys:
        return []
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    geoms = []
    for _ in range(count):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        geoms.append(Point(x, y))
    return geoms


def build_point_tree(geoms, weights=None):
    if not geoms:
        return None
    weights = weights or [1.0 for _ in geoms]
    return {"tree": STRtree(geoms), "geoms": geoms, "weights": {id(g): w for g, w in zip(geoms, weights)}}


def load_or_mock_points(path: Path, graph_proj: nx.MultiDiGraph, target_crs, mock_count: int):
    geoms = load_point_dataset(path, target_crs)
    if not geoms:
        print(f"Using mock points for {path.name} (count={mock_count})")
        geoms = mock_points_from_bounds(graph_proj, mock_count)
    weights = [random.uniform(0.5, 1.4) for _ in geoms]
    return build_point_tree(geoms, weights)


def compute_scenic_score(edge_attrs):
    """
    Returns a scenic score in [0, 1].
    Higher = calmer / more pleasant / less car-oriented.
    """

    highway_tags = {tag for tag in to_tags(edge_attrs.get("highway"))}
    maxspeed = as_int(edge_attrs.get("maxspeed", 50), 50)
    lanes = as_int(edge_attrs.get("lanes", 1), 1)
    surface = str(edge_attrs.get("surface", "")).lower()
    landuse = str(edge_attrs.get("landuse", "")).lower()
    leisure = str(edge_attrs.get("leisure", "")).lower()
    dist_green = as_float(edge_attrs.get("dist_green", DEFAULT_DIST), DEFAULT_DIST)
    dist_water = as_float(edge_attrs.get("dist_water", DEFAULT_DIST), DEFAULT_DIST)

    score = 0.2  # base bias toward neutral

    # Prefer small / human-scale streets and paths
    if highway_tags & LOW_SPEED_TAGS:
        score += 0.35
    elif highway_tags & CALM_STREET_TAGS:
        score += 0.2

    # Penalise big car roads
    if highway_tags & CAR_ROAD_TAGS:
        score -= 0.4

    # Speed heuristics
    if maxspeed <= 30:
        score += 0.2
    elif maxspeed <= 50:
        score += 0.1
    elif maxspeed >= 80:
        score -= 0.25
    elif maxspeed >= 60:
        score -= 0.15

    # Busy roads with many lanes feel less safe
    if lanes >= 4:
        score -= 0.25
    elif lanes >= 3:
        score -= 0.15

    # Reward green context (parks, gardens, forests)
    if landuse in GREEN_LANDUSE or leisure in GREEN_LEISURE:
        score += 0.25

    # Gentle reward for bridges (views) but penalise tunnels
    if truthy(edge_attrs.get("bridge")):
        score += 0.05
    if truthy(edge_attrs.get("tunnel")):
        score -= 0.2

    # Encourage well-lit, paved walkways
    if surface in SMOOTH_SURFACES:
        score += 0.05
    if surface in ROUGH_SURFACES:
        score -= 0.1
    if str(edge_attrs.get("lit", "yes")).lower() in {"no", "0", "false"}:
        score -= 0.05

    # Dedicated cycle infra counts as pleasant fallback
    if edge_attrs.get("cycleway") or edge_attrs.get("segregated"):
        score += 0.1

    # Smooth bonus for proximity to greenery/water (values in meters)
    if dist_green < DEFAULT_DIST:
        score += 0.25 * math.exp(-dist_green / SIGMA_GREEN)
    if dist_water < DEFAULT_DIST:
        score += 0.2 * math.exp(-dist_water / SIGMA_WATER)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def compute_safety_score(edge_attrs):
    """Estimate personal safety (lighting, proximity to crowd/support) for an edge."""

    highway_tags = {tag for tag in to_tags(edge_attrs.get("highway"))}
    surface = str(edge_attrs.get("surface", "")).lower()
    service = str(edge_attrs.get("service", "")).lower()
    sidewalk = str(edge_attrs.get("sidewalk", "")).lower()
    lit_value = str(edge_attrs.get("lit", "")).lower()
    width = as_float(edge_attrs.get("width", 0.0), 0.0)

    dist_safe = as_float(edge_attrs.get("dist_safe_poi", DEFAULT_DIST), DEFAULT_DIST)
    dist_busy = as_float(edge_attrs.get("dist_busy_area", DEFAULT_DIST), DEFAULT_DIST)
    dist_light = as_float(edge_attrs.get("dist_lighting", DEFAULT_DIST), DEFAULT_DIST)
    dist_crime = as_float(edge_attrs.get("dist_crime_hotspot", DEFAULT_DIST), DEFAULT_DIST)
    crime_density = max(0.0, min(1.0, float(edge_attrs.get("crime_density", 0.0))))
    bad_area = max(0.0, min(1.0, float(edge_attrs.get("bad_area_score", 0.0))))

    score = 0.35

    if highway_tags & {"pedestrian", "living_street"}:
        score += 0.25
    elif "residential" in highway_tags or "residential" == edge_attrs.get("highway"):
        score += 0.15
    elif highway_tags & {"primary", "secondary", "tertiary"}:
        score += 0.05

    if highway_tags & {"track", "path", "bridleway"}:
        score -= 0.15

    if service == "alley":
        score -= 0.45
    elif service in {"driveway", "siding"}:
        score -= 0.15

    if truthy(edge_attrs.get("tunnel")):
        score -= 0.3

    if lit_value in {"yes", "true", "1"}:
        score += 0.25
    elif lit_value in {"no", "false", "0"}:
        score -= 0.35

    if sidewalk and sidewalk not in {"no", "none", "0"}:
        score += 0.1

    if width >= 6:
        score += 0.05
    elif 0 < width <= 2.0:
        score -= 0.05

    if surface in SMOOTH_SURFACES:
        score += 0.05
    elif surface in ROUGH_SURFACES:
        score -= 0.05

    safe_aff = 0.0 if dist_safe >= DEFAULT_DIST else math.exp(-dist_safe / SIGMA_SAFE_POI)
    busy_aff = 0.0 if dist_busy >= DEFAULT_DIST else math.exp(-dist_busy / SIGMA_BUSY)
    light_aff = 0.0 if dist_light >= DEFAULT_DIST else math.exp(-dist_light / SIGMA_LIGHT)
    crime_aff = 0.0 if dist_crime >= DEFAULT_DIST else math.exp(-dist_crime / CRIME_NEAR_DECAY)

    score += 0.45 * safe_aff + 0.35 * busy_aff + 0.25 * light_aff
    if crime_density > 0:
        score -= 0.35 * crime_density
    if bad_area > 0:
        score -= 0.45 * bad_area
    if crime_aff > 0:
        score -= 0.25 * crime_aff

    return max(0.0, min(1.0, score))


def is_bike_friendly(attrs):
    if attrs.get("cycleway") or attrs.get("segregated"):
        return True
    highway = str(attrs.get("highway", "")).lower()
    if highway == "cycleway":
        return True
    bicycle = str(attrs.get("bicycle", "")).lower()
    return bicycle in {"designated", "yes"}


def is_bike_accessible(attrs):
    bicycle = str(attrs.get("bicycle", "")).lower()
    highway = str(attrs.get("highway", "")).lower()
    if bicycle == "no":
        return False
    if highway in {"footway", "pedestrian", "steps"} and bicycle not in {"designated", "yes"}:
        return False
    return True


def density_from_points(center: Point, tree_info, radius: float, sigma: float):
    if tree_info is None:
        return 0.0, 0
    buffer_geom = center.buffer(radius)
    density = 0.0
    count = 0
    candidates = tree_info["tree"].query(buffer_geom)
    for candidate in candidates:
        geom = tree_geometry_from_candidate(tree_info, candidate)
        if geom is None:
            continue
        dist = center.distance(geom)
        if dist > radius:
            continue
        weight = tree_info.get("weights", {}).get(id(geom), 1.0)
        density += weight * math.exp(-max(dist, 1.0) / sigma)
        count += 1
    return density, count


def build_linestring(graph, u, v):
    y1 = graph.nodes[u]["y"]
    x1 = graph.nodes[u]["x"]
    y2 = graph.nodes[v]["y"]
    x2 = graph.nodes[v]["x"]
    return LineString([(x1, y1), (x2, y2)])


def load_context_tree(place: str, tags: dict, target_crs):
    try:
        gdf = ox.features_from_place(place, tags=tags)
    except Exception as exc:
        print(f"Warning: could not download {tags} shapes: {exc}")
        return None

    if gdf.empty:
        return None

    try:
        gdf = gdf.to_crs(target_crs)
    except Exception as exc:
        print(f"Warning: could not project context geometries: {exc}")
        return None

    geoms = [
        geom
        for geom in gdf.geometry
        if isinstance(geom, BaseGeometry) and geom is not None and not geom.is_empty
    ]
    if not geoms:
        return None
    return {"tree": STRtree(geoms), "geoms": geoms}


def enrich_graph_with_environment(graph: nx.MultiDiGraph):
    projected = ox.project_graph(graph)
    target_crs = projected.graph.get("crs")

    print("Loading green areas...")
    green_tags = {
        "leisure": ["park", "garden", "playground"],
        "landuse": ["recreation_ground", "meadow", "grass"],
        "natural": ["wood", "grassland"],
    }
    green_tree = load_context_tree(SCENIC_PLACE, green_tags, target_crs)

    print("Loading water bodies...")
    water_tags = {"natural": "water", "waterway": ["river", "stream", "canal"]}
    water_tree = load_context_tree(SCENIC_PLACE, water_tags, target_crs)

    print("Loading safety anchors (police / emergency)...")
    safety_tree = load_context_tree(SCENIC_PLACE, SAFE_POI_TAGS, target_crs)

    print("Loading busy public amenities...")
    busy_tree = load_context_tree(SCENIC_PLACE, BUSY_PUBLIC_TAGS, target_crs)

    print("Loading lighting references...")
    lighting_tree = load_context_tree(SCENIC_PLACE, LIGHTING_TAGS, target_crs)

    print("Loading historic crime incidents...")
    crime_points = load_crime_points(target_crs)

    print("Scoring every edge with environmental context...")
    for u, v, k, data_proj in projected.edges(keys=True, data=True):
        geom = data_proj.get("geometry")
        if geom is None or geom.is_empty:
            geom = build_linestring(projected, u, v)

        center = geom.interpolate(0.5, normalized=True)

        nearest_green = resolve_tree_geometry(green_tree, center)
        dist_green = center.distance(nearest_green) if nearest_green is not None else DEFAULT_DIST

        nearest_water = resolve_tree_geometry(water_tree, center)
        dist_water = center.distance(nearest_water) if nearest_water is not None else DEFAULT_DIST

        nearest_safe = resolve_tree_geometry(safety_tree, center)
        dist_safe = center.distance(nearest_safe) if nearest_safe is not None else DEFAULT_DIST

        nearest_busy = resolve_tree_geometry(busy_tree, center)
        dist_busy = center.distance(nearest_busy) if nearest_busy is not None else DEFAULT_DIST

        nearest_light = resolve_tree_geometry(lighting_tree, center)
        dist_light = center.distance(nearest_light) if nearest_light is not None else DEFAULT_DIST

        dist_crime = DEFAULT_DIST
        crime_density = 0.0
        crime_count = 0
        bad_area = 0.0
        if crime_points is not None:
            nearest_crime = resolve_tree_geometry(crime_points, center)
            if nearest_crime is not None:
                dist_crime = center.distance(nearest_crime)

            buffer_geom = center.buffer(CRIME_INFLUENCE_RADIUS)
            nearby_crimes = crime_points["tree"].query(buffer_geom)
            for incident_candidate in nearby_crimes:
                incident = tree_geometry_from_candidate(crime_points, incident_candidate)
                if incident is None:
                    continue
                dist = center.distance(incident)
                if dist > CRIME_INFLUENCE_RADIUS:
                    continue
                weight = crime_points["weights"].get(id(incident), 1.0)
                influence = weight * math.exp(-max(dist, 1.0) / CRIME_SIGMA)
                crime_density += influence
                crime_count += 1

            if crime_density > 0:
                bad_area = min(1.0, crime_density / CRIME_DENSITY_NORMALIZER)

        data = graph[u][v][k]
        data["dist_green"] = float(dist_green)
        data["dist_water"] = float(dist_water)
        data["scenic_score"] = compute_scenic_score(data)
        data["dist_safe_poi"] = float(dist_safe)
        data["dist_busy_area"] = float(dist_busy)
        data["dist_lighting"] = float(dist_light)
        data["safe_score"] = compute_safety_score(data)
        data["dist_crime_hotspot"] = float(dist_crime)
        data["crime_density"] = float(min(1.0, crime_density))
        data["crime_count_close"] = int(crime_count)
        data["bad_area_score"] = float(bad_area)


def apply_bike_safety_layers(graph: nx.MultiDiGraph):
    """Attach bike crash and traffic risk layers to a graph."""
    projected = ox.project_graph(graph)
    target_crs = projected.graph.get("crs")
    crash_points = load_or_mock_points(BIKE_CRASH_DATA_PATH, projected, target_crs, mock_count=220)
    traffic_points = load_or_mock_points(TRAFFIC_DATA_PATH, projected, target_crs, mock_count=180)

    for u, v, k, data_proj in projected.edges(keys=True, data=True):
        geom = data_proj.get("geometry")
        if geom is None or geom.is_empty:
            geom = build_linestring(projected, u, v)
        center = geom.interpolate(0.5, normalized=True)

        crash_density, _ = density_from_points(center, crash_points, BIKE_CRASH_INFLUENCE_RADIUS, BIKE_CRASH_SIGMA)
        traffic_density, _ = density_from_points(center, traffic_points, BIKE_CRASH_INFLUENCE_RADIUS, TRAFFIC_SIGMA)
        risk_score = min(1.0, 0.7 * crash_density + 0.6 * traffic_density)

        data = graph[u][v][k]
        data["bike_crash_risk"] = float(min(1.0, crash_density))
        data["traffic_risk"] = float(min(1.0, traffic_density))
        data["bike_risk_score"] = float(risk_score)


def graph_has_environmental_data(graph: nx.MultiDiGraph) -> bool:
    for _, _, data in graph.edges(data=True):
        if not all(key in data for key in REQUIRED_EDGE_KEYS):
            return False
    return True
