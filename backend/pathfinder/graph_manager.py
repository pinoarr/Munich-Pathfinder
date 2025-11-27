import networkx as nx
import numpy as np
import osmnx as ox

from .config import CACHE_PATH, CACHE_PATH_BIKE, DEFAULT_DIST, MODE_CONFIG
from .environment import (
    apply_bike_safety_layers,
    compute_safety_score,
    compute_scenic_score,
    enrich_graph_with_environment,
    graph_has_environmental_data,
)

G_walk = None
G_bike = None
NODE_CACHE = {}


def load_or_download_graph(cache_path, network_type: str):
    if cache_path.exists():
        print(f"Loading {network_type} graph from cache: {cache_path}")
        return ox.load_graphml(cache_path)
    print(f"Downloading {network_type} graph for Munich...")
    graph = ox.graph_from_place("Munich, Germany", network_type=network_type, simplify=True)
    ox.save_graphml(graph, cache_path)
    print(f"Graph cached at {cache_path}")
    return graph


def prepare_graph(graph: nx.MultiDiGraph, cache_path, is_bike: bool = False):
    graph = ox.distance.add_edge_lengths(graph)

    if graph_has_environmental_data(graph):
        print("Refreshing scenic scores from cached environmental data...")
        for _, _, _, data in graph.edges(keys=True, data=True):
            data["scenic_score"] = compute_scenic_score(data)
            data["safe_score"] = compute_safety_score(data)
    else:
        print("Computing environmental context (green/water) ...")
        try:
            enrich_graph_with_environment(graph)
            ox.save_graphml(graph, cache_path)
            print(f"Updated graph cache with scenic context for {cache_path.name}.")
        except Exception as exc:
            print(f"Warning: scenic enrichment failed, falling back to heuristics ({exc})")
            for _, _, _, data in graph.edges(keys=True, data=True):
                data.setdefault("dist_green", DEFAULT_DIST)
                data.setdefault("dist_water", DEFAULT_DIST)
                data.setdefault("dist_safe_poi", DEFAULT_DIST)
                data.setdefault("dist_busy_area", DEFAULT_DIST)
                data.setdefault("dist_lighting", DEFAULT_DIST)
                data.setdefault("dist_crime_hotspot", DEFAULT_DIST)
                data.setdefault("crime_density", 0.0)
                data.setdefault("crime_count_close", 0)
                data.setdefault("bad_area_score", 0.0)
                data["scenic_score"] = compute_scenic_score(data)
                data["safe_score"] = compute_safety_score(data)

    if is_bike:
        print("Applying bike crash and traffic safety layers...")
        try:
            apply_bike_safety_layers(graph)
            ox.save_graphml(graph, cache_path)
        except Exception as exc:
            print(f"Warning: bike safety layer enrichment failed: {exc}")

    return graph


def build_node_cache(graph: nx.MultiDiGraph):
    """Precompute node ids and radian coordinates for manual nearest lookup."""
    node_ids = list(graph.nodes)
    lons = np.array([graph.nodes[n]["x"] for n in node_ids], dtype=float)
    lats = np.array([graph.nodes[n]["y"] for n in node_ids], dtype=float)
    return {
        "ids": node_ids,
        "lon_rad": np.deg2rad(lons),
        "lat_rad": np.deg2rad(lats),
    }


def get_node_cache(graph: nx.MultiDiGraph):
    key = id(graph)
    if key not in NODE_CACHE:
        NODE_CACHE[key] = build_node_cache(graph)
    return NODE_CACHE[key]


def fallback_nearest_node(graph: nx.MultiDiGraph, lon: float, lat: float):
    cache = get_node_cache(graph)
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    dlon = cache["lon_rad"] - lon_rad
    dlat = cache["lat_rad"] - lat_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(cache["lat_rad"]) * np.sin(dlon / 2) ** 2
    idx = int(np.argmin(a))
    return cache["ids"][idx]


def get_nearest_nodes(
    graph: nx.MultiDiGraph,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
):
    try:
        start_node = ox.distance.nearest_nodes(graph, start_lon, start_lat)
        end_node = ox.distance.nearest_nodes(graph, end_lon, end_lat)
        return start_node, end_node
    except ImportError:
        return (
            fallback_nearest_node(graph, start_lon, start_lat),
            fallback_nearest_node(graph, end_lon, end_lat),
        )


def get_graph_for_mode(mode: str):
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unknown mode: {mode}")
    if mode.startswith("bike_"):
        return G_bike
    return G_walk


def init_graphs():
    global G_walk, G_bike

    G_walk = load_or_download_graph(CACHE_PATH, "walk")
    G_walk = prepare_graph(G_walk, CACHE_PATH, is_bike=False)

    try:
        G_bike = load_or_download_graph(CACHE_PATH_BIKE, "bike")
        G_bike = prepare_graph(G_bike, CACHE_PATH_BIKE, is_bike=True)
    except Exception as exc:
        print(f"Warning: failed to load bike graph, falling back to walking graph ({exc})")
        G_bike = G_walk

    print("Graphs ready.")
    NODE_CACHE.clear()
    NODE_CACHE[id(G_walk)] = build_node_cache(G_walk)
    NODE_CACHE[id(G_bike)] = build_node_cache(G_bike)
