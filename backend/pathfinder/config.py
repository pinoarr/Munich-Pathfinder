from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Cache paths
CACHE_PATH = BASE_DIR / "data" / "munich_walk.graphml"
CACHE_PATH_BIKE = BASE_DIR / "data" / "munich_bike.graphml"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_PATH_BIKE.parent.mkdir(parents=True, exist_ok=True)

# Static data directory
STATIC_DATA_DIR = BASE_DIR / "static_data"
STATIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Environment tags and categories
LOW_SPEED_TAGS = {"footway", "path", "cycleway", "pedestrian", "living_street", "track"}
CALM_STREET_TAGS = {"residential", "service", "unclassified"}
CAR_ROAD_TAGS = {"primary", "primary_link", "secondary", "secondary_link", "trunk", "trunk_link"}
GREEN_LEISURE = {"park", "garden", "playground", "nature_reserve"}
GREEN_LANDUSE = {"forest", "grass", "meadow", "recreation_ground"}
SMOOTH_SURFACES = {"asphalt", "paved", "paving_stones", "concrete"}
ROUGH_SURFACES = {"gravel", "dirt", "ground", "unpaved"}

# Scenic and safety parameters
DEFAULT_DIST = 9999.0
SIGMA_GREEN = 420.0
SIGMA_WATER = 320.0
SIGMA_SAFE_POI = 140.0
SIGMA_BUSY = 200.0
SIGMA_LIGHT = 160.0
SCENIC_PLACE = "Munich, Germany"

# Crime and risk data
CRIME_DATA_PATH = STATIC_DATA_DIR / "munich_crime_points.json"
CRIME_INFLUENCE_RADIUS = 220.0
CRIME_SIGMA = 95.0
CRIME_DENSITY_NORMALIZER = 3.4
CRIME_NEAR_DECAY = 140.0
BIKE_CRASH_SIGMA = 140.0
BIKE_CRASH_INFLUENCE_RADIUS = 260.0
TRAFFIC_SIGMA = 180.0
BIKE_CRASH_DATA_PATH = STATIC_DATA_DIR / "munich_bike_crashes.json"
TRAFFIC_DATA_PATH = STATIC_DATA_DIR / "munich_traffic_hotspots.json"

SAFE_POI_TAGS = {
    "amenity": ["police", "fire_station", "ranger_station", "embassy", "hospital", "clinic"],
    "emergency": ["ambulance_station"],
    "building": ["police"],
}
BUSY_PUBLIC_TAGS = {
    "amenity": [
        "bus_station",
        "subway_entrance",
        "tram_stop",
        "ferry_terminal",
        "marketplace",
        "community_centre",
        "townhall",
        "library",
        "cinema",
        "theatre",
        "arts_centre",
        "university",
        "college",
        "school",
        "restaurant",
        "cafe",
    ],
    "public_transport": ["stop_position", "station", "platform"],
    "railway": ["station", "halt", "tram_stop", "light_rail"],
}
LIGHTING_TAGS = {
    "highway": "street_lamp",
    "man_made": "street_lamp",
}

VIOLENT_CRIME_TYPES = {"raub", "koerperverletzung", "vergewaltigung", "mord", "totschlag"}
PROPERTY_CRIME_TYPES = {"diebstahl", "einbruch", "taschendiebstahl"}

MODE_CONFIG = {
    "fast": {
        "length_bias": 1.05,
        "min_factor": 1.0,
        "scenic_avoid": 0.65,
        "safe_penalty": 0.8,
        "bike_penalty": 0.2,
        "lit_penalty": 0.15,
        "dark_reward": 0.4,
    },
    "scenic": {  # go all in on rivers/parks
        "length_bias": 0.65,
        "scenic_reward": 1.9,
        "water_reward": 2.8,
        "green_reward": 1.6,
        "detour_penalty": 0.02,
        "min_factor": 0.06,
        "water_decay": 120.0,
        "green_decay": 320.0,
    },
    "safe": {
        "length_bias": 1.05,
        "min_factor": 0.35,
        "safe_score_reward": 2.8,
        "safe_anchor_reward": 1.9,
        "busy_reward": 1.3,
        "lighting_reward": 1.0,
        "crime_density_penalty": 2.2,
        "bad_area_penalty": 3.4,
        "crime_near_penalty": 1.9,
        "crime_gap_reward": 1.2,
        "unsafe_penalty": 1.4,
        "unsafe_floor": 0.35,
        "unsafe_multiplier": 3.2,
        "safe_decay": 110.0,
        "busy_decay": 180.0,
        "lighting_decay": 150.0,
        "crime_gap_decay": 200.0,
    },
    "bike_fast": {
        "length_bias": 1.0,
        "min_factor": 0.6,
        "bike_lane_reward": 0.8,
        "traffic_penalty": 1.6,
        "non_bike_penalty": 50.0,
        "lit_penalty": 0.1,
    },
    "bike_safe": {
        "length_bias": 1.05,
        "min_factor": 0.45,
        "bike_lane_reward": 0.9,
        "crash_penalty": 3.2,
        "traffic_penalty": 2.4,
        "safe_score_reward": 1.6,
        "safe_anchor_reward": 1.2,
        "busy_reward": 0.8,
        "lighting_reward": 0.9,
        "non_bike_penalty": 60.0,
        "unsafe_penalty": 1.2,
        "crime_density_penalty": 1.8,
        "bad_area_penalty": 2.2,
        "crime_near_penalty": 1.4,
        "crime_gap_reward": 0.8,
        "safe_decay": 110.0,
        "busy_decay": 160.0,
        "lighting_decay": 150.0,
        "crime_gap_decay": 200.0,
        "unsafe_floor": 0.35,
        "unsafe_multiplier": 2.8,
    },
    "bike_scenic": {
        "length_bias": 0.8,
        "scenic_reward": 1.6,
        "water_reward": 2.0,
        "green_reward": 1.3,
        "detour_penalty": 0.08,
        "min_factor": 0.08,
        "water_decay": 150.0,
        "green_decay": 300.0,
        "bike_lane_reward": 0.7,
        "non_bike_penalty": 40.0,
    },
}

# Backwards-compatible aliases for earlier beta modes
MODE_CONFIG["scenic_plus"] = MODE_CONFIG["scenic"]
MODE_CONFIG["scenic_river"] = MODE_CONFIG["scenic"]

MODE_DISPLAY_NAME = {
    "fast": "Direct",
    "safe": "Night Safe",
    "scenic": "Scenic",
    "bike_fast": "Bike Direct",
    "bike_safe": "Ride Safe",
    "bike_scenic": "Bike Scenic",
}

MODE_SPEED_MPS = {
    "fast": 1.5,
    "safe": 1.35,
    "scenic": 1.3,
    "bike_fast": 4.8,
    "bike_safe": 4.4,
    "bike_scenic": 4.2,
}

DEFAULT_SPEED_MPS = 1.35
GREEN_PROX_THRESHOLD = 80.0
WATER_PROX_THRESHOLD = 80.0
SAFE_PROX_THRESHOLD = 120.0
CRIME_PROX_THRESHOLD = 80.0
CRIME_DENSITY_THRESHOLD = 0.35
COMPARISON_KEYS = (
    "length_m",
    "duration_s",
    "avg_scenic",
    "avg_safe_score",
    "green_share",
    "water_share",
    "bike_share",
    "lit_share",
    "crime_hot_share",
    "safe_anchor_share",
)
LIT_POSITIVE = {"yes", "true", "1"}
LIT_NEGATIVE = {"no", "false", "0"}

REQUIRED_EDGE_KEYS = (
    "scenic_score",
    "dist_green",
    "dist_water",
    "safe_score",
    "dist_safe_poi",
    "dist_busy_area",
    "dist_lighting",
    "crime_density",
    "dist_crime_hotspot",
    "bad_area_score",
    "crime_count_close",
)
