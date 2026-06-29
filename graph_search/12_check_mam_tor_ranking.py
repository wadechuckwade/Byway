"""
Byway — Direct check: is Winnats Pass / Mam Tor even in the top-N
candidate pool for Hathersage -> Castleton, and if not, where does it
actually rank?

WHY THIS EXISTS: the Hathersage/Castleton discovery test found a real,
different detour (via Ashopton Road/Ladybower) for Max Enjoyment, but
not specifically Winnats Pass/Mam Tor. Rather than guess why, this
searches the actual scored graph data directly for any way named
"Winnats" or "Mam Tor" (or refs B6061, A625 -- the real road numbers
for these), and reports its provisional rank and category scores --
answering "is it being excluded by the top-15 cutoff" with evidence,
not speculation.

Run from graph_search/ alongside the other numbered scripts.
"""

import os
import importlib.util


def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_this_dir = os.path.dirname(__file__)
fetch_module = _import_from_path("fetch_road_network", os.path.join(_this_dir, "01_fetch_road_network.py"))
build_module = _import_from_path("build_graph", os.path.join(_this_dir, "02_build_graph.py"))
score_module = _import_from_path("score_graph_enjoyment", os.path.join(_this_dir, "07_score_graph_enjoyment.py"))


if __name__ == "__main__":
    HATHERSAGE = {"name": "Hathersage", "lat": 53.3274, "lon": -1.6447}
    CASTLETON = {"name": "Castleton", "lat": 53.3438, "lon": -1.7752}

    margin_degrees = 0.03
    bbox = {
        "min_lat": min(HATHERSAGE["lat"], CASTLETON["lat"]) - margin_degrees,
        "max_lat": max(HATHERSAGE["lat"], CASTLETON["lat"]) + margin_degrees,
        "min_lon": min(HATHERSAGE["lon"], CASTLETON["lon"]) - margin_degrees,
        "max_lon": max(HATHERSAGE["lon"], CASTLETON["lon"]) + margin_degrees,
    }

    print("Fetching real road network data...")
    overpass_data = fetch_module.fetch_all_roads_in_bbox(bbox)
    graph, node_coords = build_module.build_graph_from_overpass_data(overpass_data)
    print(f"  Graph has {len(node_coords)} nodes, {sum(len(e) for e in graph.values())} directed edges.\n")

    print("Scoring provisionally (no elevation -- we just need ranks/categories here)...")
    provisional_scores, ways, region_features = score_module.score_all_ways_provisional(graph, node_coords)

    # Search for any way whose name or ref mentions Winnats/Mam Tor,
    # or matches the real road numbers (B6061 = Winnats Pass, A625 =
    # the historic, landslip-closed Mam Tor road).
    search_terms = ["winnats", "mam tor"]
    search_refs = ["b6061", "a625"]

    matches = []
    for way_id, way_data in ways.items():
        way_info = way_data["way_info"]
        name = (way_info.get("name") or "").lower()
        ref = (way_info.get("ref") or "").lower()
        if any(term in name for term in search_terms) or ref in search_refs:
            matches.append((way_id, way_info, way_data))

    sorted_by_score = sorted(provisional_scores.items(), key=lambda kv: -kv[1]["enjoyment_score"])
    rank_by_way_id = {way_id: rank for rank, (way_id, _) in enumerate(sorted_by_score, start=1)}

    if not matches:
        print("No way in this graph is named 'Winnats'/'Mam Tor' or has ref B6061/A625.")
        print("This could mean: (a) it's genuinely outside this bbox, (b) it's tagged in")
        print("OSM without a name/ref we're matching on, or (c) it's been split into many")
        print("small unnamed segments. Worth checking the bbox/OSM data directly if this")
        print("matters -- not something this script alone can rule out.")
    else:
        print(f"Found {len(matches)} matching way(s):\n")
        for way_id, way_info, way_data in matches:
            scores = provisional_scores.get(way_id, {})
            rank = rank_by_way_id.get(way_id, "?")
            print(f"way_id={way_id}  name='{way_info.get('name')}'  ref='{way_info.get('ref')}'  "
                  f"highway={way_info.get('highway')}")
            print(f"  Provisional rank: {rank} of {len(provisional_scores)} ways")
            print(f"  enjoyment={scores.get('enjoyment_score')}, driving={scores.get('driving_enjoyment')}, "
                  f"scenery={scores.get('scenery')}, history={scores.get('history_culture')}")
            print(f"  road_class_multiplier={scores.get('road_class_multiplier')}")
            print()

        print("=" * 60)
        print("If rank is well outside the top 15 (the current num_waypoint_candidates")
        print("default), that's the direct, evidenced reason it's never tried as a forced")
        print("waypoint -- not a discovery-mechanism failure, a candidate-count limit.")
        print("=" * 60)