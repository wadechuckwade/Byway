"""
Byway — Stark-Contrast Discovery Test: Hathersage -> Castleton
==================================================================

WHY THIS TEST EXISTS: after fixing three real bugs (curvature
boundary loss, Conservation Areas geometry format, NAME field casing)
and replacing fixed blend_factor values with a time-budget search, the
full discovery system (08_three_route_system.py) still only found a
marginal improvement over the direct route for Tillington -> Haslemere
(0.176 -> 0.184 avg enjoyment), and the route it found wasn't even the
specific one the user knows. Two explanations remain open: either the
scoring formula is missing a real signal (views/general countryside
character), or that specific route pair just has a genuinely small,
hard-to-formalize real-world gap.

This test picks a route pair with an UNDISPUTED, externally-documented
stark contrast, independent of anyone's personal opinion: Hathersage
-> Castleton via the direct valley road (A6187) vs via Mam Tor /
Winnats Pass, one of the most consistently named "great driving roads"
in England, in motoring press lists going back decades — not a
personal favourite, a famous one.

CRITICALLY, NO WAYPOINTS ARE GIVEN. Milestone 1 already tested this
pair by hand-specifying Mam Tor as a waypoint (score_route with
waypoints=[MAM_TOR]) and found a partial win (Driving Enjoyment won,
Scenery/History lost to length dilution). That proved the SCORING
formula can recognise Mam Tor's curvature/elevation when pointed at
it. This test asks a different, harder question: given a free hand to
search the whole area, does DISCOVERY find and prefer something like
Mam Tor on its own? If even a famous, stark, undisputed case doesn't
get found/preferred by genuine discovery, that's strong evidence the
gap is structural (missing signal), not just "this particular route
pair is a hard case" — since "hard case" was always the more
charitable, narrower explanation, and a famous case failing closes
that door.

Network note: needs real internet access — run in Codespaces.
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
path_module = _import_from_path("pathfinding", os.path.join(_this_dir, "04_pathfinding.py"))
score_module = _import_from_path("score_graph_enjoyment", os.path.join(_this_dir, "07_score_graph_enjoyment.py"))
three_route_module = _import_from_path("three_route_system", os.path.join(_this_dir, "08_three_route_system.py"))


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

    print("\nBuilding routable graph...")
    graph, node_coords = build_module.build_graph_from_overpass_data(overpass_data)
    print(f"  Graph has {len(node_coords)} nodes, {sum(len(e) for e in graph.values())} directed edges.\n")

    print("PHASE 1: scoring every way in the graph PROVISIONALLY (no whole-graph elevation fetch)...")
    provisional_scores, ways, region_features = score_module.score_all_ways_provisional(graph, node_coords)
    print(f"  Scored {len(provisional_scores)} unique ways (provisionally).\n")

    all_provisional_values = [s["enjoyment_score"] for s in provisional_scores.values()]
    zero_count = sum(1 for v in all_provisional_values if v == 0.0)
    print(f"Provisional enjoyment score distribution across all {len(all_provisional_values)} ways:")
    print(f"  Min: {min(all_provisional_values)}")
    print(f"  Max: {max(all_provisional_values)}")
    print(f"  Mean: {round(sum(all_provisional_values) / len(all_provisional_values), 3)}")
    print(f"  Exactly 0.0: {zero_count} of {len(all_provisional_values)} ways "
          f"({round(100 * zero_count / len(all_provisional_values))}%)")
    top_ways = sorted(provisional_scores.items(), key=lambda kv: -kv[1]["enjoyment_score"])[:10]
    print("\nTop 10 highest PROVISIONAL-enjoyment ways in the whole graph "
          "(no elevation yet — that's refined later, only for ways actually used):")
    for way_id, scores in top_ways:
        print(f"  way_id={way_id}: enjoyment={scores['enjoyment_score']}, "
              f"driving={scores['driving_enjoyment']}, scenery={scores['scenery']}, "
              f"interest={scores['interest']}")
    print()

    start_node, _ = path_module.find_nearest_node(HATHERSAGE["lat"], HATHERSAGE["lon"], node_coords)
    end_node, _ = path_module.find_nearest_node(CASTLETON["lat"], CASTLETON["lon"], node_coords)

    # NO WAYPOINTS. This is genuine discovery: does the search find
    # something resembling Mam Tor / Winnats Pass on its own, given
    # only a start and end point?
    route_results = three_route_module.find_three_routes(graph, start_node, end_node, provisional_scores, ways, region_features=region_features, node_coords=node_coords)

    results = {}
    for label, r in route_results.items():
        if label.startswith("_"):
            continue  # skip diagnostic-only keys (e.g. "_diagnostics"), not a real route tier
        avg_enjoyment = r["avg_enjoyment"]
        road_sequence = three_route_module.summarize_road_sequence(r["way_info"])

        results[label] = {
            "distance_km": round(r["real_distance_m"] / 1000, 2),
            "time_min": round(r["real_time_s"] / 60, 1),
            "avg_enjoyment": avg_enjoyment,
            "roads": road_sequence,
            "source": r["source"],
            "within_budget": r["within_budget"],
        }

        print(f"{'=' * 60}")
        print(f"{label} (source={r['source']}, within_budget={r['within_budget']})")
        print(f"{'=' * 60}")
        print(f"Distance: {results[label]['distance_km']} km")
        print(f"Time: {results[label]['time_min']} minutes")
        print(f"Average enjoyment score along route: {avg_enjoyment}")
        print("Roads:")
        for road in road_sequence:
            print(f"  - {road}")
        print()

    print(f"{'=' * 60}")
    print("THE ACTUAL QUESTION THIS TEST ASKS:")
    print("Does any road in 'Roads:' above mention Winnats Pass, Mam")
    print("Tor, Castleton Road (the high route), or Buxton Road — the")
    print("real, famous scenic alternative — for Compromise or Max")
    print("Enjoyment? If even this well-documented, stark case doesn't")
    print("get found/preferred, that's real evidence the gap is a")
    print("missing signal, not just 'this route pair is a hard one.'")
    print(f"{'=' * 60}")
    for label in ["Direct (fastest)", "Compromise", "Max Enjoyment"]:
        if label in results:
            r = results[label]
            print(f"{label}: {r['time_min']} min, {r['distance_km']} km, "
                  f"enjoyment={r['avg_enjoyment']}, source={r['source']}")
