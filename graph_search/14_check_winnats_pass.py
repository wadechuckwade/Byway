"""
Byway — Direct check: does Winnats Pass (the real, currently-open
road that replaced the historic Mam Tor Road after its 1970s landslip
closure) actually produce a viable, good candidate once given enough
chances to be tried?

WHY THIS EXISTS: 12_check_mam_tor_ranking.py found Winnats Pass
(way_id=163805110) ranked 127th of 505 ways -- well outside the
default num_waypoint_candidates=15 cutoff. 13_compare_mam_tor_
candidate.py separately confirmed the OTHER road, the historic
Mam Tor Road, is genuinely disconnected from the routable graph
(correctly reflecting its real-world closure, not a bug). This script
raises num_waypoint_candidates enough to include rank 127 and checks
directly whether Winnats Pass's candidate route is actually any good,
rather than continuing to guess.
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


if __name__ == "__main__":
    HATHERSAGE = {"name": "Hathersage", "lat": 53.3274, "lon": -1.6447}
    CASTLETON = {"name": "Castleton", "lat": 53.3438, "lon": -1.7752}
    WINNATS_PASS_WAY_ID = 163805110

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

    print("Scoring provisionally...")
    provisional_scores, ways, region_features = score_module.score_all_ways_provisional(graph, node_coords)
    simple_provisional = {wid: s["enjoyment_score"] for wid, s in provisional_scores.items()}

    start_node, _ = path_module.find_nearest_node(HATHERSAGE["lat"], HATHERSAGE["lon"], node_coords)
    end_node, _ = path_module.find_nearest_node(CASTLETON["lat"], CASTLETON["lon"], node_coords)

    direct_path, direct_time_s, _, _ = path_module.dijkstra_shortest_path(graph, start_node, end_node)
    widest_budget = direct_time_s * 1.8
    print(f"Direct time: {round(direct_time_s/60, 1)} min. Widest budget (1.8x): "
          f"{round(widest_budget/60, 1)} min.\n")

    # Raise num_waypoint_candidates enough to comfortably include rank
    # 127 (Winnats Pass), and disable dedup so we see it even if it
    # happens to overlap heavily with another candidate.
    print("Generating candidates with num_waypoint_candidates=150 (up from the default 15) "
          "so Winnats Pass (rank 127) gets a real chance to be tried...")
    candidates = path_module.generate_candidate_routes(
        graph, start_node, end_node, simple_provisional, widest_budget,
        num_waypoint_candidates=150, similarity_threshold=1.01, verbose=False,
    )
    print(f"Generated {len(candidates)} candidates total.\n")

    winnats_candidate = next((c for c in candidates if c["source"] == f"via_way_{WINNATS_PASS_WAY_ID}"), None)

    best_overall = max(candidates, key=lambda c: c["avg_enjoyment"])
    print(f"Best candidate overall (out of {len(candidates)}, dedup disabled): "
          f"source={best_overall['source']}, avg_enjoyment={best_overall['avg_enjoyment']}, "
          f"time={round(best_overall['real_time_s']/60, 1)} min\n")

    if winnats_candidate is None:
        print("Winnats Pass's candidate was NOT generated even at num_waypoint_candidates=150.")
        print("Checking connectivity directly, same as we did for Mam Tor Road...")
        rep_nodes = path_module._representative_nodes_per_way(graph)
        candidate_nodes = rep_nodes.get(WINNATS_PASS_WAY_ID, set())
        any_reachable = False
        for via_node in candidate_nodes:
            if via_node in (start_node, end_node):
                continue
            leg1_path, _, _, _ = path_module.dijkstra_shortest_path(graph, start_node, via_node)
            leg2_path, _, _, _ = path_module.dijkstra_shortest_path(graph, via_node, end_node)
            if leg1_path and leg2_path:
                any_reachable = True
        print(f"Any via-node fully reachable both ways? {any_reachable}")
    else:
        print(f"Winnats Pass's candidate: time={round(winnats_candidate['real_time_s']/60, 1)} min, "
              f"distance={round(winnats_candidate['real_distance_m']/1000, 2)} km, "
              f"avg_enjoyment={winnats_candidate['avg_enjoyment']}")
        fits = winnats_candidate["real_time_s"] <= widest_budget
        print(f"Fits within the 1.8x budget? {fits}")
        is_best = winnats_candidate is best_overall
        print(f"Is it the single best candidate found? {is_best}")
        if not is_best:
            print(f"It lost to '{best_overall['source']}' "
                  f"(avg_enjoyment {best_overall['avg_enjoyment']} vs Winnats Pass's "
                  f"{winnats_candidate['avg_enjoyment']}) -- a real, evidenced comparison, not a guess.")