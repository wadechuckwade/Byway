"""
Byway — Direct check: did the Mam Tor Road candidate actually get
generated, and if so, why did it lose to the Ashopton Road one?

WHY THIS EXISTS: 12_check_mam_tor_ranking.py confirmed Mam Tor Road
(way_id=164088231) ranked 12th of 505 ways provisionally -- well
within the top-15 candidate pool -- yet the actual Max Enjoyment
result went via Ashopton Road instead. This prints EVERY generated
candidate (not just the winner) so we can see the real numbers behind
that choice, rather than guessing why.
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

    print(f"\nDirect time: {round(direct_time_s/60, 1)} min. Widest budget (1.8x): "
          f"{round(widest_budget/60, 1)} min.\n")

    candidates = path_module.generate_candidate_routes(
        graph, start_node, end_node, simple_provisional, widest_budget,
        num_waypoint_candidates=15, similarity_threshold=1.01, verbose=False,
        # similarity_threshold=1.01 effectively DISABLES deduplication
        # (Jaccard similarity can be at most 1.0) -- purely diagnostic,
        # so we can see EVERY candidate's real numbers, including ones
        # that would normally get silently merged away for sharing a
        # common approach corridor with something else.
    )

    print(f"{'Source':<22} {'Time(min)':<11} {'Dist(km)':<10} {'#ways':<7} {'ProvAvgEnjoy':<13}")
    print("-" * 70)
    for c in sorted(candidates, key=lambda c: -c["avg_enjoyment"]):
        num_ways = len(set(w["way_id"] for w in c["way_info"]))
        fits = "✓" if c["real_time_s"] <= widest_budget else "✗ (over budget)"
        print(f"{c['source']:<22} {round(c['real_time_s']/60, 1):<11} "
              f"{round(c['real_distance_m']/1000, 2):<10} {num_ways:<7} "
              f"{c['avg_enjoyment']:<13} {fits}")

    mam_tor_candidate = next((c for c in candidates if c["source"] == "via_way_164088231"), None)
    print()
    if mam_tor_candidate is None:
        print("Mam Tor Road's candidate is STILL missing even with deduplication disabled.")
        print("That rules out deduplication -- something else is preventing it from ever")
        print("being generated. Testing directly whether either leg (start->via_node or")
        print("via_node->end) has NO PATH AT ALL -- the real Mam Tor road has been")
        print("officially closed to through traffic since a 1970s landslip, so if OSM's")
        print("routing data reflects that closure, this could be a genuine disconnection,")
        print("not a scoring or search problem.\n")

        rep_nodes = path_module._representative_nodes_per_way(graph)
        candidate_nodes = rep_nodes.get(164088231, set())
        print(f"Candidate via-nodes for way_id=164088231: {candidate_nodes}\n")

        for via_node in candidate_nodes:
            if via_node in (start_node, end_node):
                continue
            leg1_path, leg1_time, _, _ = path_module.dijkstra_shortest_path(graph, start_node, via_node)
            leg2_path, leg2_time, _, _ = path_module.dijkstra_shortest_path(graph, via_node, end_node)
            print(f"via_node={via_node}:")
            print(f"  start -> via_node:  {'PATH FOUND, ' + str(round(leg1_time, 1)) + 's' if leg1_path else 'NO PATH AT ALL'}")
            print(f"  via_node -> end:    {'PATH FOUND, ' + str(round(leg2_time, 1)) + 's' if leg2_path else 'NO PATH AT ALL'}")
            if leg1_path is None or leg2_path is None:
                print("  -> THIS is why no candidate was generated: at least one leg is "
                      "genuinely unreachable, not a scoring/search failure.")
            print()
    else:
        print(f"Mam Tor Road's own candidate: {round(mam_tor_candidate['real_time_s']/60, 1)} min, "
              f"avg_enjoyment={mam_tor_candidate['avg_enjoyment']}, "
              f"{len(set(w['way_id'] for w in mam_tor_candidate['way_info']))} unique ways used")
        print("\nWays used by this candidate (name -- this shows exactly what's dragging")
        print("the average up or down):")
        seen = set()
        for w in mam_tor_candidate["way_info"]:
            if w["way_id"] in seen:
                continue
            seen.add(w["way_id"])
            score = provisional_scores.get(w["way_id"], {}).get("enjoyment_score", "?")
            print(f"  {w.get('name') or '(unnamed ' + w.get('highway', '?') + ')'}: provisional enjoyment={score}")