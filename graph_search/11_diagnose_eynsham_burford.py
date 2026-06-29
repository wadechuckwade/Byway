"""
Byway — Discovery Test #2: Eynsham -> Burford (clean through-route case)
==========================================================================

WHY THIS TEST EXISTS, AND WHY IT'S DIFFERENT FROM THE MAM TOR TEST:
the previous discovery test (10_diagnose_hathersage_castleton.py)
came back completely flat — source="direct" for all three
routes, no detour found worth taking at any budget. But that test had
a real confound: Mam Tor sits WEST of Castleton while Hathersage sits
EAST of it, so "via Mam Tor" from Hathersage to Castleton means
driving PAST Castleton and back — a loop (Milestone 1's own numbers
confirm this: +128% distance, +10.4 min for that detour), not a
genuine second through-route the way Upperton/Lurgashall was a real
alternate path between Tillington and Haslemere. Refusing a 20+
minute there-and-back loop might be the CORRECT call, not a failure —
so that result didn't cleanly test discovery either way.

Eynsham -> Burford via Witney is a clean genuine through-route
alternative (not a loop) AND already has a CONFIRMED formula win on
record: Milestone 1's hand-specified-waypoint test found this route
"outscored the direct A40-heavy route on all three categories" —
Driving Enjoyment, Scenery, and History & Culture. That removes one
whole axis of uncertainty: we already know the SCORING formula
approves of this route when pointed at it. The only open question
left is whether DISCOVERY (no waypoints, free search of the whole
area) finds and prefers something resembling it on its own.

If discovery finds/prefers a Witney-ish route here, that's a real,
clean confirmation the search mechanism works when there's a genuine
signal to find — strengthening the case that Tillington/Haslemere's
flat result was "modest real-world gap," not "broken discovery." If
discovery STILL can't find it even here — a case the formula itself
already independently confirmed it likes — that's much stronger,
cleaner evidence of a problem in the DISCOVERY mechanism specifically
(not the scoring formula, which this test doesn't re-litigate).

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
    EYNSHAM = {"name": "Eynsham", "lat": 51.7808, "lon": -1.3745}
    BURFORD = {"name": "Burford", "lat": 51.8071, "lon": -1.6368}

    margin_degrees = 0.03
    bbox = {
        "min_lat": min(EYNSHAM["lat"], BURFORD["lat"]) - margin_degrees,
        "max_lat": max(EYNSHAM["lat"], BURFORD["lat"]) + margin_degrees,
        "min_lon": min(EYNSHAM["lon"], BURFORD["lon"]) - margin_degrees,
        "max_lon": max(EYNSHAM["lon"], BURFORD["lon"]) + margin_degrees,
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

    start_node, _ = path_module.find_nearest_node(EYNSHAM["lat"], EYNSHAM["lon"], node_coords)
    end_node, _ = path_module.find_nearest_node(BURFORD["lat"], BURFORD["lon"], node_coords)

    # NO WAYPOINTS. This is genuine discovery: does the search find
    # something resembling the Witney / Windrush valley route on its
    # own, given only a start and end point?
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
    print("Does any road in 'Roads:' above mention Witney, or run")
    print("through the Windrush valley, for Compromise or Max")
    print("Enjoyment — rather than just the direct A40? Unlike the")
    print("Mam Tor test, we already KNOW the scoring formula approves")
    print("of this route (Milestone 1 confirmed it). If discovery still")
    print("can't find/prefer it here, that's evidence pointing at the")
    print("discovery mechanism itself, not the scoring formula.")
    print(f"{'=' * 60}")
    for label in ["Direct (fastest)", "Compromise", "Max Enjoyment"]:
        if label in results:
            r = results[label]
            print(f"{label}: {r['time_min']} min, {r['distance_km']} km, "
                  f"enjoyment={r['avg_enjoyment']}, source={r['source']}")
