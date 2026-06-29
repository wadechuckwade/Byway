"""
Byway — Direct check: is Lurgashall (the specific village the user
knows and expects to see) even being targeted as a candidate, and if
so, why does it keep losing to the Dial Green Lane/Jobson's Lane/
Tennyson Lane corridor across every scoring formula iteration this
session?

WHY THIS EXISTS: that same corridor has won EVERY Tillington/Haslemere
test this entire session, across the original Milestone 1 formula,
the ScenicOrNot-added formula, the bug-fixed formula, and now the
land-cover-replacing-water/forest formula. That consistency itself is
informative -- either Lurgashall genuinely isn't as good a route by
the formula's own logic (worth knowing, and why), or something
mechanical is preventing it from ever being fairly compared (worth
finding and fixing) -- same spirit as the Winnats Pass diagnostic.

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
path_module = _import_from_path("pathfinding", os.path.join(_this_dir, "04_pathfinding.py"))
score_module = _import_from_path("score_graph_enjoyment", os.path.join(_this_dir, "07_score_graph_enjoyment.py"))


if __name__ == "__main__":
    TILLINGTON = {"name": "Tillington", "lat": 50.9896, "lon": -0.6313}
    HASLEMERE = {"name": "Haslemere", "lat": 51.089, "lon": -0.710}
    # Upperton and Lurgashall themselves -- the specific villages the
    # user expects the route to pass through.
    UPPERTON = {"name": "Upperton", "lat": 50.9974, "lon": -0.6498}
    LURGASHALL = {"name": "Lurgashall", "lat": 51.0291, "lon": -0.6661}

    margin_degrees = 0.03
    bbox = {
        "min_lat": min(TILLINGTON["lat"], HASLEMERE["lat"]) - margin_degrees,
        "max_lat": max(TILLINGTON["lat"], HASLEMERE["lat"]) + margin_degrees,
        "min_lon": min(TILLINGTON["lon"], HASLEMERE["lon"]) - margin_degrees,
        "max_lon": max(TILLINGTON["lon"], HASLEMERE["lon"]) + margin_degrees,
    }

    print("Fetching real road network data...")
    overpass_data = fetch_module.fetch_all_roads_in_bbox(bbox)
    graph, node_coords = build_module.build_graph_from_overpass_data(overpass_data)
    print(f"  Graph has {len(node_coords)} nodes, {sum(len(e) for e in graph.values())} directed edges.\n")

    print("Scoring provisionally...")
    provisional_scores, ways, region_features = score_module.score_all_ways_provisional(graph, node_coords)

    # Step 1: is Lurgashall (and Upperton) actually present in our
    # Conservation Areas data for this bbox at all? (Confirmed working
    # earlier this session, but verify directly for THIS specific run
    # rather than assume it still is.)
    conservation_areas = region_features.get("conservation_areas", [])
    lurgashall_entries = [c for c in conservation_areas if "lurgashall" in (c.get("name") or "").lower()]
    upperton_entries = [c for c in conservation_areas if "upperton" in (c.get("name") or "").lower()]
    print(f"\nLurgashall found in Conservation Areas data: {bool(lurgashall_entries)} ({lurgashall_entries})")
    print(f"Upperton found in Conservation Areas data: {bool(upperton_entries)} ({upperton_entries})")

    # Step 2: find the way(s) nearest to Lurgashall/Upperton directly,
    # and report their PROVISIONAL rank and scores -- same technique
    # used for the Mam Tor/Winnats Pass investigation.
    start_node, _ = path_module.find_nearest_node(TILLINGTON["lat"], TILLINGTON["lon"], node_coords)
    end_node, _ = path_module.find_nearest_node(HASLEMERE["lat"], HASLEMERE["lon"], node_coords)

    sorted_by_score = sorted(provisional_scores.items(), key=lambda kv: -kv[1]["enjoyment_score"])
    rank_by_way_id = {wid: rank for rank, (wid, _) in enumerate(sorted_by_score, start=1)}

    for label, place in [("Lurgashall", LURGASHALL), ("Upperton", UPPERTON)]:
        via_node, dist_m = path_module.find_nearest_node(place["lat"], place["lon"], node_coords)
        print(f"\n--- Nearest node to {label}: node={via_node}, {round(dist_m, 1)}m away ---")
        # Find which way(s) actually touch this node, for rank lookup.
        nearby_way_ids = set()
        for edges in graph.values():
            for neighbor_id, _, _, way_info in edges:
                if neighbor_id == via_node or neighbor_id == via_node:
                    nearby_way_ids.add(way_info["way_id"])
        for wid in nearby_way_ids:
            if wid in provisional_scores:
                s = provisional_scores[wid]
                rank = rank_by_way_id.get(wid, "?")
                print(f"  way_id={wid}: rank {rank} of {len(provisional_scores)}, "
                      f"enjoyment={score_module.to_ten(s['enjoyment_score'])}/10, "
                      f"driving={score_module.to_ten(s['driving_enjoyment'])}, "
                      f"scenery={score_module.to_ten(s['scenery'])}, interest={score_module.to_ten(s['interest'])}")

        # Step 3: directly force a route through this specific node and
        # compare its REAL avg_enjoyment against what actually won.
        leg1 = path_module.dijkstra_shortest_path(graph, start_node, via_node)
        leg2 = path_module.dijkstra_shortest_path(graph, via_node, end_node)
        combined_path, combined_time, combined_dist, combined_wi = path_module._combine_path_legs(leg1, leg2)
        if combined_path is None:
            print(f"  NO PATH exists via this node at all (possibly disconnected, like the Mam Tor case).")
            continue

        used_way_ids = set(w["way_id"] for w in combined_wi)
        refined = score_module.refine_scores_with_elevation(used_way_ids, ways, provisional_scores, verbose=False)
        final_scores = dict(provisional_scores)
        final_scores.update(refined)

        three_route = _import_from_path("three_route", os.path.join(_this_dir, "08_three_route_system.py"))
        avg_enjoyment, best_stretch, _ = three_route.fully_scored_route(combined_wi, final_scores)
        composition = three_route.landcover_composition_along_path(combined_wi, final_scores)

        print(f"  Forcing a route via {label} directly:")
        print(f"    Time: {round(combined_time/60, 1)} min, Distance: {round(combined_dist/1000, 2)} km")
        print(f"    avg_enjoyment={score_module.to_ten(avg_enjoyment)}/10, "
              f"best_stretch={score_module.to_ten(best_stretch)}/10")
        comp_str = ", ".join(f"{k}={round(v*100,1)}%" for k, v in composition.items() if v > 0)
        print(f"    Land cover: {comp_str}")

    print("\n" + "=" * 60)
    print("Compare the above against the ACTUAL winning route's numbers")
    print("(avg=2.5/10, best_stretch=3.1/10, Forest 49.7%/Agri 45.8%/Urban 4.1%)")
    print("from the real run -- this tells us directly whether Lurgashall")
    print("loses fairly (lower score, real reason) or never got a fair shot.")
    print("=" * 60)
