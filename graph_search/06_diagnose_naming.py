"""
Byway — Diagnostic: investigate the Midhurst Road / unnamed trunk
oscillation seen in the real Tillington -> Haslemere route.

WHY THIS EXISTS: the real route alternated between "Midhurst Road"
and "(unnamed trunk)" four times, and revisited "Midhurst Road" three
times under different surrounding names. This could be: (a) genuinely
different physical roads with confusingly similar names, (b) the same
real road (e.g. the A272) split across many OSM ways where only some
have a name tag, with the route's `ref` tag (e.g. "A272") not
currently being captured at all, or (c) something else. This script
prints the FULL way info for every edge along the route, undeduped,
so we can see exactly what's going on rather than guess.
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


if __name__ == "__main__":
    TILLINGTON = {"name": "Tillington", "lat": 50.9896, "lon": -0.6313}
    HASLEMERE = {"name": "Haslemere", "lat": 51.089, "lon": -0.710}

    margin_degrees = 0.03
    bbox = {
        "min_lat": min(TILLINGTON["lat"], HASLEMERE["lat"]) - margin_degrees,
        "max_lat": max(TILLINGTON["lat"], HASLEMERE["lat"]) + margin_degrees,
        "min_lon": min(TILLINGTON["lon"], HASLEMERE["lon"]) - margin_degrees,
        "max_lon": max(TILLINGTON["lon"], HASLEMERE["lon"]) + margin_degrees,
    }

    print("Fetching real road network data...")
    overpass_data = fetch_module.fetch_all_roads_in_bbox(bbox)

    print("\nBuilding routable graph...")
    graph, node_coords = build_module.build_graph_from_overpass_data(overpass_data)

    start_node, _ = path_module.find_nearest_node(TILLINGTON["lat"], TILLINGTON["lon"], node_coords)
    end_node, _ = path_module.find_nearest_node(HASLEMERE["lat"], HASLEMERE["lon"], node_coords)

    path, total_time_s, total_distance_m, way_info_list = path_module.dijkstra_shortest_path(
        graph, start_node, end_node
    )

    if path is None:
        print("No path found — cannot diagnose.")
    else:
        print(f"\nFull, UNDEDUPLICATED edge-by-edge breakdown ({len(way_info_list)} edges):\n")
        print(f"{'#':>4} {'way_id':>12} {'ref':<8} {'name':<25} {'highway':<15}")
        print("-" * 70)
        for i, w in enumerate(way_info_list):
            name_display = w["name"] or "(no name tag)"
            ref_display = w.get("ref", "") or "-"
            print(f"{i:>4} {w['way_id']:>12} {ref_display:<8} {name_display:<25} {w['highway']:<15}")

        # The actual theory to test: does 'ref' stay CONSISTENT (e.g.
        # always "A272") across the segments that alternate between
        # "Midhurst Road" and "(unnamed trunk)" in the name field? If
        # so, this confirms it's genuinely the same road the whole
        # time, just with inconsistent local name tagging in OSM —
        # not a routing bug.
        refs_seen = [w.get("ref", "") for w in way_info_list if w.get("ref")]
        unique_refs = set(refs_seen)
        print(f"\nNon-empty 'ref' values seen along the route: {unique_refs}")
        print("(If this is a small, consistent set like {'A272'}, it confirms")
        print("the name oscillation was just inconsistent OSM name tagging")
        print("on what is genuinely the same numbered road throughout.)")

        way_ids_in_order = [w["way_id"] for w in way_info_list]
        unique_way_ids = set(way_ids_in_order)
        print(f"\nTotal edges traversed: {len(way_ids_in_order)}")
        print(f"Unique way_ids among them: {len(unique_way_ids)}")
        print("(If unique count is much lower than total edges, we're")
        print("legitimately passing through the same way's internal")
        print("points repeatedly, which is normal for a long way split")
        print("into many small edges — NOT a bug.)")