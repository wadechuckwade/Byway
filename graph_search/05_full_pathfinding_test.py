"""
Byway — Milestone 1.5, Step 3b: Real end-to-end pathfinding test
=====================================================================

The first time this project computes a route WITHOUT calling OSRM at
all — fetch the real road graph, find the nearest nodes to Tillington
and Haslemere, and run our own Dijkstra search between them.

This uses plain distance as the cost (no enjoyment scoring yet) — see
04_pathfinding.py's docstring for why proving this works correctly
first matters before adding any scoring complexity.

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
    print(f"  Graph has {len(node_coords)} nodes, {sum(len(e) for e in graph.values())} directed edges.\n")

    print(f"Finding nearest graph node to {TILLINGTON['name']}...")
    start_node, start_dist = path_module.find_nearest_node(
        TILLINGTON["lat"], TILLINGTON["lon"], node_coords
    )
    print(f"  Nearest node: {start_node}, {round(start_dist, 1)}m from the given coordinates.")

    print(f"\nFinding nearest graph node to {HASLEMERE['name']}...")
    end_node, end_dist = path_module.find_nearest_node(
        HASLEMERE["lat"], HASLEMERE["lon"], node_coords
    )
    print(f"  Nearest node: {end_node}, {round(end_dist, 1)}m from the given coordinates.")

    print(f"\nRunning our own Dijkstra search (TIME-based cost — fastest")
    print("route, using OSRM's own default speeds per road class)...")
    path, total_time_s, total_distance_m, way_info_list = path_module.dijkstra_shortest_path(
        graph, start_node, end_node
    )

    if path is None:
        print("\nNO PATH FOUND. This would mean either:")
        print("  - The two nearest nodes are in disconnected parts of the graph")
        print("  - A bug in graph construction or pathfinding")
        print("Worth investigating before going further.")
    else:
        print(f"\nPath found! {len(path)} nodes, "
              f"{round(total_distance_m / 1000, 2)} km, "
              f"~{round(total_time_s / 60, 1)} minutes (estimated, our own speed model).")

        # Group consecutive edges by REF (road number, e.g. "A272")
        # when present, falling back to NAME only when there's no ref
        # at all. FIX: an earlier version grouped by name alone, which
        # produced a confusing-looking "oscillation" between a local
        # street name and "(unnamed trunk)" for what real-world
        # investigation confirmed is genuinely the SAME numbered A-road
        # the whole way — OSM's name tag changes by village/stretch
        # even when ref stays constant, since ref is the stable
        # identifier a driver actually thinks in terms of ("take the
        # A272, then the A286"), not the decorative local name.
        road_sequence = []
        for w in way_info_list:
            ref = w.get("ref", "")
            name = w["name"]
            if ref:
                group_key = ref
                display = ref
            else:
                group_key = name or f"(unnamed {w['highway']})"
                display = group_key

            if not road_sequence or road_sequence[-1]["key"] != group_key:
                road_sequence.append({"key": group_key, "display": display, "local_names": set()})
            if name:
                road_sequence[-1]["local_names"].add(name)

        print("\nRoad sequence (grouped by road number where available):")
        for road in road_sequence:
            local_names = sorted(road["local_names"] - {road["display"]})
            if local_names:
                print(f"  - {road['display']}  (locally: {', '.join(local_names)})")
            else:
                print(f"  - {road['display']}")

    print(f"\n{'=' * 60}")
    print("This should now be the genuinely FASTEST route (time-based),")
    print("not the shortest-distance one that turned out to resemble")
    print("the scenic route in earlier testing. Does this look like")
    print("the real fastest way from Tillington to Haslemere?")
    print(f"{'=' * 60}")