"""
Byway — Milestone 1.5, Step 2b: Real end-to-end graph build test
"""

import sys
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

fetch_all_roads_in_bbox = fetch_module.fetch_all_roads_in_bbox
build_graph_from_overpass_data = build_module.build_graph_from_overpass_data
summarize_graph = build_module.summarize_graph


if __name__ == "__main__":
    TILLINGTON = {"lat": 50.9896, "lon": -0.6313}
    HASLEMERE = {"lat": 51.089, "lon": -0.710}

    margin_degrees = 0.03
    bbox = {
        "min_lat": min(TILLINGTON["lat"], HASLEMERE["lat"]) - margin_degrees,
        "max_lat": max(TILLINGTON["lat"], HASLEMERE["lat"]) + margin_degrees,
        "min_lon": min(TILLINGTON["lon"], HASLEMERE["lon"]) - margin_degrees,
        "max_lon": max(TILLINGTON["lon"], HASLEMERE["lon"]) + margin_degrees,
    }

    print("Fetching real road network data...")
    overpass_data = fetch_all_roads_in_bbox(bbox)

    print("\nBuilding routable graph from real data...")
    graph, node_coords = build_graph_from_overpass_data(overpass_data)

    print()
    summarize_graph(graph, node_coords)

    print(f"\n{'=' * 60}")
    print("If this graph looks well-connected, we're ready to find")
    print("the nearest graph nodes to Tillington and Haslemere and")
    print("try our own pathfinding search next.")
    print(f"{'=' * 60}")
