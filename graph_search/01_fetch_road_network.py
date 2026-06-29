"""
Byway — Milestone 1.5, Step 1: Prove we can fetch a real road GRAPH
=======================================================================

What this does, in plain terms:
Before attempting any graph-building or pathfinding logic, this just
proves the most basic, riskiest assumption: can we fetch ALL the
roads in a bounding box from Overpass (not just one route's
geometry, like every previous script in this project), and is the
result a manageable size?

This is deliberately the smallest possible first step. No graph
construction, no scoring, no pathfinding yet — just: fetch, count,
sanity-check.

Test case: Tillington, West Sussex -> Haslemere, Surrey — a real
~13km trip the user knows personally and can validate results
against directly.

Network note: needs real internet access to overpass-api.de.
"""

import time
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "BywayApp-DevelopmentPrototype/0.1 (research prototype)"


def fetch_all_roads_in_bbox(bbox):
    """
    Fetch every road (highway=*) within a bounding box from Overpass,
    including full node geometry for each way.

    Unlike every previous Overpass query in this project (which asked
    for POINTS near a route), this asks for entire WAYS — the actual
    road network itself — which is what's needed to build a real
    routable graph rather than just score an already-known route.

    Returns the raw Overpass JSON response.
    """
    b = bbox
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"]
        ({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
    );
    out body;
    >;
    out skel qt;
    """
    print("Querying Overpass for all roads in the bounding box (this may take a moment)...")
    t_start = time.time()
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={"User-Agent": USER_AGENT},
        timeout=90,
    )
    response.raise_for_status()
    elapsed = round(time.time() - t_start, 1)
    print(f"  Got response in {elapsed} seconds.\n")
    return response.json()


def summarize_road_network(overpass_data):
    """
    Basic sanity-check summary of what Overpass gave us — no graph
    construction yet, just understanding the shape of the raw data.
    """
    ways = [el for el in overpass_data["elements"] if el["type"] == "way"]
    nodes = [el for el in overpass_data["elements"] if el["type"] == "node"]

    print(f"Total ways (road segments): {len(ways)}")
    print(f"Total nodes (points/intersections): {len(nodes)}")

    highway_types = {}
    for way in ways:
        h_type = way.get("tags", {}).get("highway", "unknown")
        highway_types[h_type] = highway_types.get(h_type, 0) + 1

    print("\nRoad types found:")
    for h_type, count in sorted(highway_types.items(), key=lambda x: -x[1]):
        print(f"  {h_type}: {count}")

    node_counts = [len(way.get("nodes", [])) for way in ways]
    if node_counts:
        avg_nodes_per_way = sum(node_counts) / len(node_counts)
        print(f"\nAverage nodes per way: {round(avg_nodes_per_way, 1)}")
        print(f"Min/Max nodes per way: {min(node_counts)} / {max(node_counts)}")

    oneway_count = sum(1 for way in ways if way.get("tags", {}).get("oneway") == "yes")
    print(f"\nWays explicitly tagged oneway=yes: {oneway_count} of {len(ways)}")

    return {"ways": ways, "nodes": nodes}


if __name__ == "__main__":
    TILLINGTON = {"lat": 50.9896, "lon": -0.6313}
    HASLEMERE = {"lat": 51.089, "lon": -0.710}

    margin_degrees = 0.03  # roughly 2-3km margin at this latitude
    bbox = {
        "min_lat": min(TILLINGTON["lat"], HASLEMERE["lat"]) - margin_degrees,
        "max_lat": max(TILLINGTON["lat"], HASLEMERE["lat"]) + margin_degrees,
        "min_lon": min(TILLINGTON["lon"], HASLEMERE["lon"]) - margin_degrees,
        "max_lon": max(TILLINGTON["lon"], HASLEMERE["lon"]) + margin_degrees,
    }
    print(f"Bounding box: {bbox}\n")

    overpass_data = fetch_all_roads_in_bbox(bbox)
    summary = summarize_road_network(overpass_data)

    print(f"\n{'=' * 60}")
    print("If this looks like a sane, manageable road network (not")
    print("zero results, not hundreds of thousands of ways), we're")
    print("ready to move to actual graph construction next.")
    print(f"{'=' * 60}")
