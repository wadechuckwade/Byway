"""
Byway — Milestone 0: Data Pipeline Proof of Concept
=====================================================

What this script does, in plain terms:
1. Takes our test route (Madehurst -> Goodwood House)
2. Asks OpenStreetMap's free Overpass API for every road in a box around
   that route
3. Asks a free routing engine (OSRM's public demo server) for the
   fastest driving route between the two points
4. Prints out what it found, so we can eyeball whether the data looks
   sane before we build any scoring on top of it

Nothing here is scored yet. This is purely "can we fetch real data
about this real place" — Milestone 0 from the development plan.

Requires: pip install requests
"""

import requests
import json

# -----------------------------------------------------------------------
# Our test locations (confirmed via web search against real sources)
# -----------------------------------------------------------------------
START = {"name": "Madehurst", "lat": 50.8844, "lon": -0.6008}
END = {"name": "Goodwood House", "lat": 50.8722, "lon": -0.7392}


def get_fastest_route(start, end):
    """
    Ask OSRM (Open Source Routing Machine) for the fastest driving route.
    OSRM's public demo server is free and requires no API key.

    Note: OSRM wants coordinates as (longitude, latitude) — the opposite
    order from how we usually say them. Easy thing to get backwards, so
    flagging it here.
    """
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{start['lon']},{start['lat']};{end['lon']},{end['lat']}"
        f"?overview=full&geometries=geojson&steps=true"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def get_roads_in_area(start, end, buffer_degrees=0.02):
    """
    Ask the Overpass API for all roads in a bounding box around our
    route. buffer_degrees adds some margin around the start/end points
    so we capture the scenic detour roads too, not just a straight line.

    Overpass uses its own query language (Overpass QL). The query below
    asks for every 'highway' way (OSM's term for any road/track) within
    the bounding box, along with useful tags like name, surface, and
    road classification.
    """
    min_lat = min(start["lat"], end["lat"]) - buffer_degrees
    max_lat = max(start["lat"], end["lat"]) + buffer_degrees
    min_lon = min(start["lon"], end["lon"]) - buffer_degrees
    max_lon = max(start["lon"], end["lon"]) + buffer_degrees

    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=30,
        headers={"User-Agent": "BywayApp-DevelopmentPrototype/0.1"},
    )
    response.raise_for_status()
    return response.json()


def get_route_via_waypoint(start, waypoint, end):
    """
    Ask OSRM for a route that passes through a specific waypoint, not
    just the single fastest path. This is how we force the scenic
    detour (e.g. via Eartham) instead of letting OSRM pick whatever it
    thinks is fastest overall.

    Coordinates are joined in order: start ; waypoint ; end.
    """
    coords = (
        f"{start['lon']},{start['lat']};"
        f"{waypoint['lon']},{waypoint['lat']};"
        f"{end['lon']},{end['lat']}"
    )
    url = (
        f"https://router.project-osrm.org/route/v1/driving/{coords}"
        f"?overview=full&geometries=geojson&steps=true"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def summarize_route(route_data):
    """Pull out the headline numbers from an OSRM route response."""
    route = route_data["routes"][0]
    distance_km = route["distance"] / 1000
    duration_min = route["duration"] / 60

    road_names = []
    for leg in route["legs"]:
        for step in leg["steps"]:
            name = step.get("name", "").strip()
            if name and name not in road_names:
                road_names.append(name)

    return {
        "distance_km": round(distance_km, 1),
        "duration_min": round(duration_min, 1),
        "roads_used": road_names,
    }


def summarize_roads(osm_data):
    """Count and classify the roads Overpass found in the area."""
    named_roads = {}
    for element in osm_data.get("elements", []):
        if element["type"] != "way":
            continue
        tags = element.get("tags", {})
        name = tags.get("name")
        highway_type = tags.get("highway", "unknown")
        if name:
            named_roads[name] = highway_type
    return named_roads


if __name__ == "__main__":
    print(f"Route: {START['name']} -> {END['name']}\n")

    print("--- Fetching fastest route (OSRM) ---")
    try:
        route_data = get_fastest_route(START, END)
        summary = summarize_route(route_data)
        print(f"Distance: {summary['distance_km']} km")
        print(f"Duration: {summary['duration_min']} minutes")
        print(f"Roads used: {summary['roads_used']}\n")
    except Exception as e:
        print(f"Could not fetch route: {e}\n")

    print("--- Fetching scenic route via Eartham (OSRM, forced waypoint) ---")
    EARTHAM = {"name": "Eartham", "lat": 50.8769, "lon": -0.6667}
    try:
        scenic_route_data = get_route_via_waypoint(START, EARTHAM, END)
        scenic_summary = summarize_route(scenic_route_data)
        print(f"Distance: {scenic_summary['distance_km']} km")
        print(f"Duration: {scenic_summary['duration_min']} minutes")
        print(f"Roads used: {scenic_summary['roads_used']}\n")

        print("--- Comparison ---")
        extra_km = scenic_summary['distance_km'] - summary['distance_km']
        extra_min = scenic_summary['duration_min'] - summary['duration_min']
        print(f"Scenic route is {extra_km:+.1f} km and {extra_min:+.1f} minutes "
              f"vs. the fastest route.\n")
    except Exception as e:
        print(f"Could not fetch scenic route: {e}\n")

    print("--- Fetching named roads in the area (Overpass/OSM) ---")
    try:
        osm_data = get_roads_in_area(START, END)
        named_roads = summarize_roads(osm_data)
        print(f"Found {len(named_roads)} named roads in the area:")
        for name, road_type in sorted(named_roads.items()):
            print(f"  - {name} ({road_type})")
    except Exception as e:
        print(f"Could not fetch OSM road data: {e}\n")
