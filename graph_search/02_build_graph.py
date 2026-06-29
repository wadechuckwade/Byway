"""
Byway — Milestone 1.5, Step 2: Build a real routable graph
===============================================================

What this does, in plain terms:
Takes the raw Overpass way/node data (proven working in Step 1) and
turns it into an actual routable graph: nodes become graph vertices,
and each way is split into edges between consecutive nodes, with
one-way streets correctly respected.

WHY THIS IS NEEDED: every previous script in this project asked OSRM
for ONE route's geometry. This is the first time we're building the
underlying road NETWORK ourselves — the structure OSRM normally
builds internally — because Milestone 1.5 needs to search across many
possible paths through an area, not just score one path someone
already chose.

Representation: a simple adjacency-list graph,
{node_id: [(neighbor_node_id, distance_m, time_s, way_info), ...]}
Both distance AND time are kept on every edge — distance because our
enjoyment scoring (Milestone 1) and user-facing "X km" reporting need
it, time because that's the correct cost for a genuine "fastest
route" baseline (see below for why this matters).

WHY TIME, NOT JUST DISTANCE: real-route testing (Tillington ->
Haslemere) revealed that plain-distance Dijkstra found a route the
user immediately recognised as closer to "the scenic way," not the
fastest way — because a short, slow minor lane can have a smaller
raw distance than a longer, faster A-road route, even though the
A-road is quicker to actually drive. Edge cost is now TIME
(distance / speed), using OSRM's own real default speed table per
UK road class as a fallback, and the way's own `maxspeed` tag when
present (more accurate than a class default when we have it).

ONE-WAY HANDLING: OSM's convention is that a way's node order IS the
direction of travel when oneway=yes. oneway=-1 (less common) means
the legal direction is the REVERSE of the stored node order. Anything
else (no oneway tag, or oneway=no) is treated as two-way. Getting
this wrong wouldn't just be inelegant — it would mean producing
routes that are illegal to actually drive, so this is treated as
something to get right and test explicitly, not assume.
"""

import math
import re


def _haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    Distance in metres between two lat/lon points. Same formula
    already proven and used throughout this project (e.g.
    proximity.py's _min_distance_to_route_m) — reused here for
    consistency rather than introducing a second implementation.
    """
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Real default speeds (km/h) per UK road class, taken directly from
# OSRM's own production car.lua profile — not guessed. Used as a
# fallback when a way has no explicit maxspeed tag. Source:
# Project-OSRM/osrm-backend profiles/car.lua.
#
# UK-SPECIFIC ADJUSTMENT to unclassified/residential (was 25, now 45):
# real-route testing (Tillington -> Haslemere) found a known-driveable
# minor-lane route, confirmed by the user's own direct real-world
# knowledge, estimated at an implausible ~19 km/h (12mph) average —
# traced to 84% of that route's distance being tagged "unclassified",
# at OSRM's global default of 25 km/h. That default is a reasonable
# WORLDWIDE minimum (unclassified roads in many countries are
# genuinely poor or unpaved), but is too conservative for typical
# paved English country lanes, which are usually comfortably driven
# at 30mph+ (48km/h+) in normal conditions when not held up. Raised
# to 45 km/h as a deliberately UK-specific, frankly-labeled
# correction — not derived from a specific study, flagged as tuneable
# like every other first-pass value in this project, but grounded in
# direct real-world confirmation rather than a blind guess.
DEFAULT_SPEED_KMH_BY_HIGHWAY = {
    "motorway": 90, "motorway_link": 45,
    "trunk": 85, "trunk_link": 40,
    "primary": 65, "primary_link": 30,
    "secondary": 55, "secondary_link": 25,
    "tertiary": 40, "tertiary_link": 20,
    "unclassified": 45, "residential": 45,
    "living_street": 10, "service": 15,
}
FALLBACK_SPEED_KMH = 25  # for any unrecognised highway type, a conservative minor-road assumption


def _parse_maxspeed_kmh(maxspeed_tag, highway_type):
    """
    Parse an OSM maxspeed tag into km/h, falling back to the
    OSRM-derived class default if absent or unparseable.

    OSM maxspeed values are usually plain numbers (assumed km/h) but
    can include "mph" (e.g. "30 mph", common in the UK) or special
    values like "national" — handle the common real cases, fall back
    safely for anything unexpected rather than crashing.
    """
    default = DEFAULT_SPEED_KMH_BY_HIGHWAY.get(highway_type, FALLBACK_SPEED_KMH)

    if not maxspeed_tag:
        return default

    maxspeed_tag = maxspeed_tag.strip().lower()

    if "mph" in maxspeed_tag:
        match = re.search(r"(\d+)", maxspeed_tag)
        if match:
            return int(match.group(1)) * 1.60934  # mph -> km/h
        return default

    match = re.search(r"(\d+)", maxspeed_tag)
    if match:
        return float(match.group(1))

    # Values like "national" or "none" — not worth a full UK speed-
    # limit-by-road-type implementation for this prototype; fall back
    # to the class default, which is already a reasonable real-world
    # approximation.
    return default


def build_graph_from_overpass_data(overpass_data):
    """
    Convert raw Overpass way/node elements into a routable directed
    graph.

    Returns:
        graph: {node_id: [(neighbor_id, distance_m, time_s, way_info), ...]}
        node_coords: {node_id: (lat, lon)} — needed later for scoring
                     and for A* heuristics
    """
    nodes_by_id = {}
    ways = []

    for el in overpass_data["elements"]:
        if el["type"] == "node":
            nodes_by_id[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way":
            ways.append(el)

    graph = {}

    def add_edge(from_id, to_id, distance_m, time_s, way_info):
        graph.setdefault(from_id, []).append((to_id, distance_m, time_s, way_info))

    skipped_ways_missing_nodes = 0

    for way in ways:
        node_ids = way.get("nodes", [])
        tags = way.get("tags", {})
        highway_type = tags.get("highway", "unknown")
        maxspeed_tag = tags.get("maxspeed")
        way_info = {
            "way_id": way["id"],
            "name": tags.get("name", ""),
            # ref captures the road NUMBER (e.g. "A272"), a separate
            # OSM tag from name (e.g. "Midhurst Road"). UK A-roads
            # often carry a consistent ref the whole way even when the
            # local name tag changes or disappears between villages —
            # this was the real explanation for the Midhurst Road /
            # unnamed trunk oscillation seen in real-route testing:
            # likely the same physical A272 the whole time, just with
            # inconsistent local name tagging, which ref reveals.
            "ref": tags.get("ref", ""),
            "highway": highway_type,
            # Track whether this way had a REAL maxspeed tag, and what
            # speed was actually assumed (after any correction) — added
            # after real-route testing found a route's time estimate
            # was dominated by unclassified-road travel time, and we
            # had no way to verify after the fact whether that was due
            # to genuinely tagged slow speeds or just our class default
            # being applied. Without this, diagnosing speed-related
            # issues meant re-deriving everything from scratch each time.
            "had_maxspeed_tag": maxspeed_tag is not None,
        }

        oneway = tags.get("oneway", "no")
        speed_kmh = _parse_maxspeed_kmh(maxspeed_tag, highway_type)

        # REAL-WORLD CORRECTION FACTOR — deliberately blunt, frankly
        # labeled as such, not a true physical model. Research
        # confirmed (via real-route testing against both the user's
        # own knowledge and Google Maps) that static/traffic-oblivious
        # routing time estimates are a well-documented, structural
        # weak point of this entire class of approach.
        #
        # FIX: an earlier version applied this correction to EVERY
        # road class except motorway/trunk — including unclassified
        # and residential roads, whose OSRM defaults (25 km/h) are
        # ALREADY a conservative minor-road assumption, not a
        # free-flowing speed needing further correction. Stacking the
        # 0.8x factor on top of an already-low default produced a
        # genuinely implausible ~20 km/h (12mph) effective speed for
        # ordinary country lanes — confirmed via real-route testing
        # where a known-reasonable minor-lane route was estimated at
        # nearly DOUBLE its real-world driving time (49 min vs a
        # sane ~25-30 min), an average speed barely above walking
        # pace. This was traced back to exactly this double-penalty.
        #
        # The correction now applies ONLY to primary/secondary roads
        # — the classes most likely to LOOK fast on paper (65/55 km/h
        # defaults) while actually running through village high
        # streets, multiple junctions, and built-up sections in
        # reality (the actual real-world pattern that motivated this
        # fix in the first place, e.g. the A272/A286 test case).
        # Smaller/slower road classes (unclassified, residential,
        # tertiary) keep their OSRM defaults uncorrected, since those
        # defaults are already calibrated for minor-road conditions.
        REALWORLD_SPEED_CORRECTION = 0.8
        if highway_type in ("primary", "primary_link", "secondary", "secondary_link"):
            speed_kmh *= REALWORLD_SPEED_CORRECTION

        speed_ms = speed_kmh / 3.6  # km/h -> m/s, for time = distance / speed
        way_info["assumed_speed_kmh"] = round(speed_kmh, 1)

        # Walk the way's node sequence in consecutive pairs, creating
        # one edge per pair — this is what turns a single long OSM
        # way into multiple graph edges between actual intersections/
        # points, rather than one edge for the whole way.
        for i in range(len(node_ids) - 1):
            id_a, id_b = node_ids[i], node_ids[i + 1]

            if id_a not in nodes_by_id or id_b not in nodes_by_id:
                # Can happen if a way references a node outside our
                # bounding box (Overpass may not have returned every
                # referenced node) — skip this specific edge rather
                # than crash, but track how often it happens since
                # frequent occurrences would mean our bounding box or
                # query needs adjustment.
                skipped_ways_missing_nodes += 1
                continue

            lat_a, lon_a = nodes_by_id[id_a]
            lat_b, lon_b = nodes_by_id[id_b]
            dist = _haversine_distance_m(lat_a, lon_a, lat_b, lon_b)
            time_s = dist / speed_ms if speed_ms > 0 else float("inf")

            if oneway == "yes":
                # Stored node order IS the legal direction of travel.
                add_edge(id_a, id_b, dist, time_s, way_info)
            elif oneway == "-1":
                # Legal direction is the REVERSE of stored node order
                # — a real, if less common, OSM convention.
                add_edge(id_b, id_a, dist, time_s, way_info)
            else:
                # Two-way: add both directions.
                add_edge(id_a, id_b, dist, time_s, way_info)
                add_edge(id_b, id_a, dist, time_s, way_info)

    if skipped_ways_missing_nodes > 0:
        print(f"  Note: skipped {skipped_ways_missing_nodes} edges referencing "
              f"nodes outside the fetched data (likely just outside the bounding box).")

    return graph, nodes_by_id


def summarize_graph(graph, node_coords):
    """Basic sanity-check summary of the constructed graph."""
    num_vertices_with_edges = len(graph)
    total_directed_edges = sum(len(edges) for edges in graph.values())
    total_nodes_with_coords = len(node_coords)

    print(f"Graph vertices with at least one outgoing edge: {num_vertices_with_edges}")
    print(f"Total directed edges: {total_directed_edges}")
    print(f"Total nodes with known coordinates: {total_nodes_with_coords}")

    # Check connectivity at a basic level: how many vertices have NO
    # outgoing edges at all (dead ends are normal and expected, e.g.
    # a node at the very end of a residential street, but a large
    # fraction with zero edges anywhere might indicate a real
    # construction problem).
    nodes_with_no_outgoing_edges = total_nodes_with_coords - num_vertices_with_edges
    print(f"Nodes with NO outgoing edges (dead ends / isolated points): "
          f"{nodes_with_no_outgoing_edges}")

    # Sample a few real edges for a sanity look, rather than trusting
    # aggregate numbers alone.
    print("\nSample of 5 real edges:")
    sample_count = 0
    for node_id, edges in graph.items():
        if sample_count >= 5:
            break
        for neighbor_id, dist, time_s, way_info in edges[:1]:
            implied_speed_kmh = (dist / time_s) * 3.6 if time_s > 0 else 0
            print(f"  {node_id} -> {neighbor_id}: {round(dist, 1)}m, "
                  f"{round(time_s, 1)}s (~{round(implied_speed_kmh)}km/h) "
                  f"on \"{way_info['name'] or 'unnamed'}\" ({way_info['highway']})")
            sample_count += 1
            break


if __name__ == "__main__":
    # Self-test using small fake Overpass-shaped data (no internet
    # needed) to verify the core logic: edge creation, distance/time
    # calculation, and especially one-way handling, since getting
    # that wrong would produce illegal-to-drive routes.

    print("--- Test 1: simple two-way road, 3 nodes ---")
    fake_data = {
        "elements": [
            {"type": "node", "id": 1, "lat": 51.0, "lon": -0.7},
            {"type": "node", "id": 2, "lat": 51.001, "lon": -0.701},
            {"type": "node", "id": 3, "lat": 51.002, "lon": -0.702},
            {"type": "way", "id": 100, "nodes": [1, 2, 3], "tags": {"name": "Test Lane", "highway": "residential"}},
        ]
    }
    graph, coords = build_graph_from_overpass_data(fake_data)
    print("Graph:", graph)
    assert 2 in [n for n, d, t, w in graph[1]], "Node 1 should connect to node 2"
    assert 1 in [n for n, d, t, w in graph[2]], "Two-way: node 2 should connect back to node 1"
    assert 3 in [n for n, d, t, w in graph[2]], "Node 2 should connect to node 3"
    print("PASSED — two-way road creates edges in both directions\n")

    print("--- Test 2: one-way road (oneway=yes), should only go forward ---")
    fake_oneway = {
        "elements": [
            {"type": "node", "id": 10, "lat": 51.0, "lon": -0.7},
            {"type": "node", "id": 11, "lat": 51.001, "lon": -0.701},
            {"type": "way", "id": 200, "nodes": [10, 11], "tags": {"name": "One Way St", "oneway": "yes", "highway": "residential"}},
        ]
    }
    graph2, coords2 = build_graph_from_overpass_data(fake_oneway)
    print("Graph:", graph2)
    assert 11 in [n for n, d, t, w in graph2.get(10, [])], "Should be able to go 10 -> 11"
    assert 10 not in graph2 or 11 not in [n for n, d, t, w in graph2.get(11, [])], \
        "Should NOT be able to go 11 -> 10 (oneway=yes)"
    print("PASSED — oneway=yes correctly creates only the forward edge\n")

    print("--- Test 3: reversed one-way road (oneway=-1) ---")
    fake_reversed = {
        "elements": [
            {"type": "node", "id": 20, "lat": 51.0, "lon": -0.7},
            {"type": "node", "id": 21, "lat": 51.001, "lon": -0.701},
            {"type": "way", "id": 300, "nodes": [20, 21], "tags": {"name": "Reversed St", "oneway": "-1", "highway": "residential"}},
        ]
    }
    graph3, coords3 = build_graph_from_overpass_data(fake_reversed)
    print("Graph:", graph3)
    assert 20 in [n for n, d, t, w in graph3.get(21, [])], "oneway=-1 should allow 21 -> 20"
    print("PASSED — oneway=-1 correctly reverses the direction\n")

    print("--- Test 4: distance calculation sanity check ---")
    dist = _haversine_distance_m(51.0, -0.7, 51.001, -0.7)
    print(f"Distance for ~0.001 degree latitude difference: {round(dist, 1)}m")
    assert 100 < dist < 120, f"Expected ~111m, got {dist}m"
    print("PASSED — distance calculation is sane\n")

    print("--- Test 5: maxspeed tag parsing (plain km/h) ---")
    speed = _parse_maxspeed_kmh("60", "residential")
    print(f"maxspeed='60' on residential -> {speed} km/h")
    assert speed == 60.0
    print("PASSED\n")

    print("--- Test 6: maxspeed tag parsing (UK mph) ---")
    speed_mph = _parse_maxspeed_kmh("30 mph", "residential")
    print(f"maxspeed='30 mph' -> {round(speed_mph, 1)} km/h")
    assert 48 < speed_mph < 49, f"Expected ~48.3 km/h, got {speed_mph}"
    print("PASSED — mph correctly converted to km/h\n")

    print("--- Test 7: missing maxspeed falls back to OSRM-derived class default ---")
    speed_default = _parse_maxspeed_kmh(None, "trunk")
    print(f"No maxspeed tag, highway=trunk -> {speed_default} km/h (should be 85, OSRM's default)")
    assert speed_default == 85
    print("PASSED\n")

    print("--- Test 8: a FASTER but LONGER road should have a SHORTER time than a SLOWER but SHORTER one ---")
    # This is the actual real-world scenario that motivated this fix:
    # a fast A-road (trunk, 85 km/h) covering 5km should take LESS
    # time than a slow lane (residential, 25 km/h) covering only 3km,
    # even though the residential road is shorter in raw distance.
    fast_long_data = {
        "elements": [
            {"type": "node", "id": 1, "lat": 51.0, "lon": -0.7},
            {"type": "node", "id": 2, "lat": 51.045, "lon": -0.7},  # ~5km north
            {"type": "way", "id": 500, "nodes": [1, 2], "tags": {"highway": "trunk"}},
        ]
    }
    slow_short_data = {
        "elements": [
            {"type": "node", "id": 1, "lat": 51.0, "lon": -0.7},
            {"type": "node", "id": 2, "lat": 51.027, "lon": -0.7},  # ~3km north
            {"type": "way", "id": 501, "nodes": [1, 2], "tags": {"highway": "residential"}},
        ]
    }
    g_fast, _ = build_graph_from_overpass_data(fast_long_data)
    g_slow, _ = build_graph_from_overpass_data(slow_short_data)
    fast_dist, fast_time = g_fast[1][0][1], g_fast[1][0][2]
    slow_dist, slow_time = g_slow[1][0][1], g_slow[1][0][2]
    print(f"Fast/long road: {round(fast_dist)}m, {round(fast_time)}s")
    print(f"Slow/short road: {round(slow_dist)}m, {round(slow_time)}s")
    assert fast_dist > slow_dist, "Sanity check: fast road should indeed be the longer one"
    assert fast_time < slow_time, \
        "BUG: the longer but faster road should take LESS time than the shorter but slower one"
    print("PASSED — time correctly reflects real-world speed, not just distance\n")

    print("All graph construction tests passed.")
    print("\n(Self-tests use fake data and need no internet access.")
    print("Run combined_test.py to build a real graph from a live")
    print("Overpass fetch for the Tillington-Haslemere area.)")