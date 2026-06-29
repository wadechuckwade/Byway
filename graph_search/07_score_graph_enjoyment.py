"""
Byway — Milestone 1.5, Step 4: Score every way in the graph for enjoyment
==============================================================================

What this does, in plain terms:
Scores every WAY in the graph (not just one pre-chosen route) across
four signals: driving enjoyment (curvature), scenery (land cover),
interest (landmarks), and — separately, at the route level, see
08_three_route_system.py — elevation.

MAJOR REFORMULATION THIS VERSION — separating AREA-WIDE signals from
POINT-BASED ones, rather than blending them together:

Previously, "scenery" was a blend of FOUR different things: land
cover composition, elevation, village (Conservation Area) proximity,
and ScenicOrNot proximity. Real-route testing kept finding the same
result: even when land cover genuinely scored well (a route that's
authentically 50% forest, 4% urban), the OVERALL score barely moved,
because land cover's contribution was always diluted by averaging it
against village/scenicness, which are near-zero for most of any
route's distance (they're sparse POINT features — a village or a
rated viewpoint exists at one specific spot, not continuously along
a road, unlike land cover composition, which has a real value
EVERYWHERE). Increasing land cover's weight within that blend (0.35
-> 0.50, a previous attempt) didn't meaningfully fix this — the
dilution was structural, not a weighting problem.

THE FIX: split scenery and "interest" into separate top-level
categories, matching what they actually measure:
  - scenery = landcover_score ALONE. The only true continuous,
    area-wide "what does the surrounding landscape look like" signal
    available — no dilution from anything sparse.
  - interest = a NOISY-OR blend of village (Conservation Area,
    LPA-weighted), ScenicOrNot, and historic (now Grade-weighted, see
    below) proximity — these are all the SAME kind of signal (sparse,
    point-based, "is there something notable right here"), just from
    different sources. Noisy-OR (1 - (1-a)(1-b)(1-c)) rewards being
    near MULTIPLE good things without requiring ALL of them
    simultaneously — the same AND-trap problem already found and
    fixed once for the overall driving/scenery/history split, now
    avoided again at this finer grain.

NEW — GRADED HISTORIC SITES (scoring/historic_england.py): historic
proximity used to mean "is there a NAMED OSM historic=*/tourism=
attraction tag nearby," with no notion of how significant any one
site actually is. Now combines that with Historic England's own
official Listed Building Grade (I > II* > II) and Scheduled Monuments
— a REAL, government-assigned magnitude, the same kind of upgrade
ScenicOrNot gave to "is this place pretty" for scenery.

NEW — LPA-WEIGHTED VILLAGES (scoring/villages.py): Conservation Areas
administered by a National Park Authority (genuinely scenic, legally-
designated LANDSCAPE, not just historic charm) now weight higher than
ones administered by an ordinary district/borough council — a real,
free signal that was already being fetched and simply never used for
ranking.

TWO-PHASE SCORING (kept from before): PHASE 1 (provisional) computes
driving (curvature) and interest (village/scenicornot/historic — none
of which need elevation OR land cover) for the WHOLE graph, cheaply.
land cover needs a per-point network call (no confirmed bulk-bbox
endpoint, unlike the others), so scenery is UNAVAILABLE in phase 1 —
phase 1's ranking is driving+interest only. PHASE 2 (refine_scores_
with_elevation) fetches land cover (CORRIDOR-sampled, see scoring/
landcover.py) and elevation ONLY for the small set of ways the final
candidates actually use, then computes the FULL formula including
scenery.

Elevation itself is NOT part of either per-way formula at all any
more — see refine_scores_with_elevation's docstring for the noise-
amplification reasoning; it's blended in separately, at the ROUTE
level, in 08_three_route_system.py.

GROUPING BY way_id, NOT BY INDIVIDUAL EDGE: a single OSM way is often
split into many small graph edges — grouping by way_id avoids
needlessly repeating curvature/proximity computation on what's really
one road's worth of shape.
"""

import os
import sys
import importlib.util

# Reuse Milestone 1's proven scoring modules directly, rather than
# reimplementing curvature/elevation/proximity logic a second time.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from scoring.curvature import score_curvature
from scoring.elevation import (
    fetch_elevations, score_elevation,
    subsample_points_by_distance, interpolate_elevations,
)
from scoring.proximity import route_bounding_box, fetch_historic_only, score_proximity
from scoring.villages import fetch_conservation_areas_in_bbox
from scoring.scenicornot import fetch_scenicornot_in_bbox
from scoring.historic_england import fetch_graded_historic_sites_in_bbox
from scoring.landcover import (
    fetch_landcover_classes, composition_from_classes, score_landcover, corridor_sample_points,
)


def to_ten(value):
    """
    Convert an internal 0-1 score to a 0-10, one-decimal-place number
    for display/printing — e.g. 0.84 -> 8.4. Internal math stays 0-1
    throughout (every weighted formula in this project assumes that
    range); this is a DISPLAY-ONLY conversion, applied at print time,
    not baked into the scoring itself.
    """
    return round(value * 10, 1)


def noisy_or(*scores):
    """
    Combines several independent 0-1 "is there something good here"
    signals into one, via 1 - product(1 - score) for each input.

    WHY THIS, NOT A PLAIN AVERAGE OR MAX: village/scenicornot/historic
    are conceptually the same KIND of signal (sparse, point-based
    landmarks) from different sources. A plain average would dilute a
    way that's near ONE excellent landmark just because the other two
    happen to be zero — the exact AND-trap problem already found and
    fixed once at the top level (driving/scenery/history all needing
    to be simultaneously high). A plain MAX would ignore the genuine
    extra value of being near multiple good things at once (a village
    AND a Grade I church should count for more than just the village
    alone). Noisy-OR sits between the two: rewards multiple
    simultaneous signals, without requiring all of them.

    Example: noisy_or(0.5, 0.5, 0.0) = 1 - 0.5*0.5*1.0 = 0.75 — higher
    than either alone (0.5), without needing the third to be nonzero.
    """
    product_of_complements = 1.0
    for s in scores:
        product_of_complements *= (1.0 - s)
    return round(1.0 - product_of_complements, 3)


ROAD_CLASS_ENJOYMENT_MULTIPLIER = {
    "motorway": 0.2, "motorway_link": 0.2,
    "trunk": 0.35, "trunk_link": 0.35,
    "primary": 0.55, "primary_link": 0.55,
    "secondary": 0.8, "secondary_link": 0.8,
    "tertiary": 0.95, "tertiary_link": 0.95,
    # Deliberately NOT listed (no penalty, multiplier 1.0): unclassified,
    # residential, living_street, service, and anything else.
}
DEFAULT_ROAD_CLASS_MULTIPLIER = 1.0

# CHANGED — driving/scenery/interest are now combined via noisy-OR
# (see noisy_or() above), not a weighted average — there are no
# top-level weight constants any more. Real research backs this
# directly: the official National Scenic Byway designation criteria
# require a road to exhibit AT LEAST ONE of six intrinsic qualities
# strongly at a regional level, not score well averaged across all of
# them — exactly the dynamic behind a persistent complaint that
# ordinary nice countryside was scoring 3/10 when a genuinely good
# scenery signal alone should already count for something real,
# rather than being diluted by mediocre driving/interest. Elevation is
# still NOT part of this combination at all (see module docstring) —
# it's blended in separately at the route level (08_three_route_
# system.py's ROUTE_ELEVATION_WEIGHT).


def group_edges_by_way(graph, node_coords):
    """
    Walk every edge in the graph and group them by way_id, collecting
    each way's full point sequence (in (lon, lat) form).

    Returns: {way_id: {"points": [(lon, lat), ...], "way_info": {...}}}
    """
    ways = {}

    for node_id, edges in graph.items():
        for neighbor_id, distance_m, time_s, way_info in edges:
            way_id = way_info["way_id"]
            if way_id not in ways:
                ways[way_id] = {"points": [], "way_info": way_info, "_seen_nodes": set()}

            entry = ways[way_id]
            for nid in (node_id, neighbor_id):
                if nid not in entry["_seen_nodes"]:
                    lat, lon = node_coords[nid]
                    entry["points"].append((lon, lat))
                    entry["_seen_nodes"].add(nid)

    for way_id in ways:
        del ways[way_id]["_seen_nodes"]

    return ways


def score_all_ways_provisional(graph, node_coords):
    """
    PHASE 1 — scores every way in the graph using curvature and
    INTEREST (village/scenicornot/historic — none of which need
    elevation or land cover) for the WHOLE graph, cheaply. Scenery
    (land cover) is UNAVAILABLE here — it needs a per-point network
    call with no confirmed bulk-bbox endpoint, unlike everything else
    fetched in this function — see refine_scores_with_elevation for
    where it's actually computed.

    PROVISIONAL formula (noisy-OR of driving and interest only,
    since scenery is entirely missing at this stage):
      enjoyment_provisional = noisy_or(driving, interest)
    This is fine for RANKING purposes (picking top-N ways, see
    04_pathfinding.py's generate_candidate_routes) — phase 2 recomputes
    the full formula with real magnitudes once land cover is known.
    Separately, scoring.landcover.generate_landcover_grid_targets()
    gives candidate generation a DEDICATED way to find land-cover-rich
    AREAS directly, complementing this way-ranking rather than
    needing phase 1 to estimate land cover at all.

    Returns (provisional_scores, ways, region_features):
      region_features: {"water", "forest", "historic",
        "conservation_areas", "scenicornot"} — the raw fetched lists,
        for reuse by candidate generation (see 04_pathfinding.py's
        generate_candidate_routes, source 4). water/forest are still
        fetched (bundled with historic in one Overpass query) but are
        not used in any scoring formula — kept stored for optionality.
    """
    ways = group_edges_by_way(graph, node_coords)
    print(f"  Grouped {sum(len(e) for e in graph.values())} directed edges into "
          f"{len(ways)} unique ways for scoring.")

    all_points = [p for way in ways.values() for p in way["points"]]

    bbox = route_bounding_box(all_points)
    try:
        osm_historic_sites = fetch_historic_only(bbox)
    except Exception as e:
        print(f"  Could not fetch historic/attraction data: {e}")
        osm_historic_sites = []

    try:
        conservation_areas = fetch_conservation_areas_in_bbox(bbox)
        print(f"  Found {len(conservation_areas)} Conservation Areas (villages/towns) in the route area.")
    except Exception as e:
        print(f"  Could not fetch Conservation Areas data: {e}")
        conservation_areas = []

    try:
        scenicornot_spots = fetch_scenicornot_in_bbox(bbox)
        print(f"  Found {len(scenicornot_spots)} ScenicOrNot rated spot(s) in the route area.")
    except Exception as e:
        print(f"  Could not fetch ScenicOrNot data: {e}")
        scenicornot_spots = []

    try:
        graded_historic_sites = fetch_graded_historic_sites_in_bbox(bbox)
        print(f"  Found {len(graded_historic_sites)} graded historic site(s) "
              f"(Listed Buildings/Scheduled Monuments) in the route area.")
    except Exception as e:
        print(f"  Could not fetch graded historic sites: {e}")
        graded_historic_sites = []

    # Combine OSM-tagged historic/attraction sites (default weight 1.0,
    # via score_proximity's fallback) with Historic England's
    # OFFICIALLY GRADED sites (real Grade-based weights) into one
    # list — score_proximity already handles per-feature weights
    # correctly regardless of source, so both blend naturally.
    combined_historic_features = osm_historic_sites + graded_historic_sites

    provisional_scores = {}

    for way_id, way_data in ways.items():
        points = way_data["points"]
        road_class = way_data["way_info"].get("highway", "")
        class_multiplier = ROAD_CLASS_ENJOYMENT_MULTIPLIER.get(road_class, DEFAULT_ROAD_CLASS_MULTIPLIER)

        if len(points) < 3:
            provisional_scores[way_id] = {
                "driving_enjoyment": 0.0, "scenery": 0.0, "interest": 0.0,
                "enjoyment_score": 0.0, "road_class": road_class,
                "road_class_multiplier": class_multiplier,
                "_curvature_score": 0.0, "_village_score": 0.0,
                "_scenicornot_score": 0.0, "_historic_score": 0.0,
                "_total_length_m": 0.0,
            }
            continue

        curvature_result = score_curvature(points)
        village_result = score_proximity(points, conservation_areas)
        scenicornot_result = score_proximity(points, scenicornot_spots,
                                              close_threshold_m=300, score_decay_m=1200)
        historic_result = score_proximity(points, combined_historic_features)
        # NOTE: scenicornot uses wider thresholds than the other
        # proximity categories (300m/1200m vs the usual 50m/500m) —
        # each rating represents a ~1km GRID SQUARE's character, not a
        # point feature like a named historic site, so "close" should
        # mean "within the same square's neighbourhood," not "right
        # on top of the exact photo's spot."

        driving_enjoyment = curvature_result["curvature_score"] * class_multiplier
        interest = noisy_or(
            village_result["proximity_score"],
            scenicornot_result["proximity_score"],
            historic_result["proximity_score"],
        ) * class_multiplier

        # CHANGED — top-level combination is now noisy-OR
        # (driving, interest), not a weighted average. Real research
        # backs this directly: the official National Scenic Byway
        # designation criteria require a road to exhibit AT LEAST ONE
        # of six intrinsic qualities strongly at a regional level —
        # not score well averaged across all of them. A weighted
        # average let mediocre driving/interest drag down a route
        # whose scenery was genuinely good — exactly the dynamic
        # behind a persistent complaint that ordinary nice countryside
        # was scoring 3/10 when it should read more like 6+. Scenery
        # is excluded here (provisional, phase 1 — unknown until
        # land cover is fetched in phase 2; see refine_scores_with_
        # elevation for the full three-way version).
        enjoyment_score = noisy_or(driving_enjoyment, interest)

        provisional_scores[way_id] = {
            "driving_enjoyment": round(driving_enjoyment, 3),
            "scenery": 0.0,  # genuinely unknown until phase 2 — not a real zero score
            "interest": round(interest, 3),
            "enjoyment_score": round(enjoyment_score, 3),
            "road_class": road_class,
            "road_class_multiplier": class_multiplier,
            "_curvature_score": curvature_result["curvature_score"],
            "_village_score": village_result["proximity_score"],
            "_scenicornot_score": scenicornot_result["proximity_score"],
            "_historic_score": historic_result["proximity_score"],
            "_total_length_m": curvature_result["total_length_m"],
        }

    region_features = {
        "historic": combined_historic_features,
        "conservation_areas": conservation_areas,
        "scenicornot": scenicornot_spots,
    }

    return provisional_scores, ways, region_features


def refine_scores_with_elevation(way_ids, ways, provisional_scores,
                                  target_spacing_m=100, landcover_target_spacing_m=400,
                                  verbose=True):
    """
    PHASE 2 — given a SMALL subset of way_ids (the ones actually used
    by the final candidate route set), fetches elevation AND land
    cover ONLY for those ways' points, then computes the FULL formula
    with scenery (land cover) now available:
        enjoyment_score = noisy_or(driving, scenery, interest)
    driving and interest are UNCHANGED from phase 1 (curvature alone;
    noisy-OR of village/scenicornot/historic) — neither needs
    elevation or land cover, so there's nothing to "refine" about
    them, only scenery is genuinely new here.

    ELEVATION IS NOT PART OF THIS FORMULA AT ALL (direct feedback,
    with a real mechanism behind it): SRTM's vertical accuracy is
    roughly ±10m. Over a 150m-long graph-search way, a single
    measurement error produces a spurious ~6.7% gradient — comparable
    to our whole 8% "dramatic" threshold. Over a real multi-km route,
    the same error becomes a rounding error. There's also a
    perceptual mismatch: "this road climbs dramatically" is something
    experienced over a sustained stretch, not a property any single
    200m way has alone. This function still fetches/returns "total_
    ascent_m" and "elevation_score" per way, RAW, unblended —
    08_three_route_system.py's route_level_elevation_score() aggregates
    these across a whole candidate route's ways, proportional to each
    edge's real distance, and blends ONE route-level elevation score
    into the final avg_enjoyment/best_stretch separately.

    Land cover uses CORRIDOR sampling (scoring.landcover.
    corridor_sample_points: perpendicular offsets at ±200m, not just
    points along the centerline) — approximating the validated
    research's actual methodology (land-cover composition within a
    surrounding PATCH/area, not "what's literally under the road").
    landcover_target_spacing_m stays wide (400m along the road) to
    keep total points bounded; the corridor offset count was reduced
    from 4 to 2 after real-route testing showed the original 4-offset
    version taking 1,647 seconds (a ~5x point-count multiplier at
    CORINE's live per-point latency) — this is a stopgap; the durable
    fix is the OS OpenMap Local migration (scoring/os_landcover.py),
    which removes the live-network cost entirely regardless of
    sampling density.

    Returns a dict {way_id: {...}} with corrected, complete scores for
    just the given way_ids — merge this into provisional_scores before
    recomputing avg_enjoyment for each candidate route.
    """
    way_sample_plan = {}
    way_landcover_sample_plan = {}
    all_sample_points = []
    all_landcover_points = []
    for way_id in way_ids:
        if way_id not in ways or way_id not in provisional_scores:
            continue
        points = ways[way_id]["points"]
        if len(points) < 3:
            continue
        sampled_pairs = subsample_points_by_distance(points, target_spacing_m=target_spacing_m)
        way_sample_plan[way_id] = (points, sampled_pairs)
        all_sample_points.extend(p for _, p in sampled_pairs)

        # Sparse centerline points first, THEN expand each into a
        # corridor (perpendicular offsets) — see docstring above.
        landcover_centerline_pairs = subsample_points_by_distance(points, target_spacing_m=landcover_target_spacing_m)
        landcover_centerline_points = [p for _, p in landcover_centerline_pairs]
        landcover_corridor_points = corridor_sample_points(landcover_centerline_points)
        way_landcover_sample_plan[way_id] = landcover_corridor_points
        all_landcover_points.extend(landcover_corridor_points)

    unique_sample_points = list(dict.fromkeys(all_sample_points))
    unique_landcover_points = list(dict.fromkeys(all_landcover_points))

    if verbose:
        naive_total = sum(len(ways[wid]["points"]) for wid in way_sample_plan)
        print(f"  Refining elevation for {len(way_sample_plan)} candidate-route ways: "
              f"{len(unique_sample_points)} sampled points needed "
              f"(vs {naive_total} if every point were queried individually, "
              f"vs the WHOLE graph's {len(provisional_scores)} ways if this weren't bounded at all).")
        print(f"  Refining land cover for the same {len(way_sample_plan)} ways using CORRIDOR sampling "
              f"({landcover_target_spacing_m}m along the road, +/-200m perpendicular offsets): "
              f"{len(unique_landcover_points)} points needed.")

    sampled_elevations = fetch_elevations(unique_sample_points)
    elevation_by_point = dict(zip(unique_sample_points, sampled_elevations))

    landcover_by_point = fetch_landcover_classes(unique_landcover_points, verbose=verbose)

    refined = {}
    for way_id, (points, sampled_pairs) in way_sample_plan.items():
        sampled_with_elev = [(idx, elevation_by_point.get(p)) for idx, p in sampled_pairs]
        full_elevations = interpolate_elevations(points, sampled_with_elev)
        prov = provisional_scores[way_id]
        elevation_result = score_elevation(full_elevations, total_distance_m=prov.get("_total_length_m"))

        landcover_points = way_landcover_sample_plan.get(way_id, [])
        way_point_categories = {p: landcover_by_point.get(p) for p in landcover_points}
        composition = composition_from_classes(way_point_categories)
        landcover_result = score_landcover(composition)

        class_multiplier = prov["road_class_multiplier"]

        # driving and interest are UNCHANGED from phase 1 — neither
        # needs elevation or land cover. Only scenery is genuinely new.
        driving_enjoyment = prov["_curvature_score"] * class_multiplier
        interest = noisy_or(
            prov["_village_score"], prov["_scenicornot_score"], prov["_historic_score"],
        ) * class_multiplier
        scenery = landcover_result["landcover_score"] * class_multiplier

        # CHANGED — noisy-OR across all three, not a weighted average
        # — see score_all_ways_provisional's docstring for the full
        # reasoning (grounded in the official National Scenic Byway
        # designation criteria: "at least one quality, strongly," not
        # an average across every quality at once).
        enjoyment_score = noisy_or(driving_enjoyment, scenery, interest)

        refined[way_id] = {
            "driving_enjoyment": round(driving_enjoyment, 3),
            "scenery": round(scenery, 3),
            "interest": round(interest, 3),
            "enjoyment_score": round(enjoyment_score, 3),
            "road_class": prov["road_class"],
            "road_class_multiplier": class_multiplier,
            "elevation_score": elevation_result["elevation_score"],
            "total_ascent_m": elevation_result["total_ascent_m"],
            "_total_length_m": prov.get("_total_length_m", 0.0),
            "landcover_score": landcover_result["landcover_score"],
            "landcover_composition": composition,
        }

    return refined


def score_all_ways_for_enjoyment(graph, node_coords):
    """
    CONVENIENCE WRAPPER doing BOTH phases for the WHOLE graph at once.
    NOT RECOMMENDED for graphs of more than a few hundred ways. Prefer
    score_all_ways_provisional() + refine_scores_with_elevation() for
    anything graph-sized (see 08_three_route_system.py).
    """
    provisional_scores, ways, region_features = score_all_ways_provisional(graph, node_coords)
    all_way_ids = list(provisional_scores.keys())
    refined = refine_scores_with_elevation(all_way_ids, ways, provisional_scores, verbose=False)
    final_scores = dict(provisional_scores)
    final_scores.update(refined)
    return final_scores


if __name__ == "__main__":
    # Self-tests using small fake graph data (no internet needed) for
    # everything except the live-network sections, which are mocked.
    import math as _math

    print("--- Test 1: grouping edges by way_id reconstructs correct point sequences ---")
    fake_graph = {
        1: [(2, 100, 10, {"way_id": 500, "name": "Test Road", "highway": "residential"})],
        2: [
            (1, 100, 10, {"way_id": 500, "name": "Test Road", "highway": "residential"}),
            (3, 100, 10, {"way_id": 500, "name": "Test Road", "highway": "residential"}),
        ],
        3: [(2, 100, 10, {"way_id": 500, "name": "Test Road", "highway": "residential"})],
    }
    fake_coords = {1: (51.0, -0.7), 2: (51.001, -0.701), 3: (51.002, -0.702)}
    ways = group_edges_by_way(fake_graph, fake_coords)
    assert 500 in ways
    assert len(ways[500]["points"]) == 3, f"Expected 3 unique points, got {len(ways[500]['points'])}"
    print(f"Points: {ways[500]['points']}")
    print("PASSED — way correctly reconstructed with 3 unique points, no duplicates from the two-way edges\n")

    print("--- Test 2: noisy-OR rewards multiple simultaneous landmarks without requiring all of them ---")
    one_strong = noisy_or(0.9, 0.0, 0.0)
    two_moderate = noisy_or(0.5, 0.5, 0.0)
    all_three = noisy_or(0.5, 0.5, 0.5)
    none = noisy_or(0.0, 0.0, 0.0)
    print(f"One strong signal (0.9,0,0): {one_strong}")
    print(f"Two moderate signals (0.5,0.5,0): {two_moderate}")
    print(f"All three moderate (0.5,0.5,0.5): {all_three}")
    print(f"None (0,0,0): {none}")
    assert none == 0.0
    assert two_moderate > 0.5, "Two simultaneous moderate signals should beat either alone"
    assert all_three > two_moderate, "Three simultaneous signals should beat two"
    assert one_strong < 1.0 and one_strong > 0.85, "A single strong signal should dominate, close to its own value"
    print("PASSED — noisy-OR rewards multiple landmarks without requiring all of them, avoiding the AND-trap\n")

    print("--- Test 3: road-class penalty actually dampens scores for major roads ---")
    identical_points = []
    for i in range(12):
        angle = i * 0.35
        lon = -0.70 + i * 0.0004
        lat = 51.00 + _math.sin(angle) * 0.0015
        identical_points.append((lon, lat))
    minor_graph = {i: [(i + 1, 50, 5, {"way_id": 600, "name": "Quiet Lane", "highway": "residential"})]
                   for i in range(1, len(identical_points))}
    trunk_graph = {i: [(i + 1, 50, 5, {"way_id": 700, "name": "Big Road", "highway": "trunk"})]
                   for i in range(1, len(identical_points))}
    fake_coords_2 = {i + 1: (lat, lon) for i, (lon, lat) in enumerate(identical_points)}
    minor_ways = group_edges_by_way(minor_graph, fake_coords_2)
    trunk_ways = group_edges_by_way(trunk_graph, fake_coords_2)
    minor_curv = score_curvature(minor_ways[600]["points"])
    trunk_curv = score_curvature(trunk_ways[700]["points"])
    assert minor_curv == trunk_curv
    assert minor_curv["curvature_score"] > 0.0
    minor_mult = ROAD_CLASS_ENJOYMENT_MULTIPLIER.get("residential", DEFAULT_ROAD_CLASS_MULTIPLIER)
    trunk_mult = ROAD_CLASS_ENJOYMENT_MULTIPLIER.get("trunk", DEFAULT_ROAD_CLASS_MULTIPLIER)
    assert trunk_mult < minor_mult
    raw = minor_curv["curvature_score"]
    assert raw * trunk_mult < raw * minor_mult
    print(f"Identical raw curvature ({to_ten(raw)}/10) -> residential: {to_ten(raw*minor_mult)}/10, "
          f"trunk: {to_ten(raw*trunk_mult)}/10")
    print("PASSED — identical road shapes score differently once road class is accounted for\n")

    print("--- Test 4 (the critical one): phase 1 + phase 2 EXACTLY reconstruct the same single-phase formula ---")
    test_points = []
    for i in range(8):
        angle = i * 0.4
        lon = -0.70 + i * 0.0005
        lat = 51.00 + _math.sin(angle) * 0.0012
        test_points.append((lon, lat))

    test_graph = {}
    for i in range(len(test_points) - 1):
        node_id = i + 1
        test_graph[node_id] = [(node_id + 1, 60, 6, {"way_id": 999, "name": "Test Way", "highway": "secondary"})]
    test_graph[len(test_points)] = []
    test_node_coords = {i + 1: (lat, lon) for i, (lon, lat) in enumerate(test_points)}

    fake_historic_osm = [{"name": "Test Castle", "lat": test_points[5][1], "lon": test_points[5][0]}]
    fake_graded_historic = [{"name": "Test Church", "lat": test_points[5][1], "lon": test_points[5][0], "weight": 0.7}]
    fake_conservation = [{"name": "Test Village", "lat": test_points[1][1], "lon": test_points[1][0], "weight": 1.0}]
    fake_scenicornot = [{"name": "Scenic spot", "lat": test_points[4][1], "lon": test_points[4][0], "weight": 0.7}]
    fake_elevations_by_point = {p: 50.0 + i * 5 for i, p in enumerate(test_points)}  # a steady climb

    def _fake_landcover_classifier(points, verbose=True):
        # Deterministic classifier that works for ANY point, including
        # corridor offset points (not in test_points, so a fixed
        # lookup dict keyed only on test_points wouldn't cover them).
        return {p: ("forest_natural" if int(round(p[0] * 100000)) % 2 == 0 else "agriculture") for p in points}

    # NOTE: this file uses "from module import name", which binds a
    # LOCAL copy in this module's own namespace -- reassigning the
    # names directly, here, in this module's own global namespace is
    # what actually takes effect for calls made from inside this file.
    fetch_historic_only = lambda bbox: fake_historic_osm
    fetch_conservation_areas_in_bbox = lambda bbox: fake_conservation
    fetch_scenicornot_in_bbox = lambda bbox: fake_scenicornot
    fetch_graded_historic_sites_in_bbox = lambda bbox: fake_graded_historic
    fetch_elevations = lambda points, **kwargs: [fake_elevations_by_point.get(p) for p in points]
    fetch_landcover_classes = _fake_landcover_classifier

    combined_historic = fake_historic_osm + fake_graded_historic
    manual_curv = score_curvature(test_points)
    manual_village = score_proximity(test_points, fake_conservation)
    manual_scenicornot = score_proximity(test_points, fake_scenicornot, close_threshold_m=300, score_decay_m=1200)
    manual_historic = score_proximity(test_points, combined_historic)
    manual_elev = score_elevation([fake_elevations_by_point[p] for p in test_points],
                                   total_distance_m=manual_curv["total_length_m"])
    manual_landcover_centerline_pairs = subsample_points_by_distance(test_points, target_spacing_m=400)
    manual_landcover_centerline_points = [p for _, p in manual_landcover_centerline_pairs]
    manual_landcover_corridor_points = corridor_sample_points(manual_landcover_centerline_points)
    manual_landcover_classes = _fake_landcover_classifier(manual_landcover_corridor_points)
    manual_composition = composition_from_classes(manual_landcover_classes)
    manual_landcover = score_landcover(manual_composition)
    manual_class_mult = ROAD_CLASS_ENJOYMENT_MULTIPLIER.get("secondary", DEFAULT_ROAD_CLASS_MULTIPLIER)

    manual_driving = manual_curv["curvature_score"] * manual_class_mult
    manual_scenery = manual_landcover["landcover_score"] * manual_class_mult
    manual_interest = noisy_or(
        manual_village["proximity_score"], manual_scenicornot["proximity_score"], manual_historic["proximity_score"],
    ) * manual_class_mult
    manual_enjoyment = noisy_or(manual_driving, manual_scenery, manual_interest)

    two_phase_scores = score_all_ways_for_enjoyment(test_graph, test_node_coords)
    two_phase_enjoyment = two_phase_scores[999]["enjoyment_score"]

    print(f"Manual single-phase formula:  enjoyment_score = {to_ten(manual_enjoyment)}/10")
    print(f"Two-phase (phase1 + phase2):  enjoyment_score = {to_ten(two_phase_enjoyment)}/10")
    assert abs(two_phase_enjoyment - manual_enjoyment) < 0.005, (
        f"Two-phase scoring should reconstruct the EXACT same formula as the single-phase "
        f"version, got {two_phase_enjoyment} vs manual {manual_enjoyment}"
    )
    print("PASSED — two-phase scoring exactly reconstructs the new driving/scenery/interest formula\n")

    print("--- Test 5 (NEW): score_all_ways_provisional returns region_features for targeted candidate generation ---")
    provisional_only, ways_only, region_features = score_all_ways_provisional(test_graph, test_node_coords)
    print(f"region_features keys: {list(region_features.keys())}")
    assert set(region_features.keys()) == {"historic", "conservation_areas", "scenicornot"}
    assert region_features["scenicornot"] == fake_scenicornot
    print("PASSED — raw feature lists correctly returned for downstream targeted candidate generation\n")

    print("--- Test 6: scenery (land cover) genuinely moves the score, with NO dilution from village/scenicornot ---")
    # ALL-URBAN land cover, but keep village/scenicornot/historic
    # exactly as before -- confirms scenery is now fully independent
    # of interest (no cross-contamination either direction).
    fetch_landcover_classes = lambda points, verbose=True: {p: "urban" for p in points}
    scores_urban = score_all_ways_for_enjoyment(test_graph, test_node_coords)
    enjoyment_urban = scores_urban[999]["enjoyment_score"]
    interest_urban = scores_urban[999]["interest"]
    interest_before = two_phase_scores[999]["interest"]
    print(f"Original (mixed forest/agri landcover): enjoyment={to_ten(two_phase_enjoyment)}/10, "
          f"interest={to_ten(interest_before)}/10")
    print(f"All-urban landcover:                    enjoyment={to_ten(enjoyment_urban)}/10, "
          f"interest={to_ten(interest_urban)}/10")
    assert enjoyment_urban < two_phase_enjoyment, "All-urban land cover should measurably LOWER the score"
    assert abs(interest_urban - interest_before) < 0.001, (
        "Interest must be COMPLETELY UNCHANGED by a land-cover change -- "
        "scenery and interest must not cross-contaminate"
    )
    print("PASSED — scenery moves independently of interest, confirming no dilution either direction\n")

    print("All scoring tests passed.")
