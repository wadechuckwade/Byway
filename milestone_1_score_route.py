"""
Byway — Milestone 1: Combined Route Scoring (v3 — village signal wired in,
curvature boundary-loss fixed)
=========================================================================

What this does, in plain terms:
Takes a real route (from OSRM), breaks it into segments by road name,
and scores each segment using:
  - Curvature (pure geometry, no API needed)
  - Elevation (Open Topo Data, fetched ONCE for the whole route)
  - Water / Forest / Historic proximity (OSM/Overpass, fetched ONCE
    for the whole route's bounding box)
  - Village character (official Conservation Areas, fetched ONCE for
    the whole route's bounding box) — NEW in this version, see below

These combine into spec categories:
  - Driving Enjoyment  = curvature + a little elevation
  - Scenery            = water + forest + elevation + village character
  - History & Culture  = historic site proximity

WHAT CHANGED IN THIS VERSION, AND WHY (real-route testing on
Tillington -> Haslemere via Upperton/Lurgashall):

1. VILLAGE/CONSERVATION-AREA SIGNAL NOW ACTUALLY WIRED IN. Conservation
   Area data was already being fetched in this file (fetch_conservation_
   areas_in_bbox), but only ever used for a separate, informational
   "notable villages" print line — it never fed into the Scenery score
   itself. graph_search/07_score_graph_enjoyment.py had already been
   updated to include this as a real scenery input (after the
   "Scoring Fixes Round 2" work), but that change was never ported
   back to this file, so anyone running THIS script was silently
   getting the older, narrower 3-way scenery formula. Fixed: scenery
   is now 0.3 water + 0.3 forest + 0.2 elevation + 0.2 village,
   matching 07's formula, and every Conservation Area found in the
   route's bounding box is now printed by name (not just a count) —
   this answers directly whether a specific known Conservation Area
   was found at all, separate from whether it ended up close enough
   to the route to count.

2. CURVATURE BOUNDARY-LOSS FIX. Unnamed minor lanes get split into
   many short segments (no name tag to merge consecutive OSRM steps
   on) — the Upperton/Lurgashall route had 23 separate "Unnamed road"
   segments. score_curvature() previously only ever saw one segment's
   own points, so a real bend whose 3-point window straddled a
   segment boundary was invisible to BOTH segments — see
   scoring/curvature.py's module docstring for the full explanation
   and a passing regression test demonstrating real recovered
   curvature. Fixed here by passing each segment a little point
   context from its immediate neighbours when scoring curvature,
   using the new core_start/core_end parameters so length/score
   attribution still belongs to the right segment.

WHY THE ROUTE-WIDE CACHING ARCHITECTURE (unchanged from before):
The original version fetched elevation and Overpass data separately
for EVERY segment, meaning a route with 20 segments made ~60+ live
API calls. This was slow (minutes per route) and unreliable (the free
Overpass server regularly timed out under that load, silently losing
data on exactly the segments that mattered).

This version fetches elevation and Overpass data ONCE for the entire
route, then every segment scores itself against that already-fetched
data with zero further network calls.

Requires real internet access — run this in GitHub Codespaces, not in
Claude's sandboxed tool environment.

Requires: pip install requests
"""

import math
import time
import requests

from scoring.curvature import score_curvature
from scoring.elevation import fetch_elevations, score_elevation
from scoring.proximity import route_bounding_box, fetch_region_data, score_proximity
from scoring.reputation import get_road_reputation, GENERIC_NAME_SKIP_LIST
from scoring.villages import fetch_conservation_areas_in_bbox


def get_route_with_geometry(start, end, waypoints=None):
    """Ask OSRM for a route and return both turn-by-turn steps AND geometry."""
    coord_list = [start] + (waypoints or []) + [end]
    coords = ";".join(f"{p['lon']},{p['lat']}" for p in coord_list)
    url = (
        f"https://router.project-osrm.org/route/v1/driving/{coords}"
        f"?overview=full&geometries=geojson&steps=true"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def split_into_segments(route_data, max_unnamed_segment_m=500):
    """
    Break a route into segments by road name. Returns a list of dicts:
    {name, points} where points is a list of (lon, lat) tuples.

    max_unnamed_segment_m tightened from 1500m to 500m: testing
    revealed that even after fixing the original bug (where unnamed
    steps could merge into one giant multi-kilometre blob), a 1.5km
    cap still produced segments coarse enough to hide real variation
    — a single "Unnamed road" segment spanning much of a route's
    length averages away exactly the kind of local detail (a lovely
    stretch next to a dull one) that per-segment scoring exists to
    surface. 500m gives much finer granularity for unnamed rural
    roads, at the cost of more segments overall — an acceptable
    trade-off given how fast the route-wide-caching architecture
    already is.

    NOTE: this finer granularity is exactly why the curvature
    boundary-loss bug (see module docstring) mattered enough to fix —
    more, shorter segments means more boundaries for a real curve to
    fall across.

    IMPORTANT FIX: an earlier version of this function merged
    consecutive steps whenever their names matched — including
    "Unnamed road", a placeholder WE assign to any step OSRM didn't
    give a real name. Since many genuinely different rural roads lack
    name tags in OSM, this caused dozens of distinct roads to all
    collapse into one giant fake "segment" spanning many kilometres,
    hiding exactly the segment-level detail (e.g. "this 800m near the
    museum is great") the whole design is meant to surface.

    Fix: unnamed steps are never merged with each other based on name
    alone. Instead, each unnamed step starts a new segment, UNLESS
    appending it would keep the segment under max_unnamed_segment_m —
    this still avoids creating hundreds of tiny single-step segments
    for routine short unnamed stretches, while preventing the
    multi-kilometre false-merge bug.

    SECOND FIX (the merge-cap alone wasn't enough): testing revealed
    that some OSRM steps are themselves already several kilometres
    long as a SINGLE step — e.g. one long straight unnamed rural road
    with no turns. The merge-cap logic above only controls whether to
    COMBINE separate steps; it can never split a step that was already
    too long on its own, since there's nothing to "merge" — it's one
    atomic unit from OSRM's perspective. This is why tightening the
    cap alone had no visible effect on a real long unnamed stretch.
    Fix: any unnamed step is now subdivided into roughly
    max_unnamed_segment_m-sized chunks BEFORE the merge logic runs,
    so oversized individual steps get broken up too, not just
    sequences of small steps that were merging together.
    """
    route = route_data["routes"][0]
    segments = []

    for leg in route["legs"]:
        for step in leg["steps"]:
            raw_name = step.get("name", "").strip()
            is_unnamed = not raw_name
            name = raw_name or "Unnamed road"
            step_geometry = step.get("geometry", {}).get("coordinates", [])
            points = [(c[0], c[1]) for c in step_geometry] if step_geometry else []

            if not points:
                continue

            # If this single step is itself already long, split it
            # into smaller chunks before doing anything else. Named
            # roads are left whole (a long named road is genuinely
            # one real road, e.g. "Witney Road" for 700m is fine to
            # keep as one segment) — only unnamed steps get this
            # treatment, since those are the ones masking real
            # variation along generic, untagged rural stretches.
            sub_chunks = [points]
            was_subdivided = False
            if is_unnamed:
                step_length_m = _approx_length_m(points)
                if step_length_m > max_unnamed_segment_m:
                    num_chunks = max(1, math.ceil(step_length_m / max_unnamed_segment_m))
                    chunk_size = max(2, len(points) // num_chunks)
                    # FIX: chunks must overlap by one point at each
                    # boundary, or the distance BETWEEN chunks (e.g.
                    # from the last point of chunk N to the first
                    # point of chunk N+1) is never measured by either
                    # chunk's own length calculation, silently
                    # under-counting total route length at every
                    # chunk boundary (~14% lost across ~14 boundaries
                    # in testing). Step by (chunk_size - 1) so
                    # consecutive chunks share their boundary point.
                    step_by = max(1, chunk_size - 1)
                    sub_chunks = [
                        points[i:i + chunk_size]
                        for i in range(0, len(points) - 1, step_by)
                    ]
                    if len(sub_chunks) > 1 and len(sub_chunks[-1]) < 2:
                        sub_chunks[-2] = sub_chunks[-2] + sub_chunks[-1]
                        sub_chunks.pop()
                    was_subdivided = True

            for chunk_points in sub_chunks:
                if len(chunk_points) < 2:
                    continue

                should_merge = False
                # FIX: chunks we just created by deliberately splitting
                # an oversized step must NEVER be merged back together
                # — otherwise the merge logic below (designed for
                # naturally-short separate OSRM steps) silently undoes
                # the subdivision we just did, re-combining ~2 chunks
                # at a time and leaving segments far longer than
                # max_unnamed_segment_m. Only allow merging when this
                # step was NOT itself subdivided.
                if not was_subdivided and segments and segments[-1]["name"] == name:
                    if not is_unnamed:
                        # Same real road name in a row — genuinely the
                        # same road, safe to merge.
                        should_merge = True
                    else:
                        # Both unnamed — only merge if doing so keeps
                        # the combined segment reasonably short.
                        current_length_m = _approx_length_m(segments[-1]["points"])
                        if current_length_m < max_unnamed_segment_m:
                            should_merge = True

                if should_merge:
                    segments[-1]["points"].extend(chunk_points)
                else:
                    segments.append({"name": name, "points": chunk_points})

    return segments


def _approx_length_m(points):
    """Quick approximate length of a point list, used only for the
    unnamed-segment size cap above (doesn't need to be exact)."""
    import math
    total = 0.0
    for i in range(len(points) - 1):
        lon1, lat1 = points[i]
        lon2, lat2 = points[i + 1]
        # Cheap flat approximation is fine for this size-cap check.
        dx = (lon2 - lon1) * 111320 * math.cos(math.radians(lat1))
        dy = (lat2 - lat1) * 111320
        total += math.hypot(dx, dy)
    return total


def assign_features_to_nearest_segment(segments, features):
    """
    Assign each feature (e.g. a historic site) to the SINGLE segment
    it's closest to, rather than letting every nearby segment claim
    it independently.

    WHY THIS EXISTS: testing on the Hathersage-Castleton route showed
    the same 4-5 historic sites (e.g. "Devil's Arse (Peak Cavern)",
    "Castleton War Memorial") appearing identically in MULTIPLE
    consecutive segments' nearby-sites lists. This wasn't a bug in
    the distance maths — Castleton genuinely does have several sites
    close together — but it created a real structural bias: a route
    passing through one dense cluster gets to claim "5 nearby
    features" repeatedly, once per segment within range, while a
    route with the same total number of interesting things spread
    more evenly along its length never gets that multiplied credit
    on any single segment. This systematically favoured routes that
    end at (or pass through) a single dense historic cluster over
    routes with more evenly distributed, equally real points of
    interest.

    Fix: each feature is credited to exactly ONE segment — whichever
    is geometrically closest — so a cluster of 5 sites contributes
    "5 sites" to the route's total ONCE, not once per nearby segment.

    Returns a dict: {segment_index: [features assigned to that segment]}
    """
    assignment = {i: [] for i in range(len(segments))}

    for feature in features:
        best_segment_idx = None
        best_distance = float("inf")
        for i, seg in enumerate(segments):
            # Reuse score_proximity's own distance calculation (its
            # "closest_m" field) rather than reaching into proximity.py's
            # private internals, keeping this a clean use of the
            # module's public interface.
            check = score_proximity(seg["points"], [feature], close_threshold_m=0, score_decay_m=1)
            dist = check["closest_m"]
            if dist is not None and dist < best_distance:
                best_distance = dist
                best_segment_idx = i
        if best_segment_idx is not None:
            assignment[best_segment_idx].append(feature)

    return assignment


def _curvature_context_points(all_segments, segment_index, context_points=2):
    """
    NEW: build a padded point list for curvature scoring, borrowing a
    few points from this segment's immediate neighbours, plus the
    core_start/core_end range identifying which part of the padded
    list is THIS segment's own.

    WHY THIS EXISTS: see module docstring and scoring/curvature.py's
    "BOUNDARY-CURVE FIX" — without this, a real bend whose 3-point
    window straddles a segment boundary is invisible to both
    neighbouring segments, which matters a lot once a route is broken
    into many short unnamed-lane segments.

    Returns (padded_points, core_start, core_end).
    """
    own_points = all_segments[segment_index]["points"]

    pad_before = []
    if segment_index > 0:
        prev_points = all_segments[segment_index - 1]["points"]
        # Take up to `context_points` from the END of the previous
        # segment, EXCLUDING its very last point if that point is the
        # same shared boundary point this segment's own list already
        # starts with (segments built by split_into_segments share a
        # boundary point where they were subdivided from one oversized
        # step — avoid adding that point twice).
        candidate = prev_points[:-1] if prev_points and prev_points[-1] == own_points[0] else prev_points
        pad_before = candidate[-context_points:] if candidate else []

    pad_after = []
    if segment_index < len(all_segments) - 1:
        next_points = all_segments[segment_index + 1]["points"]
        candidate = next_points[1:] if next_points and next_points[0] == own_points[-1] else next_points
        pad_after = candidate[:context_points] if candidate else []

    padded_points = pad_before + own_points + pad_after
    core_start = len(pad_before)
    core_end = core_start + len(own_points) - 1  # own edges: N points -> N-1 edges
    return padded_points, core_start, core_end


def score_segment(segment, region_data, route_elevations_by_point,
                   assigned_historic_features=None, all_segments=None, segment_index=None):
    """
    Score one segment using data that was ALREADY fetched for the
    whole route (region_data, route_elevations_by_point). No network
    calls happen in this function — it's pure local computation,
    which is the entire point of the redesign.

    assigned_historic_features: if provided, use this segment's
    pre-assigned subset of historic features (each feature credited
    to exactly one segment route-wide, via
    assign_features_to_nearest_segment) instead of checking against
    the full route-wide historic list — this is the fix for the
    cluster double-counting bug described above. Falls back to the
    full list if not provided, for backward compatibility.

    all_segments / segment_index: if BOTH provided, curvature is
    scored using a little context from neighbouring segments (see
    _curvature_context_points) so boundary-straddling curves aren't
    lost. If either is omitted, falls back to scoring curvature on
    this segment's own points only (the old behaviour) — kept for
    backward compatibility with any other caller of this function.
    """
    points = segment["points"]
    name = segment["name"]

    if len(points) < 3:
        return {"name": name, "note": "Segment too short to score meaningfully", "scores": {}}

    # --- Curvature (pure geometry, always available). Use padded
    # context from neighbouring segments when available, so a real
    # bend straddling this segment's boundary isn't lost. ---
    if all_segments is not None and segment_index is not None:
        padded_points, core_start, core_end = _curvature_context_points(all_segments, segment_index)
        curvature_result = score_curvature(padded_points, core_start=core_start, core_end=core_end)
    else:
        curvature_result = score_curvature(points)

    # --- Elevation: look up this segment's points in the already-
    # fetched whole-route elevation data, by coordinate. ---
    segment_elevations = [
        route_elevations_by_point.get(p) for p in points
    ]
    elevation_result = score_elevation(segment_elevations)

    # --- Proximity: score against data already fetched for the whole
    # route — purely local distance calculations now. ---
    water_result = score_proximity(points, region_data["water"])
    forest_result = score_proximity(points, region_data["forest"])

    # --- Village character (NEW): proximity to an officially
    # designated Conservation Area. This data was already being
    # fetched for the route (see score_route below) but, until this
    # version, was never actually plugged into a segment's score —
    # only reported separately as an informational "notable villages"
    # line. Falls back to an empty list if the route-level fetch
    # found nothing or failed, so this is always safe to call. ---
    village_result = score_proximity(points, region_data.get("conservation_areas", []))

    historic_features_for_this_segment = (
        assigned_historic_features if assigned_historic_features is not None
        else region_data["historic"]
    )
    historic_result = score_proximity(points, historic_features_for_this_segment)

    # --- Combine into spec categories (weights still a first-pass
    # hypothesis, flagged as tuneable) ---
    # Driving Enjoyment re-weighted from 80/20 to 60/40
    # curvature/elevation: testing on the Hathersage-Castleton route
    # showed famous "great driving road" segments (Winnats Pass,
    # Buxton Road) scoring relatively low on this metric despite
    # their real-world reputation, suggesting curvature alone was
    # over-weighted relative to elevation drama for this category.
    driving_enjoyment = (
        0.6 * curvature_result["curvature_score"]
        + 0.4 * elevation_result["elevation_score"]
    )
    # Scenery REWORKED (ported from graph_search/07_score_graph_
    # enjoyment.py, which had this fix but this file didn't): village
    # character added as a real input, per direct user judgement that
    # passing through a genuinely charming, officially-recognised
    # village/town is a real part of "scenery" in the broad sense this
    # category is meant to capture, not just water/forest/elevation.
    scenery = (
        0.3 * water_result["proximity_score"]
        + 0.3 * forest_result["proximity_score"]
        + 0.2 * elevation_result["elevation_score"]
        + 0.2 * village_result["proximity_score"]
    )
    history_culture = historic_result["proximity_score"]

    nearby_historic_names = [f["name"] for f in historic_features_for_this_segment][:5]
    nearby_village_names = [
        f["name"] for f in region_data.get("conservation_areas", [])
        if village_result["nearby_count"] > 0
    ][:5]

    return {
        "name": name,
        "length_m": curvature_result["total_length_m"],
        "scores": {
            "driving_enjoyment": round(driving_enjoyment, 3),
            "scenery": round(scenery, 3),
            "history_culture": round(history_culture, 3),
        },
        "raw_data": {
            "curvature": curvature_result,
            "elevation": elevation_result,
            "water": water_result,
            "forest": forest_result,
            "village": village_result,
            "historic": historic_result,
            "historic_sites_found": nearby_historic_names,
            "village_areas_nearby": nearby_village_names,
        },
    }


def score_route(start, end, waypoints=None, route_label="Route"):
    """
    Top-level function: fetch a route, fetch ALL needed data ONCE for
    the whole route, then score every segment using purely local
    computation against that already-fetched data.
    """
    t_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Scoring: {route_label}")
    print(f"{'=' * 60}")

    route_data = get_route_with_geometry(start, end, waypoints)
    route_summary = route_data["routes"][0]
    print(f"Total distance: {round(route_summary['distance'] / 1000, 1)} km")
    print(f"Total duration: {round(route_summary['duration'] / 60, 1)} minutes\n")

    segments = split_into_segments(route_data)
    print(f"Split into {len(segments)} segments\n")

    # --- Fetch ALL data for the WHOLE route, once, up front ---
    all_points = [p for seg in segments for p in seg["points"]]

    # Deduplicate before fetching elevation: segments can share a
    # boundary point, and Open Topo Data's free tier has a strict
    # 1000-calls/day cap, so we don't want to pay for the same
    # coordinate twice.
    unique_points = list(dict.fromkeys(all_points))

    print(f"Fetching elevation for the whole route "
          f"({len(unique_points)} unique points, one batched call)...")
    unique_elevations = fetch_elevations(unique_points)
    route_elevations_by_point = dict(zip(unique_points, unique_elevations))
    print(f"  Got elevation for {len([e for e in unique_elevations if e is not None])} "
          f"of {len(unique_points)} unique points.\n")

    bbox = route_bounding_box(all_points)

    try:
        region_data = fetch_region_data(bbox)
    except Exception as e:
        print(f"  Could not fetch region data: {e}")
        print("  Continuing with empty water/forest/historic data for this route.\n")
        region_data = {"water": [], "forest": [], "historic": [], "places": []}

    # --- Road reputation lookups (Wikipedia-grounded), BOUNDED ---
    # Bounded to the top 10 longest non-generic, uniquely-named
    # --- Road reputation lookups (Wikipedia, per ROAD name) ---
    # DISABLED per decision: testing showed this mostly measures
    # "is the destination town/village famous" rather than "is this
    # road itself a good drive" — e.g. Buxton Road and Hope Road both
    # matched to the generic Hathersage village article, not anything
    # specific to those roads. Left in place (commented, not deleted)
    # in case it's revisited with a different approach later.
    #
    # MAX_REPUTATION_LOOKUPS = 10
    # place_context = start.get("name", "")
    # candidate_segments = [
    #     seg for seg in segments
    #     if seg["name"].strip().lower() not in GENERIC_NAME_SKIP_LIST
    # ]
    # seen_names = set()
    # unique_named_segments = []
    # for seg in sorted(candidate_segments, key=lambda s: -_approx_length_m(s["points"])):
    #     if seg["name"] not in seen_names:
    #         seen_names.add(seg["name"])
    #         unique_named_segments.append(seg)
    # lookup_targets = unique_named_segments[:MAX_REPUTATION_LOOKUPS]
    # reputation_by_name = {}
    # if lookup_targets:
    #     print(f"  Checking road reputation for top {len(lookup_targets)} named roads "
    #           f"(Wikipedia, bounded lookup)...")
    #     for seg in lookup_targets:
    #         try:
    #             result = get_road_reputation(seg["name"], place_context)
    #             reputation_by_name[seg["name"]] = result
    #             if result["found"]:
    #                 print(f"    '{seg['name']}': found real source — {result['source_title']}")
    #         except Exception as e:
    #             print(f"    (Reputation lookup failed for '{seg['name']}': {e})")
    #             reputation_by_name[seg["name"]] = {"found": False, "source_title": None, "source_url": None, "extract": None}
    #     print()
    reputation_by_name = {}

    # --- Notable village/town matching via official Conservation
    # Areas data (replaces both the earlier road-reputation idea AND
    # the hardcoded village list) — fetches every officially
    # designated Conservation Area in the route's bounding box from
    # Historic England's open dataset, then checks which ones the
    # route actually passes close to (same distance-based proximity
    # pattern as water/forest/historic). This scales to ANY route in
    # England, not just the ~25 villages a hardcoded list happened to
    # cover, and is grounded in a real official designation rather
    # than a magazine's subjective opinion. Known limitation: this
    # dataset has incomplete coverage (some councils haven't
    # submitted data) — a "no match" here means "no signal," not
    # "this place definitely isn't charming."
    #
    # NEW: every Conservation Area found in the bbox is now printed by
    # NAME (not just a count), regardless of whether it ends up close
    # enough to the route to count toward the score. This is the
    # direct way to answer "did the fetch even find <specific place>
    # at all" — separate from "was it close enough to matter" — since
    # those are two different possible failure points and conflating
    # them was making the earlier "0 matches" conclusion ambiguous
    # (real Historic England coverage gap, vs a found-but-too-far
    # area, vs a real fetch/match bug). ---
    try:
        conservation_areas = fetch_conservation_areas_in_bbox(bbox, debug=True)
        print(f"  Found {len(conservation_areas)} Conservation Area(s) in the route's "
              f"bounding box: {[a['name'] for a in conservation_areas]}")
    except Exception as e:
        print(f"  Could not fetch Conservation Areas data: {e}\n")
        conservation_areas = []

    # NEW: make conservation_areas available to score_segment via
    # region_data, so it can actually be used as a scenery input
    # rather than only ever appearing in the informational
    # "notable villages" line below.
    region_data["conservation_areas"] = conservation_areas

    notable_villages_found = []
    if conservation_areas:
        # Use a slightly wider radius than other proximity checks —
        # a conservation area's centroid can sit a little away from
        # the exact road if the designated area is large, and we
        # want to credit "passes near/through this place" rather
        # than only "drives directly across its centre point."
        nearby_result = score_proximity(
            all_points, conservation_areas,
            close_threshold_m=200, score_decay_m=1000
        )
        if nearby_result["nearby_count"] > 0:
            # Identify WHICH areas specifically were close, for
            # reporting (score_proximity gives us a count/score but
            # not names, so check each individually — fine to do
            # here since conservation_areas per route is a short list).
            for area in conservation_areas:
                single_check = score_proximity(all_points, [area], close_threshold_m=200, score_decay_m=1000)
                if single_check["proximity_score"] > 0:
                    notable_villages_found.append(area["name"])

    if notable_villages_found:
        print(f"  Notable villages/towns on this route (official Conservation Areas): "
              f"{notable_villages_found}\n")
    else:
        print(f"  No Conservation Areas found close enough to the route to count "
              f"(see full bbox list above for what was found at all, regardless of "
              f"distance to the route).\n")

    # --- Assign each historic feature to its single nearest segment,
    # ONCE for the whole route, fixing the cluster double-counting
    # bug described in assign_features_to_nearest_segment's docstring. ---
    historic_assignment = assign_features_to_nearest_segment(segments, region_data["historic"])

    # --- Now score every segment using only local computation ---
    segment_scores = []
    for i, seg in enumerate(segments):
        result = score_segment(
            seg, region_data, route_elevations_by_point,
            assigned_historic_features=historic_assignment[i],
            all_segments=segments, segment_index=i,
        )
        result["reputation"] = reputation_by_name.get(seg["name"], {"found": False, "source_title": None, "source_url": None, "extract": None})
        segment_scores.append(result)

        if result.get("scores"):
            print(f"  {result['name']} ({round(result['length_m'])}m)")
            print(f"    Driving Enjoyment: {result['scores']['driving_enjoyment']}")
            print(f"    Scenery:           {result['scores']['scenery']}")
            print(f"    History & Culture: {result['scores']['history_culture']}")
            if result["raw_data"]["historic_sites_found"]:
                print(f"    Nearby sites: {result['raw_data']['historic_sites_found']}")
            if result["raw_data"]["village_areas_nearby"]:
                print(f"    Nearby Conservation Area(s): {result['raw_data']['village_areas_nearby']}")
            if result["reputation"]["found"]:
                print(f"    Road reputation: {result['reputation']['source_title']} "
                      f"({result['reputation']['source_url']})")
                print(f"      \"{result['reputation']['extract'][:200]}...\"")
            print()

    total_length = sum(s.get("length_m", 0) for s in segment_scores)
    if total_length > 0:
        overall = {}
        for category in ["driving_enjoyment", "scenery", "history_culture"]:
            weighted_sum = sum(
                s["scores"].get(category, 0) * s.get("length_m", 0)
                for s in segment_scores if s.get("scores")
            )
            overall[category] = round(weighted_sum / total_length, 3)

        print(f"--- {route_label}: Overall (length-weighted average) ---")
        for category, value in overall.items():
            print(f"  {category}: {value}")

        # --- "Best stretch" score, addressing length dilution ---
        # The length-weighted average treats a route as "mostly what
        # its longest segments are like" — a real problem when a
        # route has spectacular peak segments (e.g. Winnats Pass,
        # Buxton Road scoring 0.7+) connected by long, ordinary
        # stretches. A driver's actual experience and memory of a
        # route is disproportionately shaped by its best moments, not
        # its mean. This computes the same length-weighted average,
        # but restricted to the top BEST_STRETCH_FRACTION of the
        # route's total length, by score, per category — "how good
        # are this route's best moments," distinct from "what's the
        # average throughout." Both metrics are kept; neither replaces
        # the other, since the overall picture still matters too.
        BEST_STRETCH_FRACTION = 0.3
        best_stretch_target_m = total_length * BEST_STRETCH_FRACTION

        best_stretch = {}
        for category in ["driving_enjoyment", "scenery", "history_culture"]:
            scored_segments = [
                s for s in segment_scores if s.get("scores")
            ]
            sorted_by_category = sorted(
                scored_segments,
                key=lambda s: s["scores"].get(category, 0),
                reverse=True,
            )
            running_length = 0.0
            weighted_sum = 0.0
            for s in sorted_by_category:
                if running_length >= best_stretch_target_m:
                    break
                seg_len = s.get("length_m", 0)
                weighted_sum += s["scores"].get(category, 0) * seg_len
                running_length += seg_len
            best_stretch[category] = round(weighted_sum / running_length, 3) if running_length > 0 else 0.0

        print(f"\n--- {route_label}: Best {int(BEST_STRETCH_FRACTION*100)}% of route (peak-weighted) ---")
        for category, value in best_stretch.items():
            print(f"  {category}: {value}")

    elapsed = round(time.time() - t_start, 1)
    print(f"\n(Took {elapsed} seconds)")

    return {
        "segments": segment_scores,
        "distance_km": round(route_summary["distance"] / 1000, 1),
        "duration_min": round(route_summary["duration"] / 60, 1),
        "overall": overall if total_length > 0 else None,
        "best_stretch": best_stretch if total_length > 0 else None,
        "notable_villages": notable_villages_found,
    }


def compare_routes(direct_result, alt_result, alt_label="Alternative route"):
    """
    Report the explicit tradeoff between a direct route and an
    alternative: how much MORE time/distance does the alternative
    cost, alongside how much BETTER it scores. This addresses a real
    gap — previously, extra length only ever hurt a route's score via
    dilution in the average, but the actual cost (e.g. "12 extra
    minutes") was never shown explicitly. Surfacing both sides lets
    the tradeoff be judged on its own terms, rather than silently
    baked into one diluted number.
    """
    extra_distance_km = round(alt_result["distance_km"] - direct_result["distance_km"], 1)
    extra_duration_min = round(alt_result["duration_min"] - direct_result["duration_min"], 1)
    extra_distance_pct = round(
        (extra_distance_km / direct_result["distance_km"]) * 100, 0
    ) if direct_result["distance_km"] > 0 else 0

    print(f"\n--- Detour cost: {alt_label} vs direct ---")
    print(f"  Extra distance: +{extra_distance_km} km ({extra_distance_pct:+.0f}%)")
    print(f"  Extra time:     +{extra_duration_min} minutes")

    if direct_result["overall"] and alt_result["overall"]:
        print(f"\n  Score improvement (overall average):")
        for category in ["driving_enjoyment", "scenery", "history_culture"]:
            diff = round(alt_result["overall"][category] - direct_result["overall"][category], 3)
            print(f"    {category}: {diff:+.3f}")

    if direct_result["best_stretch"] and alt_result["best_stretch"]:
        print(f"\n  Score improvement (best-stretch / peak moments):")
        for category in ["driving_enjoyment", "scenery", "history_culture"]:
            diff = round(alt_result["best_stretch"][category] - direct_result["best_stretch"][category], 3)
            print(f"    {category}: {diff:+.3f}")

    direct_villages = set(direct_result.get("notable_villages", []))
    alt_villages = set(alt_result.get("notable_villages", []))
    extra_villages = alt_villages - direct_villages
    if extra_villages:
        print(f"\n  Notable villages/towns the alternative passes that the direct route doesn't: {sorted(extra_villages)}")


if __name__ == "__main__":
    # Re-run the real test case that exposed both bugs fixed in this
    # version: Tillington -> Haslemere via Upperton/Lurgashall.
    TILLINGTON = {"name": "Tillington", "lat": 50.9896, "lon": -0.6313}
    HASLEMERE = {"name": "Haslemere", "lat": 51.089, "lon": -0.710}
    UPPERTON = {"name": "Upperton", "lat": 50.996, "lon": -0.637}
    LURGASHALL = {"name": "Lurgashall", "lat": 51.035, "lon": -0.6655}

    direct_result = score_route(TILLINGTON, HASLEMERE, route_label="Direct (fastest) route")
    scenic_result = score_route(
        TILLINGTON, HASLEMERE, waypoints=[UPPERTON, LURGASHALL],
        route_label="Route via Upperton / Lurgashall"
    )
    compare_routes(direct_result, scenic_result, alt_label="Route via Upperton / Lurgashall")