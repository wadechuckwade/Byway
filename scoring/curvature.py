"""
Byway — Curvature Scoring
==========================

What this does, in plain terms:
Takes a sequence of GPS points (the shape of a road) and works out how
"curvy" it is. This is the "Driving Enjoyment" category from the spec.

How it works (the actual method used by existing curvy-road-finder
tools like roadcurvature.com):
For every set of 3 consecutive points along the road, fit a circle
through them. A tight circle (small radius) means a sharp bend. A huge
circle (or no circle at all, i.e. a straight line) means the road is
basically straight there. We then look at what fraction of the road's
total length is spent in tight curves — that fraction becomes our
curvature score.

This is real geometry, not a guess: it only needs the road's shape,
which we already get for free from OSRM/OSM. No new API calls needed.

BOUNDARY-CURVE FIX (added after real-route testing on the
Tillington->Haslemere "via Upperton/Lurgashall" route): when a route
gets chopped into many short segments — which happens a lot for
unnamed minor lanes, since they have no name tag to merge consecutive
OSRM steps on — a real bend whose 3-point window straddles the cut
between two segments was previously invisible to BOTH segments, since
each one only ever saw its own point list. A continuous winding lane
made of many short unnamed segments could lose a meaningful fraction
of its real curvature this way, purely as an artifact of how finely
it happened to get chopped — which systematically disadvantages
exactly the kind of real backroad this project cares most about
scoring well.

Fix: callers can now pass a few extra CONTEXT points from neighbouring
segments before/after the real segment, plus core_start/core_end
marking which part of the (possibly padded) point list is this
segment's own. The full padded list is used to correctly detect
whether a boundary-straddling window is a tight curve; only length
falling inside [core_start, core_end) is counted toward this
segment's own score and length, so padding never double-counts a
neighbour's length as if it belonged to this segment too.

Default behaviour (no core_start/core_end passed) is unchanged —
existing callers (e.g. graph_search's whole-way scoring, the
self-tests below) work exactly as before.
"""

import math


def haversine_distance_m(p1, p2):
    """
    Distance in metres between two (lon, lat) points on the Earth's
    surface. We use this instead of simple Pythagoras because lines of
    longitude get closer together near the poles — at UK latitudes this
    matters enough to get wrong if ignored.
    """
    lon1, lat1 = p1
    lon2, lat2 = p2
    R = 6371000  # Earth's radius in metres

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def radius_of_curvature(p1, p2, p3):
    """
    Fit a circle through 3 consecutive points and return its radius in
    metres. A small radius = a sharp bend. Returns None if the points
    are (near enough) in a straight line, since a straight line has no
    meaningful circle (infinite radius).

    Points are (lon, lat) tuples. For this local-scale calculation we
    treat lon/lat as flat x/y, which is an acceptable approximation
    over the short distances between consecutive route points (tens of
    metres), even though it would break down over large distances.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Convert the small lon/lat differences to approximate metres so
    # the radius comes out in metres rather than in degrees.
    # 1 degree latitude ≈ 111,320 m everywhere.
    # 1 degree longitude ≈ 111,320 * cos(latitude) m.
    lat_ref = y2
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(lat_ref))

    x1, y1 = x1 * m_per_deg_lon, y1 * m_per_deg_lat
    x2, y2 = x2 * m_per_deg_lon, y2 * m_per_deg_lat
    x3, y3 = x3 * m_per_deg_lon, y3 * m_per_deg_lat

    # Standard circle-through-3-points formula.
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    e = ((x1 ** 2 - x2 ** 2) - (y2 ** 2 - y1 ** 2)) / 2
    f = ((x1 ** 2 - x3 ** 2) - (y3 ** 2 - y1 ** 2)) / 2

    denominator = (a * d - b * c)
    if abs(denominator) < 1e-9:
        # Points are colinear (or virtually so) — straight line, no
        # meaningful curve here.
        return None

    center_x = (d * e - b * f) / denominator
    center_y = (a * f - c * e) / denominator

    radius = math.hypot(x1 - center_x, y1 - center_y)
    return radius


def score_curvature(points, tight_curve_threshold_m=150, core_start=None, core_end=None):
    """
    Given a list of (lon, lat) points describing a road, return a
    curvature score from 0 (dead straight) to 1 (very twisty).

    tight_curve_threshold_m: any curve with a radius smaller than this
    counts as a "tight curve" for scoring purposes. 150m is a starting
    point — roughly the kind of bend that's noticeable but not a
    hairpin. This is exactly the kind of number we expect to tune once
    we've tested it against real routes (see Decisions Log).

    core_start / core_end: see module docstring's "BOUNDARY-CURVE FIX"
    above. Default (None, None) means "the whole list is this
    segment's own" — identical to the original behaviour, fully
    backward compatible. Pass these when `points` has been padded with
    a little context from neighbouring segments, so curves straddling
    the real boundary still get detected, without that context's own
    length being double-counted into this segment's total.

    Returns a dict with the score plus the raw numbers behind it, so
    we're never hiding how the score was reached — this feeds directly
    into the spec's "tap to see breakdown" requirement later.
    """
    if len(points) < 3:
        return {
            "curvature_score": 0.0,
            "total_length_m": 0.0,
            "tight_curve_length_m": 0.0,
            "note": "Not enough points to assess curvature",
        }

    if core_start is None:
        core_start = 0
    if core_end is None:
        core_end = len(points)

    total_length_m = 0.0
    tight_curve_length_m = 0.0

    for i in range(len(points) - 2):
        p1, p2, p3 = points[i], points[i + 1], points[i + 2]
        seg_length = haversine_distance_m(p1, p2)

        # Attribute this edge (p1 -> p2, index i -> i+1) to whichever
        # segment's core range owns its LEFT point. Padding points
        # added purely so a boundary curve has real neighbours to form
        # a 3-point window with never get their own length counted
        # here — that length still belongs to whichever segment
        # originally contained them, and gets counted there when THAT
        # segment's score_curvature call runs.
        if not (core_start <= i < core_end):
            continue

        total_length_m += seg_length

        radius = radius_of_curvature(p1, p2, p3)
        if radius is not None and radius < tight_curve_threshold_m:
            tight_curve_length_m += seg_length

    # Add the final segment's length (the loop above only counts up to
    # the second-to-last point pair), same core-ownership check.
    final_edge_idx = len(points) - 2
    if len(points) >= 2 and core_start <= final_edge_idx < core_end:
        total_length_m += haversine_distance_m(points[-2], points[-1])

    if total_length_m == 0:
        score = 0.0
    else:
        score = tight_curve_length_m / total_length_m

    return {
        "curvature_score": round(score, 3),
        "total_length_m": round(total_length_m, 1),
        "tight_curve_length_m": round(tight_curve_length_m, 1),
    }


if __name__ == "__main__":
    # Self-test with two synthetic roads: one straight, one curvy.
    # This lets us sanity-check the maths without needing internet
    # access — real route geometry gets plugged in once this is proven
    # correct.

    print("--- Test 1: a dead-straight road ---")
    straight_road = [(-0.60 + i * 0.001, 50.88) for i in range(20)]
    result = score_curvature(straight_road)
    print(result)
    assert result["curvature_score"] == 0.0, "Straight road should score 0"
    print("PASSED: straight road scores 0\n")

    print("--- Test 2: a tight, winding road ---")
    winding_road = []
    for i in range(60):
        angle = i * 0.3
        # Small sine-wave wiggle to simulate a twisty country lane.
        lon = -0.60 + i * 0.0005
        lat = 50.88 + math.sin(angle) * 0.0015
        winding_road.append((lon, lat))
    result = score_curvature(winding_road)
    print(result)
    assert result["curvature_score"] > 0.0, "Winding road should score above 0"
    print("PASSED: winding road scores above 0\n")

    print("--- Test 3: comparison ---")
    print(f"Straight road score: {score_curvature(straight_road)['curvature_score']}")
    print(f"Winding road score:  {score_curvature(winding_road)['curvature_score']}")
    print("\nIf the winding road scores meaningfully higher, the logic is sound.")

    print("\n--- Test 4 (NEW): a curve split across a segment boundary is still detected ---")
    # Build one continuous tight curve, then simulate chopping it into
    # two "segments" that SHARE a boundary point at index 6 — matching
    # how real adjacent segments are built in this codebase (chunks
    # deliberately overlap by one point, per split_into_segments).
    # Index 6 is deliberately chosen to land mid-bend (verified by
    # inspecting per-window radii beforehand): the window at original
    # i=5 (points 5,6,7) is a genuine tight curve that neither half
    # can see on its own under the OLD behaviour, since point 5
    # belongs only to segment A and point 7 belongs only to segment B.
    full_curve = []
    for i in range(20):
        angle = i * 0.25
        lon = -0.60 + i * 0.0004
        lat = 50.88 + math.sin(angle) * 0.0012
        full_curve.append((lon, lat))

    SPLIT = 6  # shared boundary point index

    # Whole curve scored as one piece (ground truth).
    whole_result = score_curvature(full_curve)

    # OLD behaviour: each half only sees its own points (sharing the
    # boundary point, as real segments do) — no context beyond that.
    half_a = full_curve[0:SPLIT + 1]       # indices 0..6 (7 points)
    half_b = full_curve[SPLIT:]            # indices 6..19 (14 points)
    old_a = score_curvature(half_a)
    old_b = score_curvature(half_b)
    old_total_tight = old_a["tight_curve_length_m"] + old_b["tight_curve_length_m"]
    old_total_length = old_a["total_length_m"] + old_b["total_length_m"]

    # FIXED behaviour: pad each half with 2 extra points borrowed from
    # its neighbour, and use core_start/core_end so each segment only
    # claims length/curve-length for its OWN points — the padding
    # exists purely to let boundary windows be evaluated correctly.
    PAD = 2
    padded_a = full_curve[0:SPLIT + 1] + full_curve[SPLIT + 1:SPLIT + 1 + PAD]
    new_a = score_curvature(padded_a, core_start=0, core_end=len(half_a) - 1)

    padded_b = full_curve[SPLIT - PAD:SPLIT] + full_curve[SPLIT:]
    new_b = score_curvature(padded_b, core_start=PAD, core_end=PAD + len(half_b) - 1)

    new_total_tight = new_a["tight_curve_length_m"] + new_b["tight_curve_length_m"]
    new_total_length = new_a["total_length_m"] + new_b["total_length_m"]

    print(f"Whole curve, scored as one piece:                {whole_result['tight_curve_length_m']}m tight, {whole_result['total_length_m']}m total")
    print(f"Chopped at boundary, NO padding (old behaviour):  {round(old_total_tight, 1)}m tight, {round(old_total_length, 1)}m total")
    print(f"Chopped at boundary, WITH padding (fixed):        {round(new_total_tight, 1)}m tight, {round(new_total_length, 1)}m total")

    assert new_total_tight > old_total_tight, (
        "Padded version should detect MORE real curvature than the "
        "unpadded version at this deliberately-chosen boundary — "
        "if this fails, the boundary loss this fix targets isn't "
        "actually being demonstrated"
    )
    assert abs(new_total_tight - whole_result["tight_curve_length_m"]) < 0.5, (
        "Padded halves should recover essentially the SAME tight-curve "
        "length as scoring the whole curve in one piece"
    )
    assert abs(new_total_length - whole_result["total_length_m"]) < 0.5, (
        "Padding must not double-count length — halves should still "
        "sum to the original whole-route length, not more"
    )
    print("PASSED — padding recovers boundary curvature without double-counting length\n")