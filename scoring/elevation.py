"""
Byway — Elevation Scoring
===========================

What this does, in plain terms:
Takes a sequence of GPS points along a road and asks a free elevation
API (Open Topo Data, built on NASA's SRTM satellite data) how high
above sea level each point is. From that we compute total climbing —
a road that goes up and down through hills "feels" more dramatic and
adventurous than a flat one, even at the same distance.

This feeds the "Scenery" and "Adventure" categories from the spec
(elevation is a named input to both).

NEW — PERSISTENT LOCAL CACHE (after a real run on the Eynsham/Burford
graph — 26,773 points — took 8-10+ minutes): Open Topo Data's free
tier allows max 1 request/second. For a WHOLE BOUNDED GRAPH (not a
single route, which is all this ever fetched for before Milestone 1.5)
that's ~298 batches x 1.1s = ~5.5 minutes on the deliberate rate-limit
delay alone, before any network latency. That limit is real and
shouldn't be bypassed — exceeding a free public service's stated rate
limit risks this project's outbound IP getting blocked entirely, which
would be far worse than a slow run.

What CAN be fixed: this project re-fetches the SAME bounding areas
repeatedly during iterative testing — this exact graph was fetched at
least 4 times in one session. A persistent local file cache (keyed by
rounded lat/lon, ~1m precision) means every point already seen in ANY
previous run — for this graph or any overlapping one — is answered
instantly with zero network calls. The FIRST fetch for a genuinely new
area is unavoidably still slow (the rate limit is real and respected);
every subsequent run reusing any of those points is not.

Network note: needs real internet access (e.g. GitHub Codespaces),
not Claude's sandboxed tool environment — except for points already
in the local cache, which need no network at all.
"""

import json
import os
import time
import requests


OPENTOPODATA_DELAY_SECONDS = 1.1  # API allows max 1 call/second; small margin added
OPENTOPODATA_MAX_RETRIES = 4

ELEVATION_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".elevation_cache.json",
)
# Stored at the project root (one level up from scoring/), so it
# persists across runs and across every script in this project that
# needs elevation for an overlapping area — not just the same route.

CACHE_COORD_PRECISION = 5
# ~1.1m at UK latitudes — precise enough to treat genuinely-identical
# graph nodes as cache hits (common: multiple ways/segments often
# share boundary nodes), while not accidentally merging two real,
# distinct nearby points into the same cache entry.


def _load_cache():
    if os.path.exists(ELEVATION_CACHE_PATH):
        try:
            with open(ELEVATION_CACHE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"    (Could not read elevation cache, starting fresh: {e})")
            return {}
    return {}


def _save_cache(cache):
    try:
        with open(ELEVATION_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError as e:
        print(f"    (Could not save elevation cache — this run's new data won't "
              f"speed up future runs, but isn't otherwise affected: {e})")


def _cache_key(lon, lat):
    return f"{round(lat, CACHE_COORD_PRECISION)},{round(lon, CACHE_COORD_PRECISION)}"


def fetch_elevations(points, batch_size=90, use_cache=True):
    """
    Given a list of (lon, lat) points, return a list of elevations in
    metres, in the same order.

    batch_size is kept under Open Topo Data's 100-locations-per-request
    limit, with a little headroom.

    The free public API allows max 1 request per second — we space
    NEW (not-yet-cached) batches out to respect that, and retry with
    backoff if we still get rate-limited (429), since on a shared
    environment like Codespaces other usage on the same network can
    use up that allowance unpredictably.

    use_cache: if True (default), checks the persistent local cache
    first for every point, only hitting the network for genuinely new
    ones, and saves newly-fetched points back to the cache for future
    runs. Set False to bypass the cache entirely (e.g. to force a
    fresh fetch, or to test the cache mechanism itself).

    Returns None for any point where elevation data wasn't available
    (e.g. data gap) rather than crashing — this matters for the
    confidence scoring planned later: missing data should be visible,
    not silently treated as zero.
    """
    cache = _load_cache() if use_cache else {}
    elevations = [None] * len(points)
    points_to_fetch = []  # list of (original_index, (lon, lat))

    cache_hits = 0
    for idx, (lon, lat) in enumerate(points):
        key = _cache_key(lon, lat)
        if use_cache and key in cache:
            elevations[idx] = cache[key]
            cache_hits += 1
        else:
            points_to_fetch.append((idx, (lon, lat)))

    if use_cache and cache_hits:
        print(f"    {cache_hits} of {len(points)} points already cached from a "
              f"previous run — skipping the network for those.")

    if not points_to_fetch:
        return elevations

    fetch_points = [p for _, p in points_to_fetch]
    num_batches = (len(fetch_points) + batch_size - 1) // batch_size
    if num_batches > 1:
        est_minutes = round(num_batches * OPENTOPODATA_DELAY_SECONDS / 60, 1)
        print(f"    Fetching {len(fetch_points)} new point(s) from Open Topo Data "
              f"({num_batches} batches, max 1 request/second — genuinely rate-limited, "
              f"not a bug — expect at least ~{est_minutes} min just from that delay)...")

    for batch_num, i in enumerate(range(0, len(fetch_points), batch_size)):
        if batch_num > 0:
            time.sleep(OPENTOPODATA_DELAY_SECONDS)
        if num_batches > 10 and batch_num > 0 and batch_num % 20 == 0:
            print(f"      ... batch {batch_num}/{num_batches} "
                  f"({round(100 * batch_num / num_batches)}%)")

        batch = fetch_points[i:i + batch_size]
        batch_orig_indices = [points_to_fetch[i + j][0] for j in range(len(batch))]

        # Open Topo Data wants "lat,lon" pairs (note: opposite order
        # from the (lon, lat) convention we use elsewhere, matching
        # how OSRM does it — easy to mix up, flagging it clearly).
        locations_param = "|".join(f"{lat},{lon}" for lon, lat in batch)
        url = "https://api.opentopodata.org/v1/srtm30m"

        for attempt in range(OPENTOPODATA_MAX_RETRIES):
            response = requests.post(
                url,
                data={"locations": locations_param},
                timeout=20,
            )
            if response.status_code == 429:
                wait_time = OPENTOPODATA_DELAY_SECONDS * (attempt + 2)
                print(f"    (Open Topo Data rate limit hit, waiting {wait_time:.1f}s and retrying...)")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            result = response.json()
            for j, r in enumerate(result.get("results", [])):
                elevation = r.get("elevation")
                elevations[batch_orig_indices[j]] = elevation
                if use_cache and elevation is not None:
                    lon, lat = batch[j]
                    cache[_cache_key(lon, lat)] = elevation
            break
        else:
            # Exhausted retries for this batch — record these points
            # as missing data rather than crashing the whole route.
            print(f"    (Giving up on elevation batch {batch_num + 1}/{num_batches} "
                  f"after {OPENTOPODATA_MAX_RETRIES} attempts — recording as missing data)")
            for orig_idx in batch_orig_indices:
                elevations[orig_idx] = None

    if use_cache:
        _save_cache(cache)

    return elevations


def _haversine_distance_m(p1, p2):
    """Same formula used throughout this project, kept local rather
    than imported — matches this project's existing convention of each
    module having its own small copy (see curvature.py, 04_pathfinding.py)."""
    import math
    R = 6371000
    lon1, lat1 = p1
    lon2, lat2 = p2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def subsample_points_by_distance(points, target_spacing_m=100):
    """
    NEW — given an ordered list of (lon, lat) points along a path,
    return a reduced list of (original_index, point) pairs at roughly
    target_spacing_m intervals, always including the first and last
    point.

    WHY THIS EXISTS: elevation changes smoothly over short distances —
    querying every single graph node individually (often 10-50m apart)
    is far finer resolution than the scoring actually needs, and each
    point costs real time against Open Topo Data's 1-request/second
    limit. Sampling every ~100m and interpolating the rest (see
    interpolate_elevations() below) cuts the number of points that
    need fetching, with minimal real accuracy loss for a "how hilly is
    this road" score — UK hills don't meaningfully change shape within
    a single 100m stretch.

    Returns a list of (original_index, point) tuples — pass this
    straight to fetch_elevations() (extracting just the points) to get
    the values needed for interpolate_elevations().
    """
    if len(points) <= 2:
        return list(enumerate(points))

    sampled = [(0, points[0])]
    cumulative_since_last = 0.0
    for i in range(1, len(points)):
        cumulative_since_last += _haversine_distance_m(points[i - 1], points[i])
        if cumulative_since_last >= target_spacing_m:
            sampled.append((i, points[i]))
            cumulative_since_last = 0.0

    if sampled[-1][0] != len(points) - 1:
        sampled.append((len(points) - 1, points[-1]))

    return sampled


def interpolate_elevations(points, sampled_index_elevation_pairs):
    """
    NEW — given the FULL ordered point list and a set of (index,
    elevation) pairs for a SUBSET of those points (typically from
    subsample_points_by_distance() + fetch_elevations()), linearly
    interpolate elevation for every point in between, based on
    along-path DISTANCE (not raw index position, since real point
    spacing along a road is rarely uniform — a tight bend has points
    closer together than a straight stretch).

    Points between two sampled points where EITHER side's elevation is
    None (missing data) are left as None rather than fabricating a
    number — consistent with this module's existing "missing data
    should be visible, not silently treated as zero" principle.

    Returns a list of elevations, same length and order as `points`.
    """
    if not sampled_index_elevation_pairs:
        return [None] * len(points)

    cumdist = [0.0] * len(points)
    for i in range(1, len(points)):
        cumdist[i] = cumdist[i - 1] + _haversine_distance_m(points[i - 1], points[i])

    result = [None] * len(points)
    for idx, elev in sampled_index_elevation_pairs:
        result[idx] = elev

    sampled_sorted = sorted(sampled_index_elevation_pairs, key=lambda x: x[0])
    for k in range(len(sampled_sorted) - 1):
        idx_a, elev_a = sampled_sorted[k]
        idx_b, elev_b = sampled_sorted[k + 1]
        if elev_a is None or elev_b is None:
            continue  # can't interpolate across a missing-data gap
        span = cumdist[idx_b] - cumdist[idx_a]
        for i in range(idx_a + 1, idx_b):
            if span <= 0:
                result[i] = elev_a
            else:
                frac = (cumdist[i] - cumdist[idx_a]) / span
                result[i] = elev_a + frac * (elev_b - elev_a)

    return result


def fetch_elevations_along_path(points, target_spacing_m=100, batch_size=90, use_cache=True):
    """
    NEW — convenience wrapper combining subsampling + fetching +
    interpolation for a single ordered path's worth of points. This is
    the function callers should normally use for a real road's point
    sequence, rather than manually orchestrating subsample_points_by_
    distance() / fetch_elevations() / interpolate_elevations()
    separately.

    Returns a list of elevations, same length/order as `points`.
    """
    if len(points) < 3:
        return fetch_elevations(points, batch_size=batch_size, use_cache=use_cache)

    sampled_pairs = subsample_points_by_distance(points, target_spacing_m=target_spacing_m)
    sample_points = [p for _, p in sampled_pairs]
    sample_elevations = fetch_elevations(sample_points, batch_size=batch_size, use_cache=use_cache)
    sampled_with_elev = [(idx, elev) for (idx, _), elev in zip(sampled_pairs, sample_elevations)]
    return interpolate_elevations(points, sampled_with_elev)


def score_elevation(elevations, total_distance_m=None):
    """
    Given a list of elevations (metres) along a route, work out total
    ascent, total descent, and a simple 0-1 "elevation drama" score.

    NEW — GRADIENT-BASED SCORING (optional, via total_distance_m): the
    original scoring normalised against a FIXED ABSOLUTE CLIMB (200m),
    calibrated for Milestone 1's segments — often multi-km merged
    stretches (e.g. a 6.9km named road). The graph-search pipeline
    scores individual graph WAYS, frequently just 100-600m long — a
    300m-long way physically cannot climb 200m (a 67% gradient), so
    almost every short way was structurally capped near-zero on this
    metric regardless of how dramatic its actual terrain was. A fixed
    absolute-metres threshold is scale-DEPENDENT: 200m of climb spread
    over 6km (3.3% average grade) is gentle, unremarkable hill country
    on a real drive; the same 200m over 300m is a serious mountain
    pass — the old method scored both as equally maximal (1.0),
    unable to tell them apart.

    Fix: when total_distance_m is provided, score based on AVERAGE
    GRADIENT (rise/run) instead — scale-invariant, works the same for
    a 100m way or a 5km route. reasonable_max_gradient=0.08 (8%
    average grade) is a deliberate first-pass hypothesis for "very
    hilly/dramatic" on a UK road, same spirit as every other tuneable
    threshold in this project — flagged, not finished science.

    Falls back to the OLD absolute-climb behaviour if total_distance_m
    isn't provided, for backward compatibility with any caller that
    doesn't have it handy (e.g. milestone_1_score_route.py, whose
    longer merged segments were the original calibration target for
    that approach and haven't been re-examined here).

    HONEST CAVEAT: very short ways (well under ~60m) risk noisy
    gradient readings given SRTM's ~30m horizontal resolution — a
    "measured" gradient over a very short span may reflect elevation-
    data granularity as much as real terrain. Not specially handled
    here; worth revisiting if short-way gradient scores look erratic
    in practice.
    """
    clean_elevations = [e for e in elevations if e is not None]

    if len(clean_elevations) < 2:
        return {
            "total_ascent_m": 0.0,
            "total_descent_m": 0.0,
            "elevation_score": 0.0,
            "missing_points": len(elevations) - len(clean_elevations),
            "note": "Not enough elevation data to score",
        }

    total_ascent = 0.0
    total_descent = 0.0
    for i in range(1, len(clean_elevations)):
        diff = clean_elevations[i] - clean_elevations[i - 1]
        if diff > 0:
            total_ascent += diff
        else:
            total_descent += abs(diff)

    if total_distance_m is not None and total_distance_m > 0:
        average_gradient = total_ascent / total_distance_m
        reasonable_max_gradient = 0.08  # 8% average gradient = "very hilly/dramatic"
        score = min(average_gradient / reasonable_max_gradient, 1.0)
    else:
        # Old behaviour: assume 200m of total climbing over a route of
        # this sample length is "very hilly" — a deliberately tuneable
        # assumption, calibrated for Milestone 1's longer segments.
        reasonable_max_climb = 200
        score = min(total_ascent / reasonable_max_climb, 1.0)

    return {
        "total_ascent_m": round(total_ascent, 1),
        "total_descent_m": round(total_descent, 1),
        "elevation_score": round(score, 3),
        "missing_points": len(elevations) - len(clean_elevations),
    }


if __name__ == "__main__":
    # Self-test using fake elevation data (no internet needed) to
    # confirm the scoring maths is correct. The real fetch_elevations()
    # call is tested separately once running with real internet access.

    print("--- Test 1: flat route ---")
    flat = [50, 50, 51, 50, 50, 49, 50]
    result = score_elevation(flat)
    print(result)
    assert result["elevation_score"] < 0.1, "Flat route should score near 0"
    print("PASSED\n")

    print("--- Test 2: hilly route ---")
    hilly = [50, 80, 120, 90, 150, 100, 180, 130]
    result = score_elevation(hilly)
    print(result)
    assert result["elevation_score"] > 0.3, "Hilly route should score noticeably higher"
    print("PASSED\n")

    print("--- Test 2b (NEW): gradient-based scoring distinguishes steep-short from gentle-long ---")
    # Same 200m total climb, two very different real-world contexts:
    # a SHORT 300m climb (a serious ~67% average grade, more dramatic
    # than almost any real UK road) vs a LONG 6km climb (a gentle 3.3%
    # average grade, unremarkable hill country). The OLD absolute
    # method scored both as equally maximal (1.0) -- unable to tell a
    # mountain pass from gentle countryside. Gradient-based scoring
    # must tell them apart.
    steep_short = [0, 200]  # represents a climb over a short span
    gentle_long = [0, 200]  # same RAW elevation list -- the distance is what differs
    old_steep = score_elevation(steep_short)  # no total_distance_m -- old behaviour
    old_gentle = score_elevation(gentle_long)
    print(f"OLD method, no distance info: steep-short={old_steep['elevation_score']}, "
          f"gentle-long={old_gentle['elevation_score']} (identical -- can't tell them apart)")
    assert old_steep["elevation_score"] == old_gentle["elevation_score"] == 1.0, (
        "Confirming the OLD method's blind spot: both score maximally regardless of distance"
    )

    new_steep = score_elevation(steep_short, total_distance_m=300)
    new_gentle = score_elevation(gentle_long, total_distance_m=6000)
    print(f"NEW gradient-based: steep-short (300m span)={new_steep['elevation_score']}, "
          f"gentle-long (6km span)={new_gentle['elevation_score']}")
    assert new_steep["elevation_score"] == 1.0, "A 67% average grade should max out the score"
    assert new_gentle["elevation_score"] < 0.5, (
        f"A gentle 3.3% average grade should score modestly, not maximally — "
        f"got {new_gentle['elevation_score']}"
    )
    assert new_steep["elevation_score"] > new_gentle["elevation_score"], (
        "The steep short climb must score higher than the gentle long one once "
        "distance is accounted for — this is the entire point of the fix"
    )
    print("PASSED — gradient-based scoring correctly distinguishes dramatic short climbs "
          "from gentle long ones, which the old method could never tell apart\n")

    print("--- Test 2c (NEW): omitting total_distance_m falls back to the old behaviour exactly ---")
    fallback_result = score_elevation(hilly)
    explicit_old_style = score_elevation(hilly, total_distance_m=None)
    assert fallback_result == explicit_old_style
    print("PASSED — backward compatible: no total_distance_m means identical old behaviour\n")

    print("--- Test 3: missing data handled gracefully ---")
    with_gaps = [50, None, 80, None, None, 120]
    result = score_elevation(with_gaps)
    print(result)
    assert result["missing_points"] == 3
    print("PASSED — missing points correctly counted, no crash\n")

    print("--- Test 4 (NEW): subsample_points_by_distance picks roughly every target_spacing_m ---")
    # 51 points spaced 10m apart along a straight line = 500m total.
    # At target_spacing_m=100, should pick roughly every 10th point.
    straight_line = [(-1.0 + i * 0.0001, 51.0) for i in range(51)]  # ~11m apart at this latitude
    sampled = subsample_points_by_distance(straight_line, target_spacing_m=100)
    print(f"Sampled {len(sampled)} of {len(straight_line)} points: indices {[idx for idx, _ in sampled]}")
    assert sampled[0][0] == 0, "First point must always be included"
    assert sampled[-1][0] == len(straight_line) - 1, "Last point must always be included"
    assert len(sampled) < len(straight_line) / 2, (
        "Should meaningfully reduce point count for a long, evenly-spaced path"
    )
    print("PASSED — meaningfully fewer points, first/last always included\n")

    print("--- Test 5 (NEW): interpolate_elevations correctly fills a simple linear ramp ---")
    # 11 points; only sample indices 0 and 10 (elevations 100 and 200).
    # Built INCREMENTALLY (not index*scale, which would create one
    # discontinuous jump instead of genuinely gradual unevenness):
    # first 5 segments closely spaced (~7m each), last 5 segments much
    # further apart (~63m each) — so most of the TOTAL distance is in
    # the second half, even though it's the same number of points.
    # Every point in between should interpolate proportionally by
    # along-path DISTANCE, not by naive index position.
    uneven_points = [(-1.0, 51.0)]
    _lon = -1.0
    for i in range(10):
        _lon += 0.0001 if i < 5 else 0.0009
        uneven_points.append((_lon, 51.0))
    sampled_pairs = [(0, 100.0), (10, 200.0)]
    interpolated = interpolate_elevations(uneven_points, sampled_pairs)
    print(f"Interpolated: {[round(e, 1) if e is not None else None for e in interpolated]}")
    assert interpolated[0] == 100.0 and interpolated[10] == 200.0
    # Only 1/10th of the total distance is covered by the time we
    # reach index 5 (5 small segments out of 5 small + 5 much-larger
    # ones) — so index 5 should sit well below the naive index-based
    # midpoint (150), close to 110, not anywhere near 150.
    print(f"  index 5 interpolated to {round(interpolated[5], 1)} (naive index-midpoint would be 150)")
    assert interpolated[5] < 125, (
        f"Distance-aware interpolation should put index 5 well below the naive "
        f"index-based midpoint (150) given the uneven spacing, got {interpolated[5]}"
    )
    print("PASSED — interpolation is correctly distance-aware, not just index-based\n")

    print("--- Test 6 (NEW): interpolate_elevations leaves gaps as None across missing data ---")
    gap_pairs = [(0, 100.0), (5, None), (10, 200.0)]
    gap_result = interpolate_elevations(uneven_points, gap_pairs)
    print(f"With a None sample in the middle: {gap_result}")
    assert gap_result[5] is None, "An explicitly-sampled None must stay None, not be overwritten"
    assert all(gap_result[i] is None for i in range(1, 5)), (
        "Points between a real sample and a None sample can't be interpolated -- "
        "must stay None, not silently fabricate a value"
    )
    assert gap_result[0] == 100.0 and gap_result[10] == 200.0
    print("PASSED — missing data correctly leaves a visible gap, never fabricates a number\n")

    print("--- Test 7 (NEW): fetch_elevations_along_path matches direct fetch for a real cache hit case ---")
    # End-to-end sanity check using the real cache (mocked network):
    # fetching via subsample+interpolate for a path where ALL points
    # happen to already be cached (so interpolation isn't even
    # exercised) should return exactly the cached values.
    ELEVATION_CACHE_PATH = "/tmp/test_elevation_cache_selftest.json"
    if os.path.exists(ELEVATION_CACHE_PATH):
        os.remove(ELEVATION_CACHE_PATH)

    class _FakeResponse:
        def __init__(self, locations_param):
            self.status_code = 200
            self._locations_param = locations_param
        def raise_for_status(self):
            pass
        def json(self):
            locs = self._locations_param.split("|")
            results = []
            for loc in locs:
                lat, lon = map(float, loc.split(","))
                results.append({"elevation": round((lat + lon) * 10, 1)})
            return {"results": results}

    def _fake_post(url, data, timeout):
        return _FakeResponse(data["locations"])

    requests.post = _fake_post

    path_test_points = [(-1.0 + i * 0.0001, 51.0) for i in range(30)]
    along_path_result = fetch_elevations_along_path(path_test_points, target_spacing_m=100)
    direct_result = fetch_elevations(path_test_points)  # now fully cached, no real network needed
    print(f"Along-path (subsampled+interpolated): {[round(e, 1) for e in along_path_result[:5]]}...")
    print(f"Direct fetch (all cached now):        {[round(e, 1) for e in direct_result[:5]]}...")
    # Allow tiny floating-point interpolation differences at non-
    # sampled points, but sampled points themselves should match exactly.
    for i in range(len(path_test_points)):
        assert abs(along_path_result[i] - direct_result[i]) < 0.5, (
            f"Point {i}: along-path gave {along_path_result[i]}, direct gave {direct_result[i]} "
            f"-- should be very close for a near-linear synthetic elevation function"
        )
    print("PASSED — subsample+interpolate produces results consistent with direct fetching\n")
    os.remove(ELEVATION_CACHE_PATH)

    print("If hilly route scores meaningfully higher than flat, logic is sound.")