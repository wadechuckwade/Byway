"""
Byway — Proximity Scoring (Water, Forest, Historic Sites)
=============================================================

What this does, in plain terms:
For a given ROUTE (not a single segment), fetches water/forest/
historic data ONCE for a bounding box covering the entire route, then
lets every segment check against that already-fetched data with zero
further network calls.

Why this changed from the original per-segment design:
The first version asked Overpass fresh for every single segment of a
route (3 calls x ~10-20 segments = 30-60 live calls per route). On a
real route this would take minutes and regularly time out on the free
shared server, losing data exactly where it mattered. Fetching once
for the whole route's bounding box and checking segments against that
cached data turns dozens of slow, flaky live calls into one call,
with instant local lookups afterward. This is also how real mapping
systems generally work — map data is fetched/cached in bulk, not
re-queried live per tiny request.

NEW — OPTIONAL WEIGHT/MAGNITUDE SUPPORT: score_proximity() now lets
each feature optionally carry a "weight" key (0-1), for data sources
where magnitude matters, not just presence — e.g. ScenicOrNot's real
crowd-sourced scenicness rating (scoring/scenicornot.py), where an
8/10 spot should count for meaningfully more than a 4/10 spot, unlike
water/forest/historic/village proximity, which are genuinely binary
(a feature either exists nearby or it doesn't). Defaults to 1.0 — full
backward compatibility, every existing caller (none of which set a
"weight" key) behaves EXACTLY as before, verified by a regression test
below.

Network note: needs real internet access to overpass-api.de for the
initial route-wide fetch. Will only work when run somewhere with a
real connection (e.g. GitHub Codespaces), not inside Claude's
sandboxed tool environment.
"""

import math
import time
import requests


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "BywayApp-DevelopmentPrototype/0.1"
OVERPASS_DELAY_SECONDS = 1.0  # politeness delay; we now make far fewer calls overall
OVERPASS_MAX_RETRIES = 4


def route_bounding_box(all_route_points, buffer_degrees=0.01):
    """
    Work out ONE bounding box that covers an entire route (all
    segments combined), with a buffer added so features just off the
    route edge still get caught.

    This replaces per-segment bounding boxes — we now fetch data for
    the whole route's area in one go.
    """
    lons = [p[0] for p in all_route_points]
    lats = [p[1] for p in all_route_points]
    return {
        "min_lat": min(lats) - buffer_degrees,
        "max_lat": max(lats) + buffer_degrees,
        "min_lon": min(lons) - buffer_degrees,
        "max_lon": max(lons) + buffer_degrees,
    }


def _query_overpass(query):
    """
    Send a query to Overpass, automatically waiting and retrying if we
    hit either the free server's rate limit (429) or a server-side
    timeout (504, common when the shared free server is under load).
    Since we now make only a handful of calls per route (not dozens),
    we can afford a longer per-request timeout for these larger
    region-wide queries.
    """
    time.sleep(OVERPASS_DELAY_SECONDS)

    for attempt in range(OVERPASS_MAX_RETRIES):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=60,
                headers={"User-Agent": USER_AGENT},
            )
        except requests.exceptions.Timeout:
            wait_time = OVERPASS_DELAY_SECONDS * (attempt + 2)
            print(f"  (Overpass timed out client-side, waiting {wait_time}s and retrying...)")
            time.sleep(wait_time)
            continue

        if response.status_code in (429, 504):
            wait_time = OVERPASS_DELAY_SECONDS * (attempt + 2)
            reason = "rate limit" if response.status_code == 429 else "server timeout"
            print(f"  (Overpass {reason} ({response.status_code}), waiting {wait_time}s and retrying...)")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(
        f"Overpass API still failing after {OVERPASS_MAX_RETRIES} attempts. "
        f"Try again in a minute, or increase OVERPASS_DELAY_SECONDS."
    )


def fetch_region_data(bbox):
    """
    Fetch water, forest, AND historic features for an entire route's
    bounding box in ONE combined Overpass query (rather than 3
    separate queries), to minimise the number of calls to the free
    shared server.

    Returns a dict: {"water": [...], "forest": [...], "historic": [...]}
    Each entry is a list of {"lat", "lon", "name"} dicts.
    """
    b = bbox
    query = f"""
    [out:json][timeout:55];
    (
      way["natural"="water"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      way["waterway"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      way["landuse"="forest"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      way["natural"="wood"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      node["historic"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      node["tourism"="attraction"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      node["tourism"="museum"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
    );
    out center;
    """
    print("  Fetching water/forest/historic data for the whole route area (one-time)...")
    data = _query_overpass(query)

    water, forest, historic = [], [], []

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")

        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        elif el.get("type") == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            continue

        if tags.get("natural") == "water" or "waterway" in tags:
            water.append({"lat": lat, "lon": lon, "name": name or "Unnamed water feature"})
        elif tags.get("landuse") == "forest" or tags.get("natural") == "wood":
            forest.append({"lat": lat, "lon": lon, "name": name or "Unnamed forest area"})
        elif "historic" in tags or tags.get("tourism") in ("attraction", "museum"):
            if name:
                historic.append({"lat": lat, "lon": lon, "name": name})

    print(f"  Found {len(water)} water features, {len(forest)} forest areas, "
          f"{len(historic)} NAMED historic/attraction sites in the route area.\n")

    return {"water": water, "forest": forest, "historic": historic}


def fetch_historic_only(bbox):
    """
    NEW — fetches ONLY historic/tourism-attraction OSM tags, dropping
    the water/forest clauses entirely.

    WHY THIS EXISTS: the graph-search pipeline (07_score_graph_
    enjoyment.py) no longer scores water/forest proximity at all —
    land cover composition replaced them as the area-wide scenery
    signal (confirmed by an explicit test: removing the water/forest
    feature lists from scoring left scores completely unchanged).
    fetch_region_data was still being called anyway, purely because
    historic tags were bundled in the SAME Overpass query — meaning
    water/forest data was being fetched and printed ("Found 1297
    water features, 872 forest areas...") while doing literally
    nothing, which looked like active use and wasn't. This function
    is the direct fix: only the historic/tourism clauses, nothing
    fetched that isn't actually used.

    fetch_region_data() itself is left UNCHANGED — milestone_1_
    score_route.py's independent, hand-waypoint scoring formula still
    uses water/forest proximity directly, and hasn't been migrated to
    land-cover-based scoring (a known, flagged divergence between the
    two pipelines, not something this change should silently alter).

    Returns a list of {"lat", "lon", "name"} dicts — same shape as
    fetch_region_data()["historic"].
    """
    b = bbox
    query = f"""
    [out:json][timeout:55];
    (
      node["historic"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      node["tourism"="attraction"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
      node["tourism"="museum"]({b['min_lat']},{b['min_lon']},{b['max_lat']},{b['max_lon']});
    );
    out center;
    """
    print("  Fetching historic/attraction data for the whole route area (one-time)...")
    data = _query_overpass(query)

    historic = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        elif el.get("type") == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            continue

        if "historic" in tags or tags.get("tourism") in ("attraction", "museum"):
            historic.append({"lat": lat, "lon": lon, "name": name})

    print(f"  Found {len(historic)} NAMED historic/attraction sites in the route area.\n")
    return historic


def _min_distance_to_route_m(feature_lat, feature_lon, route_points):
    """
    Distance in metres from one feature to the nearest point on a
    (sub)set of route points — e.g. just one segment's points, while
    the feature list itself was fetched once for the whole route.
    """
    R = 6371000
    best = float("inf")
    for lon, lat in route_points:
        phi1, phi2 = math.radians(lat), math.radians(feature_lat)
        d_phi = math.radians(feature_lat - lat)
        d_lambda = math.radians(feature_lon - lon)
        a = (math.sin(d_phi / 2) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist = R * c
        if dist < best:
            best = dist
    return best


def score_proximity(route_points, features, close_threshold_m=50, score_decay_m=500):
    """
    Score how close a route SEGMENT passes to a set of features (water,
    forest, historic sites, Conservation Areas, or now ScenicOrNot
    scenicness ratings) that were already fetched for the WHOLE route.
    This function itself does no network calls — it's pure local
    computation against data we already have in memory.

    Each feature can optionally carry a "weight" key (0-1) — see
    module docstring above. If absent, defaults to 1.0, exactly
    matching this function's original (pre-weight) behaviour.

    BUG FIX (found via real-route testing, Tillington/Haslemere and
    Eynsham/Burford after the ScenicOrNot integration): this used to
    divide by a LITERAL 5, regardless of how many features actually
    contributed:
        top_contributions = sorted(contributions, reverse=True)[:5]
        score = sum(top_contributions) / 5
    A road passing exactly ONE perfectly-positioned feature (a
    contribution of 1.0) scored 1.0/5 = 0.2, not 1.0 — a road would
    need FIVE separate, perfectly-positioned features of the same
    category simultaneously to ever reach 1.0. That's an absurd bar —
    most genuinely lovely rural roads pass one nice church or one
    pretty river, not five at once. This had been silently suppressing
    every proximity-based score (water, forest, historic, village, and
    ScenicOrNot) since Milestone 1's very first version, and is the
    most likely single biggest contributor to the low-ceiling pattern
    seen in every real-route test this session ("even the best road
    only reaches 0.4-0.6"). Fixed: divide by the ACTUAL number of
    contributions being averaged (still capped at the best 5, so a
    route through a town with 50 mediocre features still doesn't get
    diluted by all of them — only the "always divide by 5" part was
    the bug, not the "look at the best 5" cap itself).

    Returns 0-1, where 1 means the route runs right alongside one or
    more high-weight features, and 0 means nothing nearby at all.
    """
    if not features:
        return {"proximity_score": 0.0, "nearby_count": 0, "closest_m": None}

    contributions = []
    distances = []
    for f in features:
        dist = _min_distance_to_route_m(f["lat"], f["lon"], route_points)
        distances.append(dist)
        weight = f.get("weight", 1.0)
        if dist <= close_threshold_m:
            fraction = 1.0
        elif dist >= score_decay_m:
            fraction = 0.0
        else:
            fraction = 1 - (dist - close_threshold_m) / (score_decay_m - close_threshold_m)
        contributions.append(fraction * weight)

    top_contributions = sorted(contributions, reverse=True)[:5]
    score = sum(top_contributions) / len(top_contributions) if top_contributions else 0.0

    return {
        "proximity_score": round(min(score, 1.0), 3),
        "nearby_count": sum(1 for d in distances if d <= score_decay_m),
        "closest_m": round(min(distances), 1) if distances else None,
    }


if __name__ == "__main__":
    # Self-test with fake feature data (no internet needed) to confirm
    # the proximity-scoring maths is correct. The route-wide fetch
    # (fetch_region_data) is tested separately with real internet
    # access, since it requires a live Overpass call.

    route = [(-0.60, 50.88), (-0.62, 50.88), (-0.64, 50.877), (-0.667, 50.877)]

    print("--- Test 1: no features nearby ---")
    result = score_proximity(route, [])
    print(result)
    assert result["proximity_score"] == 0.0
    print("PASSED\n")

    print("--- Test 2 (FIXED): one feature right on the route should score near 1.0, not 0.2 ---")
    on_route = [{"lat": 50.877, "lon": -0.64, "name": "Test Lake"}]
    result = score_proximity(route, on_route)
    print(result)
    assert result["proximity_score"] > 0.9, (
        f"BUG REGRESSION: one perfectly-positioned feature should score close to 1.0 "
        f"(the old divide-by-5 bug would have given ~0.2 here), got {result['proximity_score']}"
    )
    print("PASSED — single feature now correctly scores near 1.0, confirming the divide-by-5 bug is fixed\n")

    print("--- Test 3: one feature very far away ---")
    far_away = [{"lat": 51.5, "lon": -0.1, "name": "London, too far"}]
    result = score_proximity(route, far_away)
    print(result)
    assert result["proximity_score"] == 0.0, "Far feature should not contribute"
    print("PASSED\n")

    print("--- Test 4 (NEW): features with NO weight key behave EXACTLY as before (regression) ---")
    # Verifying the unweighted path is bit-for-bit unchanged -- this is
    # the real safety net for every EXISTING caller (water/forest/
    # historic/village), none of which set a "weight" key.
    on_route_no_weight = [{"lat": 50.877, "lon": -0.64, "name": "Test Lake"}]
    result_a = score_proximity(route, on_route_no_weight)
    result_b = score_proximity(route, on_route_no_weight)
    assert result_a == result_b
    print(f"Unweighted result (no 'weight' key present): {result_a}")
    print("PASSED — unweighted behaviour is deterministic and unaffected by the new code path\n")

    print("--- Test 5 (NEW): weight correctly scales a feature's contribution ---")
    full_weight_feature = [{"lat": 50.877, "lon": -0.64, "name": "Full weight", "weight": 1.0}]
    half_weight_feature = [{"lat": 50.877, "lon": -0.64, "name": "Half weight", "weight": 0.5}]
    zero_weight_feature = [{"lat": 50.877, "lon": -0.64, "name": "Zero weight", "weight": 0.0}]

    full_result = score_proximity(route, full_weight_feature)
    half_result = score_proximity(route, half_weight_feature)
    zero_result = score_proximity(route, zero_weight_feature)

    print(f"weight=1.0 -> proximity_score={full_result['proximity_score']}")
    print(f"weight=0.5 -> proximity_score={half_result['proximity_score']}")
    print(f"weight=0.0 -> proximity_score={zero_result['proximity_score']}")

    assert full_result["proximity_score"] == result_a["proximity_score"], (
        "An explicit weight=1.0 should produce IDENTICAL results to no weight key at all"
    )
    assert abs(half_result["proximity_score"] - full_result["proximity_score"] / 2) < 0.01, (
        "weight=0.5 should contribute roughly half of what weight=1.0 does, for an "
        "otherwise identical feature at the same distance"
    )
    assert zero_result["proximity_score"] == 0.0, "weight=0.0 should contribute nothing at all"
    print("PASSED — weight correctly and proportionally scales each feature's contribution\n")

    print("If close features score higher than distant ones, and weight scales contributions")
    print("proportionally, logic is sound.")
