"""
Byway — CORINE Land Cover Composition Scoring
================================================

What this does, in plain terms:
Computes what TYPE of landscape a road is actually running through —
Urban, Agriculture, Forest & natural areas, Wetland, or Water — using
the CORINE Land Cover dataset (the EU's standard, government-produced
land classification, derived from satellite imagery, freely available
under open Copernicus licensing).

WHY THIS EXISTS: real published research (Levering, Marcos & Tuia,
2021, ISPRS Journal — validated directly against ScenicOrNot, the
same dataset scoring/scenicornot.py uses) found that a SIMPLE LINEAR
model on just these 5 broad land-cover categories predicts real human
scenicness ratings almost as well as a full deep-learning model
analyzing actual satellite photographs (Kendall's τ 0.417 vs
0.456-0.457 — a 5-category linear model gets ~90% of the way to what
a CNN achieves). This is a genuinely different, AREA-based signal from
point-feature proximity (the old water/forest proximity checks, which
asked "is there a feature within 500m") — this asks "what does the
surrounding landscape actually consist of," which the research
suggests matters more.

DECISION (direct instruction): this REPLACES water/forest point-
proximity in the scenery formula rather than supplementing it —
land-cover composition is a more comprehensive, validated way of
asking the same underlying question ("is this road surrounded by
nice natural landscape, or built-up/farmed land"), so keeping both
would mostly double-count the same signal. Village (Conservation
Area) and ScenicOrNot proximity stay separate — they measure
genuinely different things (historic charm; direct human aesthetic
judgment) that land cover composition doesn't capture on its own.

KEY VALIDATED FINDINGS THIS MODULE'S WEIGHTS ARE BASED ON (see
Decisions Log for full detail):
- Forest/natural areas: strongly positive (well-established).
- Human/built-up (Urban) presence: measurably NEGATIVE, confirmed
  across multiple independently-cited studies, not just one paper.
- Water: positive, though the source research found coastline scores
  more strongly than rivers/estuaries — CORINE's broad-category
  resolution doesn't distinguish these here, so this is currently one
  single, slightly more moderate positive weight, not split by water
  sub-type. Flagged as a known simplification, not finished science.
- Agriculture: mild positive/near-neutral — the source research found
  genuinely mixed effects depending on context ("strong mixing
  between modes"), not a clean signal either way.
- Wetland: mild positive — also genuinely context-dependent in the
  source research (positive in highland bogs/lochs, negative in some
  coastal contexts); simplified to one weight here.

DATA SOURCE: CORINE Land Cover 2018, via the EEA's live ArcGIS REST
identify service (image.discomap.eea.europa.eu/arcgis/rest/services/
Corine/CLC2018_WM/MapServer) — free, open access under Copernicus
licensing (attribution required, no commercial-use blocker).

PERFORMANCE / ARCHITECTURE NOTE: like elevation, this needs a network
call PER SAMPLE POINT — there's no confirmed bulk-query endpoint for
this specific service (unlike Conservation Areas/ScenicOrNot, which
allow one bulk fetch per bounding box). To avoid reintroducing
elevation's original whole-graph bottleneck, land cover is computed in
PHASE 2 ONLY (see 07_score_graph_enjoyment.py's refine_scores_with_
elevation, which now also fetches land cover for the same bounded set
of candidate-route ways, reusing the SAME sampled points already
fetched for elevation) — never for the whole graph in phase 1. A local
cache (.landcover_cache.json, same pattern as elevation) avoids
re-fetching points across runs.

HONEST CAVEAT, UNVERIFIED UNTIL RUN AGAINST THE LIVE SERVICE: this is
this session's first real-network use of this specific ArcGIS
endpoint. CORINE's own CODE NUMBERING (1xx=Artificial/Urban,
2xx=Agricultural, 3xx=Forest/semi-natural, 4xx=Wetlands, 5xx=Water) is
stable, official, and documented — that part is solid. What ISN'T
independently confirmed is exactly which field name this particular
ArcGIS MapServer's identify response uses to expose that code —
_extract_corine_code() tries several plausible field names
defensively, and returns None (skip this point, not a crash or a
silent misattribution) if none match. Every other new external data
source integrated this project has needed at least one real-run
correction after first contact with the live service (Conservation
Areas needed two); expect the same honest possibility here, and check
real output carefully on the first live run rather than assuming this
is already exactly right.
"""

import os
import json
import time
import requests


CORINE_IDENTIFY_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/identify"
CORINE_DELAY_SECONDS = 0.2
CORINE_MAX_RETRIES = 2
CORINE_TIMEOUT_SECONDS = 8
# TIGHTENED (real-route testing found a 707-SECOND total runtime on a
# run where elevation's own pre-existing cache-hit logging showed
# fine, but land cover had NO equivalent logging at all -- meaning the
# actual cause (a handful of slow/failing fresh fetches? a hanging
# connection?) was genuinely invisible, not just slow). Reduced
# timeout (15s -> 8s) and retries (3 -> 2) bound the worst case much
# more tightly: at most ~2*8+2*0.4=~17s lost to a single failing
# point, not whatever the old settings allowed to compound.

LOCAL_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".landcover_cache.json"
)

LANDCOVER_CLASS_WEIGHTS = {
    "forest_natural": 1.0,
    "water": 0.6,
    "wetland": 0.5,
    "agriculture": 0.3,
    "urban": -0.5,
}


import math


def _bearing_radians(lat1, lon1, lat2, lon2):
    """Initial bearing (radians) from point 1 to point 2, standard formula."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return math.atan2(y, x)


def _offset_point(lat, lon, bearing_radians, distance_m):
    """Project a point at distance_m metres, given bearing (radians), from (lat, lon)."""
    R = 6371000
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    delta = distance_m / R
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(bearing_radians))
    lambda2 = lambda1 + math.atan2(
        math.sin(bearing_radians) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(lambda2), math.degrees(phi2)  # (lon, lat)


def corridor_sample_points(points, offsets_m=(200, -200)):
    """
    NEW — for each point along a road, also generates points OFFSET
    PERPENDICULAR to the road's local bearing, approximating a
    CORRIDOR/PATCH around the road rather than sampling only its own
    centerline.

    TIGHTENED (real-route testing): the original version used 4
    offsets (150/-150/350/-350), a 5x point-count multiplier. A real
    run needed 4,309 land-cover points (vs ~680 before corridor
    sampling existed) and took 1,647 seconds — almost entirely new,
    never-cached corridor coordinates, at CORINE's ~0.45s/call live
    latency. Reduced to 2 offsets (a single ±200m distance) — a 3x
    multiplier instead of 5x — as an immediate mitigation. This is a
    real tradeoff (less corridor width sampled, a small step back from
    full methodology fidelity) accepted deliberately while CORINE
    remains the live data source. The DURABLE fix is the OS OpenMap
    Local migration already in progress: once land cover is a local
    GeoPackage lookup instead of a live per-point API call, sampling
    density stops being a speed tradeoff at all — see scoring/
    os_landcover.py.

    WHY THIS EXISTS: the actual validated research (Levering, Marcos &
    Tuia, 2021) measured land-cover COMPOSITION WITHIN A FIXED PATCH
    AREA surrounding each point (~1.6km square, hundreds of pixels in
    every direction) — not "what's literally under this single point."
    Sampling only the road's own centerline asks a different, narrower
    question: "what does the road itself sit on," not "what does the
    landscape around it consist of." A lane running directly beside a
    forest edge could sample as 100% agriculture if the centerline
    happens to sit right on the field-side boundary, never checking
    either side. This is the direct fix for that methodology gap,
    flagged honestly in conversation rather than assumed fine.

    For each input point, bearing is estimated from its neighbours
    (or the single available neighbour at the start/end of a short
    list), and offset points are generated at each distance in
    offsets_m, perpendicular to that bearing (i.e. bearing + 90°).
    Positive offsets are conventionally "right" of travel direction,
    negative "left" — the actual side doesn't matter for composition
    purposes, only that both sides get sampled.

    Returns a list of (lon, lat) points — the ORIGINAL centerline
    points PLUS all generated offset points, ready to pass to
    fetch_landcover_classes/an equivalent classifier.
    """
    if len(points) < 2:
        return list(points)

    corridor_points = list(points)
    for i, (lon, lat) in enumerate(points):
        if i == 0:
            ref_lon, ref_lat = points[i + 1]
            bearing = _bearing_radians(lat, lon, ref_lat, ref_lon)
        else:
            prev_lon, prev_lat = points[i - 1]
            bearing = _bearing_radians(prev_lat, prev_lon, lat, lon)

        perpendicular_bearing = bearing + math.pi / 2
        for offset_m in offsets_m:
            corridor_points.append(_offset_point(lat, lon, perpendicular_bearing, offset_m))

    return corridor_points


def _code_to_category(code):
    """
    Maps a CORINE numeric code to one of our 5 broad categories, using
    CORINE's own stable, officially-documented numbering convention
    (1xx/2xx/3xx/4xx/5xx) — see module docstring for what is and isn't
    independently verified here.
    """
    try:
        code = int(float(code))
    except (TypeError, ValueError):
        return None
    if 100 <= code < 200:
        return "urban"
    elif 200 <= code < 300:
        return "agriculture"
    elif 300 <= code < 400:
        return "forest_natural"
    elif 400 <= code < 500:
        return "wetland"
    elif 500 <= code < 600:
        return "water"
    return None


def _load_cache():
    if os.path.exists(LOCAL_CACHE_PATH):
        try:
            with open(LOCAL_CACHE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache):
    try:
        with open(LOCAL_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass


def _cache_key(lon, lat):
    return f"{round(lat, 5)},{round(lon, 5)}"


def _extract_corine_code(identify_response):
    """
    Defensively pulls a CORINE numeric code out of an ArcGIS identify
    response, trying the field names most likely to be used for this
    kind of raster service, in order. Returns None (not an exception)
    if nothing recognizable is found — the caller skips that point
    rather than crash or silently misattribute it.
    """
    results = identify_response.get("results", [])
    if not results:
        return None
    attributes = results[0].get("attributes", {})
    for field_name in ("Pixel Value", "GRIDCODE", "CODE_18", "Value", "value"):
        if field_name in attributes:
            return attributes[field_name]
    # Some raster identify responses put the value directly at the
    # top level rather than inside "attributes".
    return results[0].get("value")


def _query_corine_identify(lat, lon):
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "visible",
        "tolerance": "1",
        "mapExtent": f"{lon - 0.01},{lat - 0.01},{lon + 0.01},{lat + 0.01}",
        "imageDisplay": "400,400,96",
        "returnGeometry": "false",
        "f": "json",
    }
    for attempt in range(CORINE_MAX_RETRIES):
        try:
            response = requests.get(CORINE_IDENTIFY_URL, params=params, timeout=CORINE_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            wait_time = CORINE_DELAY_SECONDS * (attempt + 2)
            print(f"  (CORINE identify call failed ({e}), waiting {wait_time}s and retrying...)")
            time.sleep(wait_time)
    return None


def _fetch_landcover_classes_corine(points, verbose=True):
    """
    CORINE-backed implementation (live ArcGIS identify() calls, one
    per uncached point) — kept as the FALLBACK when OS OpenMap Local
    data isn't available (see fetch_landcover_classes below, the
    actual active entry point everything else calls).

    Given a list of (lon, lat) points (already subsampled by the
    caller — see 07_score_graph_enjoyment.py, which now does its OWN,
    DELIBERATELY SPARSER subsampling pass for land cover than for
    elevation, since CORINE's own minimum mapping unit is ~25
    hectares — sampling every 100m, as elevation does, is finer than
    the underlying data can usefully resolve anyway), fetches each
    point's CORINE land-cover category. Mirrors scoring.elevation.
    fetch_elevations' shape exactly, for the same reuse reason.

    Returns {(lon, lat): category_string_or_None}. Cached locally
    (.landcover_cache.json) — repeat runs over previously-fetched
    points cost zero network calls.
    """
    import time as _time
    t_start = _time.time()

    cache = _load_cache()
    cache_dirty = False
    results = {}
    cache_hits = 0
    fresh_fetches = 0
    fresh_fetch_failures = 0

    for lon, lat in points:
        key = _cache_key(lon, lat)
        if key in cache:
            category = cache[key]
            cache_hits += 1
        else:
            time.sleep(CORINE_DELAY_SECONDS)
            response = _query_corine_identify(lat, lon)
            category = None
            if response is not None:
                code = _extract_corine_code(response)
                category = _code_to_category(code)
            else:
                fresh_fetch_failures += 1
            cache[key] = category
            cache_dirty = True
            fresh_fetches += 1
        results[(lon, lat)] = category

    if verbose:
        elapsed = round(_time.time() - t_start, 1)
        print(f"  Land cover (CORINE fallback): {cache_hits} cached, {fresh_fetches} fetched fresh "
              f"({fresh_fetch_failures} failed after retries) — {elapsed}s total.")

    if cache_dirty:
        _save_cache(cache)

    return results


# NEW — path to a downloaded OS OpenMap Local GeoPackage tile. If this
# file exists, it becomes the ACTIVE land-cover backend (local lookup,
# no network at all) instead of CORINE's live per-point API. See
# scoring/os_landcover.py for the full story — CORINE's live identify()
# endpoint was the source of two separate real-world performance
# problems this session (707s, then 1647s after corridor sampling was
# added) before being fixed down to a more reasonable cost; the OS
# migration removes the live-network dependency entirely, regardless
# of how densely we sample.
OS_GEOPACKAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "opmplc_gb.gpkg"
)


def fetch_landcover_classes(points, verbose=True):
    """
    THE ACTIVE ENTRY POINT — everything else in this codebase
    (07_score_graph_enjoyment.py, scoring/os_landcover.py's own test
    script) calls this name specifically, not the CORINE or OS
    implementations directly, so the backend can be swapped here in
    ONE place.

    Uses OS OpenMap Local (scoring/os_landcover.py's classify_points_os
    — local GeoPackage lookup, confirmed working on real test points:
    Guildford correctly classified urban, Bewl Water correctly
    classified water) if OS_GEOPACKAGE_PATH exists; falls back to
    CORINE's live API (_fetch_landcover_classes_corine) if it doesn't
    — e.g. in an environment without the file downloaded yet.

    Returns {(lon, lat): category_string_or_None} — same shape
    regardless of which backend actually served the request.
    """
    if os.path.exists(OS_GEOPACKAGE_PATH):
        from scoring.os_landcover import classify_points_os
        if verbose:
            print(f"  Land cover: using local OS OpenMap Local data "
                  f"({len(points)} points, no network) — {OS_GEOPACKAGE_PATH}")
        return classify_points_os(points, OS_GEOPACKAGE_PATH, verbose=verbose)

    if verbose:
        print(f"  Land cover: no OS OpenMap Local file found at {OS_GEOPACKAGE_PATH} — "
              f"falling back to CORINE's live API.")
    return _fetch_landcover_classes_corine(points, verbose=verbose)


def composition_from_classes(point_categories):
    """
    Given {point: category_or_None} for ONE way's sampled points
    (a subset of fetch_landcover_classes' return value), tallies the
    FRACTION falling into each of the 5 broad categories.

    Returns {"urban": 0-1, "agriculture": 0-1, "forest_natural": 0-1,
    "wetland": 0-1, "water": 0-1} (fractions sum to ~1.0 across
    successfully-classified points). All-zero (not an error) if
    nothing classified — treated downstream as "no signal," not
    "definitely no nice landscape here."
    """
    counts = {k: 0 for k in LANDCOVER_CLASS_WEIGHTS}
    classified_count = 0

    for category in point_categories.values():
        if category in counts:
            counts[category] += 1
            classified_count += 1

    if classified_count == 0:
        return {k: 0.0 for k in LANDCOVER_CLASS_WEIGHTS}

    return {k: round(v / classified_count, 3) for k, v in counts.items()}


def score_landcover(composition):
    """
    Combines a composition dict (from composition_from_classes) into a
    single 0-1 score, using LANDCOVER_CLASS_WEIGHTS (see module
    docstring for the research these weights are based on).

    Raw weighted sum ranges roughly -0.5 (100% urban) to 1.0 (100%
    forest/natural) — normalised into 0-1 via (raw + 0.5) / 1.5,
    clipped, so it combines cleanly with this project's other 0-1
    scores.
    """
    raw = sum(composition.get(k, 0.0) * w for k, w in LANDCOVER_CLASS_WEIGHTS.items())
    normalised = (raw + 0.5) / 1.5
    return {
        "landcover_score": round(min(max(normalised, 0.0), 1.0), 3),
        "composition": composition,
    }


def generate_grid_cells(bbox, cell_size_km=1.0):
    """
    NEW — divides a bbox into a grid of roughly cell_size_km x
    cell_size_km cells, returning each cell's centroid as a (lon, lat)
    point.

    1km is a deliberate default, not arbitrary: it matches ScenicOrNot's
    own native grid resolution (a nice conceptual consistency between
    the two area-wide signals) and is comfortably coarser than
    CORINE's ~25-hectare minimum mapping unit, so this isn't over-
    sampling relative to what the underlying data can actually resolve.
    """
    avg_lat = sum([bbox["min_lat"], bbox["max_lat"]]) / 2
    lat_step_deg = cell_size_km / 111.0  # ~111km per degree of latitude, everywhere
    lon_step_deg = cell_size_km / (111.0 * math.cos(math.radians(avg_lat)))
    # (longitude degree size shrinks toward the poles — cos(latitude)
    # accounts for that so cells stay roughly square at any latitude)

    centroids = []
    lat = bbox["min_lat"] + lat_step_deg / 2
    while lat < bbox["max_lat"]:
        lon = bbox["min_lon"] + lon_step_deg / 2
        while lon < bbox["max_lon"]:
            centroids.append((lon, lat))
            lon += lon_step_deg
        lat += lat_step_deg
    return centroids


def generate_landcover_grid_targets(bbox, classify_fn, cell_size_km=1.0, top_n=20):
    """
    NEW — grids the bbox, classifies each cell's centroid via
    classify_fn (e.g. fetch_landcover_classes, or a future OS-
    GeoPackage-backed equivalent with the same signature — this
    function doesn't care which backend is underneath), scores each
    cell with score_landcover, ranks them, and returns the top_n as
    point-features: {"name", "lat", "lon", "weight"} — the SAME shape
    ScenicOrNot/Conservation Area entries already use, so these slot
    directly into generate_candidate_routes' target_features
    (04_pathfinding.py, source 4) without any new mechanism.

    WHY THIS EXISTS: ScenicOrNot, Conservation Areas, and historic
    sites are all naturally a finite list of specific PLACES — easy to
    rank and target the best of directly. Land cover is a continuous
    surface covering the whole bbox, with no pre-existing list of
    "best spots" to target — this function MAKES that list, so land
    cover can actively decide which roads get tried as candidates,
    not just describe whichever roads happen to already be generated
    by other means.

    Deliberately ONE point per cell (not corridor-sampled per cell) —
    grid cells are for identifying broadly-excellent AREAS to target,
    a coarser, cheaper pass than the fine-grained corridor sampling
    already applied once we know which specific roads to examine (see
    07_score_graph_enjoyment.py's refine_scores_with_elevation).
    """
    centroids = generate_grid_cells(bbox, cell_size_km=cell_size_km)
    point_classes = classify_fn(centroids)

    cell_scores = []
    for lon, lat in centroids:
        category = point_classes.get((lon, lat))
        composition = composition_from_classes({(lon, lat): category})
        result = score_landcover(composition)
        cell_scores.append({
            "name": f"Land cover cell ({to_ten_helper(result['landcover_score'])}/10, "
                    f"{category or 'unclassified'})",
            "lat": lat, "lon": lon,
            "weight": result["landcover_score"],
        })

    cell_scores.sort(key=lambda c: -c["weight"])
    return cell_scores[:top_n]


def to_ten_helper(value):
    """Local display helper (0-1 -> 0-10, one decimal) — avoids importing
    07_score_graph_enjoyment.py just for this, which would be circular."""
    return round(value * 10, 1)


if __name__ == "__main__":
    print("--- Test 0 (NEW): corridor sampling generates genuinely offset points, not centerline duplicates ---")
    road_points = [(-0.700, 51.000), (-0.699, 51.001), (-0.698, 51.002)]
    corridor = corridor_sample_points(road_points, offsets_m=(150, -150))
    print(f"Original points: {len(road_points)}, corridor points (with offsets): {len(corridor)}")
    assert len(corridor) == len(road_points) * 3, (
        "Expected original points plus 2 offsets per point (150m and -150m)"
    )
    # Confirm offset points are genuinely displaced, not duplicates.
    offset_points = corridor[len(road_points):]
    for lon, lat in offset_points:
        min_dist = min(
            math.sqrt((lon - olon) ** 2 + (lat - olat) ** 2) for olon, olat in road_points
        )
        assert min_dist > 0.0005, f"Offset point {(lon, lat)} is suspiciously close to the centerline"
    print("PASSED — corridor points are genuinely displaced perpendicular to the road, "
          "approximating an area/patch rather than the centerline alone\n")

    # Self-test of the pure logic (code mapping, composition tally,
    # weighting/normalisation) — no network needed. The live identify
    # call itself is exercised for real the first time this runs
    # against an actual graph (see 07_score_graph_enjoyment.py).

    print("--- Test 1: CORINE code -> category mapping ---")
    assert _code_to_category(112) == "urban"          # e.g. discontinuous urban fabric
    assert _code_to_category(231) == "agriculture"     # e.g. pastures
    assert _code_to_category(311) == "forest_natural"  # e.g. broad-leaved forest
    assert _code_to_category(411) == "wetland"         # e.g. inland marshes
    assert _code_to_category(511) == "water"           # e.g. water courses
    assert _code_to_category(None) is None
    assert _code_to_category("not a number") is None
    print("PASSED — code ranges map to the correct broad category\n")

    print("--- Test 2: composition tally from classified points ---")
    fake_points = {
        (0.0, 51.0): "forest_natural", (0.001, 51.0): "forest_natural",
        (0.002, 51.0): "agriculture", (0.003, 51.0): None,  # unclassified point, correctly excluded
    }
    composition = composition_from_classes(fake_points)
    print(composition)
    assert composition["forest_natural"] == round(2 / 3, 3)
    assert composition["agriculture"] == round(1 / 3, 3)
    assert composition["urban"] == 0.0
    print("PASSED — unclassified points correctly excluded from the denominator, not counted as zero-category\n")

    print("--- Test 3: scoring correctly favours forest, penalises urban ---")
    forest_only = score_landcover({"forest_natural": 1.0, "urban": 0.0, "agriculture": 0.0, "wetland": 0.0, "water": 0.0})
    urban_only = score_landcover({"forest_natural": 0.0, "urban": 1.0, "agriculture": 0.0, "wetland": 0.0, "water": 0.0})
    print(f"100% forest: {forest_only['landcover_score']}, 100% urban: {urban_only['landcover_score']}")
    assert forest_only["landcover_score"] == 1.0
    assert urban_only["landcover_score"] == 0.0
    assert forest_only["landcover_score"] > urban_only["landcover_score"]
    print("PASSED — forest scores maximally, urban scores minimally, matching the validated research direction\n")

    print("All landcover module logic tests passed (network call itself untested here -- see live run).")
