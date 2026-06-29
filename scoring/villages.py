"""
Byway — Notable Village/Town Matching (Conservation Areas, official data)
============================================================================

What this does, in plain terms:
For a route's bounding box, queries Historic England's official,
nationwide Conservation Areas dataset to find any legally-designated
conservation areas the route passes near — a real, structured proxy
for "this village/town's character is officially recognised as worth
protecting," which correlates strongly with the kind of charming,
well-preserved places this category is trying to capture.

WHY THIS REPLACED THE HARDCODED VILLAGE LIST: an earlier version used
a manually curated list of ~25 villages compiled from a handful of
media "prettiest villages" articles. That approach has a real,
unavoidable problem: it only ever covers places that happened to get
written about by national media, and would never scale to "find all
nice villages along any route in England" — exactly what's needed
for this to become a genuine user preference rather than a parlour
trick on a few pre-picked test routes. Conservation Areas are an
official legal designation, covering roughly 10,000 places across
England, made by local councils specifically because of "special
architectural or historic interest" — a structured, comprehensive,
nationwide signal rather than a sampled list.

DATA SOURCE: Historic England's open ArcGIS FeatureServer.
https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services/Conservation_Areas/FeatureServer
Free, no API key required, standard ArcGIS REST query interface.

HONEST LIMITATIONS (Historic England's own description, not ours):
- This is described as "indicative, not definitive" data.
- Coverage is INCOMPLETE — some local authorities haven't submitted
  data and are marked "No data available for publication by HE." A
  village with no conservation area shown here might still have one
  that simply isn't in this dataset yet, or might genuinely not have
  one. Either way, treat "no match" as "no signal," not as "this
  place lacks charm."
- A conservation area can cover part of a town (e.g. just the historic
  core), not the whole administrative area — a route passing through
  the modern outskirts of a town with a historic centre conservation
  area might not actually pass near the protected part.

GEOMETRY FORMAT FIX (this version — after the Bibury self-test below
returned 0 results, a place definitely in this dataset, confirming a
real bug rather than a coverage gap):

Fetching this service's own metadata (https://services-eu1.arcgis.com/
ZOdPfBS3aqqDYPUQ/arcgis/rest/services/Conservation_Areas/FeatureServer)
shows its native Spatial Reference is 27700 (British National Grid),
not 4326. The previous version sent the bbox as a plain comma string
("xmin,ymin,xmax,ymax") with inSR=4326 telling the server to
reproject it — the standard approach, but a known weak point for
services whose native SR isn't already 4326: some hosted ArcGIS
layers don't reliably apply that reprojection on the plain-string
geometry format, and instead read the numbers as if they were ALREADY
in the layer's own SR (metres). A WGS84 bbox like
"-1.85,51.74,-1.78,51.78" misread as raw OSGB36 metres is a few
centimetres wide, sitting near grid reference (0,0) off the Cornish
coast — nowhere near Bibury, Upperton, or anywhere else this function
gets called. That would produce exactly the symptom seen: a clean,
error-free response with 0 features, every time, anywhere in England.

Fix: send geometry as an explicit JSON envelope object with the
spatial reference embedded directly inside it, rather than as a plain
string relying on the separate inSR param — the more robust of the
two formats ArcGIS REST supports, and the one less likely to depend
on a specific service's reprojection behaviour.

NOT YET CONFIRMED LIVE: this is a strong, well-evidenced hypothesis
(the service's SR really is 27700, confirmed via its own metadata
endpoint) — but it could not be tested against the live QUERY
endpoint while writing it, only the metadata endpoint. Re-run this
file's self-test (Bibury) first, before trusting it. Debug printing
below shows the exact geometry sent and the raw feature count back,
so if this still returns 0, we'll have real evidence for the next
hypothesis instead of guessing again.

Network note: needs real internet access. Will only work when run
somewhere with a real connection (e.g. GitHub Codespaces), not inside
Claude's sandboxed tool environment.
"""

import json
import time
import requests

from scoring.arcgis_utils import paginated_arcgis_query


CONSERVATION_AREAS_URL = (
    "https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services/"
    "Conservation_Areas/FeatureServer/0/query"
)
USER_AGENT = "BywayApp-DevelopmentPrototype/0.1 (research prototype)"
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3


def _compute_village_weight(lpa):
    """
    NEW — LPA-based weighting: a Conservation Area administered by a
    NATIONAL PARK AUTHORITY (rather than an ordinary district/borough
    council) sits inside officially-designated beautiful LANDSCAPE,
    not just a historically charming village — a real, free signal
    already present in this exact data (the "lpa" field), simply
    never used for ranking before. Confirmed live on a real result:
    Lurgashall's own LPA reads "South Downs National Park Authority".

    Returns 1.0 for National Park/AONB-administered areas, 0.6 for
    everything else (still a real, officially-designated Conservation
    Area, just without the added landscape-designation signal).
    """
    lpa_lower = (lpa or "").lower()
    is_protected_landscape = any(
        term in lpa_lower for term in ("national park", "aonb", "area of outstanding natural beauty")
    )
    return 1.0 if is_protected_landscape else 0.6


def fetch_conservation_areas_in_bbox(bbox, debug=False):
    """
    Query Historic England's Conservation Areas dataset for any
    designated areas whose centroid falls within the given route
    bounding box (same {min_lat, max_lat, min_lon, max_lon} shape
    used elsewhere in this codebase, e.g. proximity.py).

    Returns a list of {"name": str, "lpa": str, "lat": float, "lon": float,
    "weight": float} dicts. Returns an empty list (not an error) if
    the service has no data for this area — a real, expected outcome
    given known incomplete coverage, not a failure case.

    CHANGED — now PAGINATES (scoring.arcgis_utils.paginated_arcgis_
    query) rather than capping at a single 200-record page. A large
    National Park or a region with many designated Conservation Areas
    could plausibly exceed 200 in one bbox — the same "unordered cap
    risks silently excluding what matters" risk found and fixed for
    OS land cover's building density and historic_england.py's NHLE
    query, generalized here too rather than left as a narrow fix.

    debug: if True, print the exact geometry sent and a summary of the
    raw response (feature count per page, any error/
    exceededTransferLimit flags) — use this the first few times this
    runs after a query change, to get real evidence rather than just
    a final 0-or-not count.
    """
    geometry = json.dumps({
        "xmin": bbox["min_lon"],
        "ymin": bbox["min_lat"],
        "xmax": bbox["max_lon"],
        "ymax": bbox["max_lat"],
        "spatialReference": {"wkid": 4326},
    })

    params = {
        "where": "1=1",
        "outFields": "Name,LPA",
        "geometry": geometry,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326",  # ask for results back in lat/lon too
        "f": "json",
    }

    if debug:
        print(f"    [DEBUG] Conservation Areas query geometry: {geometry}")

    features = paginated_arcgis_query(
        CONSERVATION_AREAS_URL, params, user_agent=USER_AGENT, delay_seconds=REQUEST_DELAY_SECONDS, debug=debug,
    )

    if debug and features:
        print(f"    [DEBUG] Total across all pages: {len(features)} feature(s)")
        print(f"    [DEBUG] First raw feature attributes: {features[0].get('attributes')}")

    results = []
    for feature in features:
        attrs = feature.get("attributes", {})
        # FIX: the server's real attribute key is "NAME" (all
        # caps) — confirmed directly from a debug run's raw
        # feature attributes ({'NAME': 'Ablington', 'LPA': ...}).
        # The original code checked "Name" (title case), which
        # never matched, so every real result was silently
        # skipped by the `if not name` check below — this was the
        # actual cause of "found features but reported 0 areas,"
        # separate from the earlier geometry/SR fix. Checking both
        # casings keeps this safe if the server's casing ever
        # changes or differs across environments.
        name = attrs.get("NAME") or attrs.get("Name")
        if not name:
            continue

        # The query returns polygon geometry; we use its centroid
        # (an approximate average of ring points) as a single
        # representative point for distance-based matching against
        # a route, the same pattern used for other proximity
        # features elsewhere in this codebase.
        geometry_result = feature.get("geometry", {})
        rings = geometry_result.get("rings", [])
        if not rings or not rings[0]:
            continue
        ring_points = rings[0]
        centroid_lon = sum(p[0] for p in ring_points) / len(ring_points)
        centroid_lat = sum(p[1] for p in ring_points) / len(ring_points)

        # See _compute_village_weight's docstring for why LPA is
        # used as a real, free significance signal here.
        lpa = attrs.get("LPA", "")
        weight = _compute_village_weight(lpa)

        results.append({
            "name": name,
            "lpa": lpa,
            "lat": centroid_lat,
            "lon": centroid_lon,
            "weight": weight,
        })

    return results


if __name__ == "__main__":
    print("--- Offline test: LPA-based village weighting (no network needed) ---")
    # Real LPA string observed directly in a live run (Lurgashall) --
    # confirms the matching logic against actual data, not a guess.
    assert _compute_village_weight("South Downs National Park Authority") == 1.0
    assert _compute_village_weight("Chichester District Council") == 0.6
    assert _compute_village_weight("Cotswolds AONB") == 1.0
    assert _compute_village_weight("") == 0.6
    assert _compute_village_weight(None) == 0.6
    print("PASSED — National Park/AONB authorities correctly weighted higher than ordinary councils\n")

    # This self-test requires real internet access (Claude's sandbox
    # doesn't have it) — run in Codespaces to verify against a real
    # known area. Bibury, Gloucestershire, is a well-documented
    # Cotswolds conservation area, used here as a sanity check that
    # the query actually returns real, sensible data.
    print("--- Live test: querying around Bibury, Gloucestershire ---")
    bibury_bbox = {"min_lat": 51.74, "max_lat": 51.78, "min_lon": -1.85, "max_lon": -1.78}
    results = fetch_conservation_areas_in_bbox(bibury_bbox, debug=True)
    print(f"Found {len(results)} conservation area(s):")
    for r in results:
        print(f"  {r['name']} (LPA: {r['lpa']}) at ({r['lat']:.4f}, {r['lon']:.4f})")
    if results:
        print("\nPASSED — got real data back")
    else:
        print("\nNo results — could mean no internet access here, or genuinely no data for this area.")
        print("Check the [DEBUG] lines above: if 'Raw response: 0 feature(s)' with no error,")
        print("the query reached the server fine but matched nothing — re-check the geometry")
        print("printed above against Bibury's real location (~51.76, -1.83) by eye.")
