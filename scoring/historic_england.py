"""
Byway — Historic England National Heritage List (Listed Buildings, Scheduled Monuments)
============================================================================================

What this does, in plain terms:
Fetches OFFICIALLY GRADED historic landmarks — Listed Buildings and
Scheduled Monuments — from Historic England's own National Heritage
List for England (NHLE), giving a REAL MAGNITUDE for historic
significance (Grade I > Grade II* > Grade II) rather than the flat
"named historic site, yes/no" signal previously used (plain OSM
`historic=*`/`tourism=attraction` tags, no notion of how significant
any one of them actually is).

WHY THIS EXISTS: direct feedback — we want to target the BEST
villages/churches/landmarks in a bbox, not a random or first-found
selection. ScenicOrNot already gives a real magnitude (1-10 rating)
for scenery; historic sites had nothing equivalent until now. Grade is
that equivalent: an official, government-assigned ranking, not a
guess — Grade I is "of exceptional interest," Grade II* "particularly
important," Grade II "of special interest" (the most common tier,
~92% of all listings — still genuinely worth visiting, just not as
rare as I/II*).

DATA SOURCE: Historic England's NHLE, via the SAME ArcGIS organisation
(ZOdPfBS3aqqDYPUQ on services-eu1.arcgis.com) already used for
Conservation Areas (scoring/villages.py) — confirmed directly from the
live service's own root metadata, not guessed:
  https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/ArcGIS/rest/services/
  National_Heritage_List_for_England_NHLE_v02_VIEW/FeatureServer
Confirmed layers: 0 = Listed Building points, 6 = Scheduled Monuments
(others are Building Preservation Notices / Certificates of Immunity,
not used here). Free, Open Government Licence, no API key required.

HONEST CONFIDENCE NOTE — read before trusting field names: the SERVICE
URL and LAYER STRUCTURE above are confirmed directly from this exact
service's own live metadata. The "Grade" field name, however, was
confirmed from a DIFFERENT, third-party-hosted mirror of similar data
(a consultancy's ArcGIS instance), not this exact official service —
"Grade" is Historic England's own standard public terminology, so it's
a well-grounded guess, but not independently verified against THIS
service's response yet. outFields=* is used deliberately (request
everything, parse defensively) rather than naming specific fields, to
stay robust if the real field names differ slightly — same spirit as
Conservation Areas' NAME/Name casing fix and CORINE's multi-field-name
fallback. Scheduled Monuments aren't graded I/II*/II the way buildings
are (it's a binary designation, like Conservation Areas) — given a
flat weight reflecting "officially significant" without a finer tier.

Uses the EXACT same British National Grid (EPSG:27700) geometry fix
already proven for Conservation Areas: an explicit JSON envelope with
spatialReference embedded, not a plain comma-string bbox relying on a
separate inSR param.

Network note: needs real internet access. Will only work when run
somewhere with a real connection (e.g. GitHub Codespaces), not inside
Claude's sandboxed tool environment.
"""

import json
import time
import requests

from scoring.arcgis_utils import paginated_arcgis_query


NHLE_BASE_URL = (
    "https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/ArcGIS/rest/services/"
    "National_Heritage_List_for_England_NHLE_v02_VIEW/FeatureServer"
)
LISTED_BUILDINGS_LAYER = 0
SCHEDULED_MONUMENTS_LAYER = 6

USER_AGENT = "BywayApp-DevelopmentPrototype/0.1 (research prototype)"
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3

# Official Historic England grading — a real, government-assigned
# magnitude, the direct equivalent of ScenicOrNot's 1-10 rating for
# historic significance instead of scenery. Grade II is deliberately
# NOT zero — it's still a nationally-protected, genuinely significant
# site, just the most common tier, not the rarest.
LISTED_BUILDING_GRADE_WEIGHTS = {
    "i": 1.0,
    "ii*": 0.7,
    "ii": 0.4,
}
DEFAULT_GRADE_WEIGHT = 0.3  # unrecognised/missing grade value — still something, not nothing
SCHEDULED_MONUMENT_WEIGHT = 0.6  # flat — not graded I/II*/II the way buildings are


def _query_nhle_layer(layer_id, bbox, out_fields="*", debug=False):
    """
    Shared query logic for any layer on the NHLE FeatureServer — same
    geometry-envelope pattern as Conservation Areas' fetch_
    conservation_areas_in_bbox, factored out since both Listed
    Buildings and Scheduled Monuments query the same service, just a
    different layer ID.

    CHANGED — now PAGINATES (scoring.arcgis_utils.paginated_arcgis_
    query) rather than capping at a single 500-record page. A dense
    historic city (York, Bath, Oxford) could plausibly have more than
    500 listed buildings in one bbox, and the server's own return
    order has no guarantee of favouring the most significant (Grade I)
    ones — the exact same "unordered cap risks silently excluding what
    matters" risk already found and fixed for OS land cover's building
    density, generalized here rather than left as a narrow, one-off fix.
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
        "outFields": out_fields,
        "geometry": geometry,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326",
        "f": "json",
    }

    url = f"{NHLE_BASE_URL}/{layer_id}/query"

    if debug:
        print(f"    [DEBUG] NHLE layer {layer_id} query geometry: {geometry}")

    features = paginated_arcgis_query(url, params, user_agent=USER_AGENT, delay_seconds=REQUEST_DELAY_SECONDS, debug=debug)

    if debug and features:
        print(f"    [DEBUG] Total across all pages: {len(features)} feature(s)")
        print(f"    [DEBUG] First raw feature attributes: {features[0].get('attributes')}")

    return features


def _first_matching_field(attrs, candidate_names):
    """Defensively check several plausible field-name casings/variants, in order."""
    for name in candidate_names:
        if name in attrs and attrs[name] is not None:
            return attrs[name]
    return None


def fetch_listed_buildings_in_bbox(bbox, debug=False):
    """
    Query Historic England's Listed Buildings layer for any building
    within the given bbox, returning a real Grade-based weight per
    site rather than flat presence.

    Returns a list of {"name", "lat", "lon", "weight", "grade"} dicts
    — "weight" (0-1) is what feeds directly into score_proximity's
    weighted contribution, same shape as ScenicOrNot's entries.
    """
    features = _query_nhle_layer(LISTED_BUILDINGS_LAYER, bbox, debug=debug)

    results = []
    for feature in features:
        attrs = feature.get("attributes", {})
        name = _first_matching_field(attrs, ["Name", "NAME", "ListEntry", "ListEntryName"])
        grade_raw = _first_matching_field(attrs, ["Grade", "GRADE", "grade"])
        geom = feature.get("geometry", {})
        lon, lat = geom.get("x"), geom.get("y")
        if lon is None or lat is None:
            continue

        grade_key = str(grade_raw).strip().lower() if grade_raw else None
        weight = LISTED_BUILDING_GRADE_WEIGHTS.get(grade_key, DEFAULT_GRADE_WEIGHT)

        results.append({
            "name": name or "Listed Building",
            "lat": lat, "lon": lon,
            "weight": weight,
            "grade": grade_raw,
        })

    return results


def fetch_scheduled_monuments_in_bbox(bbox, debug=False):
    """
    Query Historic England's Scheduled Monuments layer. Not graded
    I/II*/II the way buildings are — a flat SCHEDULED_MONUMENT_WEIGHT
    reflects "officially significant" without a finer tier (see module
    docstring for why).

    Returns a list of {"name", "lat", "lon", "weight"} dicts, same
    shape as fetch_listed_buildings_in_bbox.
    """
    features = _query_nhle_layer(SCHEDULED_MONUMENTS_LAYER, bbox, debug=debug)

    results = []
    for feature in features:
        attrs = feature.get("attributes", {})
        name = _first_matching_field(attrs, ["Name", "NAME", "ListEntry", "ListEntryName"])
        geom = feature.get("geometry", {})

        # Scheduled Monuments may come back as points OR polygons,
        # unlike Listed Buildings (points only) — handle both, same
        # ring-centroid approach Conservation Areas already uses for
        # polygon geometry.
        if "x" in geom and "y" in geom:
            lon, lat = geom["x"], geom["y"]
        elif "rings" in geom and geom["rings"] and geom["rings"][0]:
            ring_points = geom["rings"][0]
            lon = sum(p[0] for p in ring_points) / len(ring_points)
            lat = sum(p[1] for p in ring_points) / len(ring_points)
        else:
            continue

        results.append({
            "name": name or "Scheduled Monument",
            "lat": lat, "lon": lon,
            "weight": SCHEDULED_MONUMENT_WEIGHT,
        })

    return results


def fetch_graded_historic_sites_in_bbox(bbox, debug=False):
    """
    Convenience wrapper: fetches BOTH Listed Buildings and Scheduled
    Monuments for a bbox, combined into one weighted list — the
    direct, real-magnitude replacement for the old flat "historic"
    OSM-tag-based signal.
    """
    listed = fetch_listed_buildings_in_bbox(bbox, debug=debug)
    monuments = fetch_scheduled_monuments_in_bbox(bbox, debug=debug)
    return listed + monuments


if __name__ == "__main__":
    print("--- Offline test: grade weighting and defensive field matching (no network needed) ---")
    assert LISTED_BUILDING_GRADE_WEIGHTS["i"] > LISTED_BUILDING_GRADE_WEIGHTS["ii*"] > LISTED_BUILDING_GRADE_WEIGHTS["ii"]
    print(f"Grade I weight: {LISTED_BUILDING_GRADE_WEIGHTS['i']}, "
          f"Grade II* weight: {LISTED_BUILDING_GRADE_WEIGHTS['ii*']}, "
          f"Grade II weight: {LISTED_BUILDING_GRADE_WEIGHTS['ii']}")
    print("PASSED — grades rank in the correct order (I > II* > II)\n")

    fake_attrs_capitalized = {"Name": "Test Building", "Grade": "I"}
    fake_attrs_allcaps = {"NAME": "Test Building", "GRADE": "II*"}
    name1 = _first_matching_field(fake_attrs_capitalized, ["Name", "NAME"])
    grade1 = _first_matching_field(fake_attrs_capitalized, ["Grade", "GRADE"])
    name2 = _first_matching_field(fake_attrs_allcaps, ["Name", "NAME"])
    grade2 = _first_matching_field(fake_attrs_allcaps, ["Grade", "GRADE"])
    assert name1 == "Test Building" and grade1 == "I"
    assert name2 == "Test Building" and grade2 == "II*"
    print("PASSED — defensive field matching works regardless of casing convention used\n")

    print("--- Live test: querying around Lacock, Wiltshire (needs real internet access) ---")
    lacock_bbox = {"min_lat": 51.40, "max_lat": 51.43, "min_lon": -2.135, "max_lon": -2.095}

    listed = fetch_listed_buildings_in_bbox(lacock_bbox, debug=True)
    print(f"\nFound {len(listed)} listed building(s):")
    grade_counts = {}
    for r in listed[:10]:
        print(f"  {r['name']} — Grade {r['grade']} (weight={r['weight']}) at ({r['lat']:.4f}, {r['lon']:.4f})")
    for r in listed:
        g = r.get("grade") or "unknown"
        grade_counts[g] = grade_counts.get(g, 0) + 1
    print(f"\nGrade distribution: {grade_counts}")

    monuments = fetch_scheduled_monuments_in_bbox(lacock_bbox, debug=True)
    print(f"\nFound {len(monuments)} scheduled monument(s):")
    for r in monuments[:5]:
        print(f"  {r['name']} (weight={r['weight']}) at ({r['lat']:.4f}, {r['lon']:.4f})")

    if listed or monuments:
        print("\nPASSED — got real data back")
    else:
        print("\nNo results — could mean no internet access here, or a real query problem.")
        print("Check the [DEBUG] lines above: if 'Raw response: 0 feature(s)' with no error,")
        print("the query reached the server fine but matched nothing — worth checking field")
        print("names against the raw attributes of a feature from a wider bbox query.")
