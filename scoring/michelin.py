"""
Byway — Michelin Guide distinction matching (free, structured, no AI)
============================================================================

What this does, in plain terms:
Gives food/drink venues a real, free, expert-curated quality signal —
Michelin Guide distinctions (3/2/1 Stars, Bib Gourmand, Selected
Restaurants) — by downloading a maintained, structured CSV mirror of
the Michelin Guide and matching it against the OSM-sourced venues this
project already fetches (scoring/food_drink.py), rather than scraping
guide.michelin.com directly (which has its own cookie/ToS friction) or
relying on an AI model's opinion of food quality.

WHY THIS EXISTS — direct correction of an earlier wrong proxy: this
project initially reached for CAMRA's National Inventory of Historic
Pub Interiors as a food/drink "quality" signal. Real research found
that's a category error — it measures architectural preservation, not
food quality (a Three Star Inventory pub can have no kitchen at all).
Michelin's distinctions are actually inspector-judged on cooking,
ingredients, and execution — the right KIND of signal, even though
coverage will be sparse outside towns/cities (Michelin's own nature,
not a flaw in how this is fetched). Bib Gourmand and Selected
Restaurants specifically target "good food, fair price, worth a stop"
— much closer to this project's actual use case than 3-star tasting
menus.

DATA SOURCE: a maintained, regularly-updated CSV mirror of the
Michelin Guide (https://github.com/ngshiheng/michelin-my-maps),
downloaded directly from GitHub's raw content host — no Kaggle
account, no API key, no scraping of guide.michelin.com's own ToS-
sensitive pages. This is SOMEONE ELSE'S scrape, republished — real and
currently maintained, but not an official Michelin feed. Treat as
"good until proven stale," same honest framing already applied to
CAMRA's Historic Pub Interiors list, and re-check the URL/schema
periodically rather than assuming it's permanent infrastructure.

WHY BBOX-FILTERING NEEDS NO COUNTRY-STRING MATCHING: the CSV is
GLOBAL (every country Michelin covers). Rather than trust the
"Location" text field's country naming (inconsistent — "Dubai" alone,
"Hamburg, Germany", etc.), this filters purely by real lat/lon against
the route's own bbox, exactly like scenicornot.py already does — a
UK route's bbox will only ever contain UK (or occasionally Irish
border-area) rows, with zero risk from string-matching on country
names that might not even be present.

COPYRIGHT NOTE: the "Description" column is Michelin's own editorial
review text. This module deliberately never loads or surfaces it —
only the factual fields (name, coordinates, award tier, price,
cuisine) are used. Don't add Description to the fields read here
without checking copyright handling first.

PERFORMANCE NOTE: same shape as scenicornot.py — ONE network call ever
(the initial CSV download), then pure in-memory filtering per bbox
query. Safe to call freely once downloaded.

Network note: needs real internet access for the one-time download —
will only work somewhere with a real connection (e.g. GitHub
Codespaces), not inside Claude's sandboxed tool environment.
"""

import os
import csv
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.curvature import haversine_distance_m


CSV_URL = "https://raw.githubusercontent.com/ngshiheng/michelin-my-maps/refs/heads/main/data/michelin_my_maps.csv"
LOCAL_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".michelin_restaurants.csv",
)
# Stored at the project root, matching the same one-time-download-then-
# cache convention as scoring/scenicornot.py's .scenicornot_votes.tsv
# and scoring/elevation.py's .elevation_cache.json.

# Award tier -> weight (0-1), the direct equivalent of ScenicOrNot's
# rating/10 or Historic England's Grade weighting — a real magnitude,
# not flat presence. Ordered by how Michelin itself ranks these tiers,
# EXCEPT Bib Gourmand and Selected Restaurants are deliberately given
# real, non-trivial weight (not "almost zero") because they're the
# tiers actually aimed at "good food, fair price, worth a stop" —
# closer to this project's real use case than 3-star tasting menus.
AWARD_WEIGHTS = {
    "3 stars": 1.0,
    "2 stars": 0.85,
    "1 star": 0.7,
    "bib gourmand": 0.55,
    "selected restaurants": 0.4,
}
DEFAULT_AWARD_WEIGHT = 0.3  # an award string we don't recognise -- still something, not nothing
# Same defensive "unrecognised value still gets a non-zero weight"
# choice already used in historic_england.py's DEFAULT_GRADE_WEIGHT,
# in case the upstream CSV ever adds a new distinction tier we haven't
# seen yet.

DEFAULT_MATCH_DISTANCE_M = 200
# How close an OSM venue's coordinates must be to a Michelin entry's
# coordinates to be treated as "the same place." 200m is deliberately
# generous -- OSM nodes and Michelin's own geocoding can each be
# slightly off from a building's true centre, especially for venues
# inside larger complexes (hotels, retail/dining precincts).

_dataset_cache = None  # in-memory cache across calls within one run/process


def _ensure_dataset_downloaded():
    if not os.path.exists(LOCAL_CACHE_PATH):
        print("  Downloading Michelin Guide dataset mirror (one-time)...")
        urllib.request.urlretrieve(CSV_URL, LOCAL_CACHE_PATH)
        print(f"  Downloaded and cached at {LOCAL_CACHE_PATH} — every future run reuses this, no re-download.")


def _award_weight(award_raw):
    return AWARD_WEIGHTS.get((award_raw or "").strip().lower(), DEFAULT_AWARD_WEIGHT)


def _load_full_dataset():
    """
    Loads the WHOLE Michelin CSV into memory once per process (not
    once per bbox query) -- same spirit as scenicornot.py's _load_
    full_dataset. Uses csv.DictReader (not positional indexing) since
    this is a real, comma-delimited, QUOTED CSV (the Description and
    Address fields routinely contain embedded commas) -- positional
    splitting would silently misalign columns; DictReader handles
    quoting correctly and is robust if the upstream column ORDER ever
    changes, since we read by header name, not position.
    """
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    _ensure_dataset_downloaded()

    records = []
    with open(LOCAL_CACHE_PATH, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("Latitude", ""))
                lon = float(row.get("Longitude", ""))
            except (ValueError, TypeError):
                continue
            award = (row.get("Award") or "").strip()
            records.append({
                "name": (row.get("Name") or "").strip(),
                "lat": lat,
                "lon": lon,
                "award": award,
                "weight": _award_weight(award),
                "price": (row.get("Price") or "").strip() or None,
                "cuisine": (row.get("Cuisine") or "").strip() or None,
                "url": (row.get("Url") or "").strip() or None,
            })

    _dataset_cache = records
    return records


def fetch_michelin_in_bbox(bbox):
    """
    Returns every Michelin Guide entry whose coordinates fall within
    the given bounding box (same {min_lat, max_lat, min_lon, max_lon}
    shape used everywhere else in this codebase).

    Returns a list of {"name", "lat", "lon", "weight", "award",
    "price", "cuisine", "url"} dicts. Empty list (not an error) is the
    normal, expected outcome for most rural stretches -- Michelin
    coverage is real but sparse outside towns, by the guide's own
    nature, not a fetch problem.
    """
    records = _load_full_dataset()
    return [
        r for r in records
        if bbox["min_lat"] <= r["lat"] <= bbox["max_lat"] and bbox["min_lon"] <= r["lon"] <= bbox["max_lon"]
    ]


def _normalize_name(name):
    """Lowercase, strip whitespace/punctuation noise, for loose name comparison."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def attach_michelin_matches(venues, bbox, max_distance_m=DEFAULT_MATCH_DISTANCE_M, verbose=True):
    """
    Cross-references OSM-sourced food/drink venues (the shape returned
    by scoring.food_drink.fetch_food_drink_venues_in_bbox -- dicts
    with "name", "lat", "lon") against Michelin Guide entries in the
    same bbox, by proximity (within max_distance_m) AND a loose name
    match -- proximity alone could match the wrong venue inside a
    dense restaurant row; name alone could match a same-named venue in
    a different town. Requiring both is a deliberately conservative
    bar: a missed real match (false negative) just means no Michelin
    badge shown, business as usual; a wrong match (false positive)
    would misrepresent a real venue's distinction, which matters more
    to get right than catching every possible real match.

    Mutates and returns the SAME venues list -- each matched venue
    gains a "michelin" key (the matched Michelin record's "award",
    "weight", "price", "cuisine", "url"); unmatched venues are left
    untouched (no "michelin" key added at all, not set to None --
    callers should use .get("michelin") rather than assuming the key
    exists).

    Pure in-memory matching after one dataset load -- no extra network
    calls beyond fetch_michelin_in_bbox's own (cached) dataset fetch.
    """
    michelin_entries = fetch_michelin_in_bbox(bbox)
    if not michelin_entries:
        return venues

    match_count = 0
    for venue in venues:
        venue_name_norm = _normalize_name(venue["name"])
        best_match = None
        best_distance_m = None
        for entry in michelin_entries:
            distance_m = haversine_distance_m((venue["lon"], venue["lat"]), (entry["lon"], entry["lat"]))
            if distance_m > max_distance_m:
                continue
            entry_name_norm = _normalize_name(entry["name"])
            if entry_name_norm not in venue_name_norm and venue_name_norm not in entry_name_norm:
                continue
            if best_distance_m is None or distance_m < best_distance_m:
                best_match, best_distance_m = entry, distance_m

        if best_match is not None:
            venue["michelin"] = {
                "award": best_match["award"],
                "weight": best_match["weight"],
                "price": best_match["price"],
                "cuisine": best_match["cuisine"],
                "url": best_match["url"],
            }
            match_count += 1

    if verbose:
        print(f"  Michelin cross-reference: {match_count} of {len(venues)} OSM venue(s) matched "
              f"against {len(michelin_entries)} Michelin entry/entries in this area.")

    return venues


if __name__ == "__main__":
    import tempfile

    print("--- Offline test: award weighting ---")
    assert _award_weight("3 Stars") == 1.0
    assert _award_weight("Bib Gourmand") == 0.55
    assert _award_weight("Selected Restaurants") == 0.4
    assert _award_weight("Some New Tier Nobody's Seen Yet") == DEFAULT_AWARD_WEIGHT
    assert AWARD_WEIGHTS["3 stars"] > AWARD_WEIGHTS["2 stars"] > AWARD_WEIGHTS["1 star"] > AWARD_WEIGHTS["bib gourmand"] > AWARD_WEIGHTS["selected restaurants"]
    print("PASSED — award tiers weighted in the correct order, unrecognised tiers still get a real weight\n")

    print("--- Offline test: bbox filtering + DictReader correctly handles quoted, comma-containing fields ---")
    fake_csv_rows = [
        ["Name", "Address", "Location", "Price", "Cuisine", "Longitude", "Latitude", "PhoneNumber", "Url", "WebsiteUrl", "Award", "GreenStar", "FacilitiesAndServices", "Description"],
        ["The Crown", "1 High Street, Anytown, Surrey, GB", "Anytown, United Kingdom", "££", "British, Modern",
         "-0.700", "51.090", "+44123456", "https://guide.michelin.com/fake/crown", "https://crown.example", "1 Star", "0",
         "Car park,Terrace", "A description, with a comma, and \"quoted\" text inside it."],
        ["Bib Spot", "2 Lane Road, Anytown, Surrey, GB", "Anytown, United Kingdom", "£", "Pub fare",
         "-0.701", "51.091", "+44123457", "https://guide.michelin.com/fake/bibspot", "https://bibspot.example", "Bib Gourmand", "0",
         "", "Another, comma-laden, description."],
        ["Far Away Place", "Somewhere else entirely", "Elsewhere, France", "€€€", "French",
         "2.500", "48.800", "+33100000", "https://guide.michelin.com/fake/faraway", "https://faraway.example", "2 Stars", "0", "", "Not nearby."],
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as tmp:
        writer = csv.writer(tmp)
        for row in fake_csv_rows:
            writer.writerow(row)
        tmp_path = tmp.name

    LOCAL_CACHE_PATH = tmp_path
    _dataset_cache = None

    test_bbox = {"min_lat": 51.08, "max_lat": 51.10, "min_lon": -0.71, "max_lon": -0.69}
    results = fetch_michelin_in_bbox(test_bbox)
    print(f"Found {len(results)} entr(y/ies) within bbox (expected 2 -- 'Far Away Place' must be excluded).")
    assert len(results) == 2
    names_found = {r["name"] for r in results}
    assert names_found == {"The Crown", "Bib Spot"}
    print("PASSED — bbox filtering correctly excludes far-away rows, and the quoted/comma-laden "
          "Description field never misaligned the columns\n")

    print("--- Offline test: attach_michelin_matches matches by proximity AND name, not either alone ---")
    fake_osm_venues = [
        {"name": "The Crown", "lat": 51.0901, "lon": -0.7001, "amenity_type": "pub", "cuisine": "british"},
        {"name": "The Crown", "lat": 48.80, "lon": 2.50, "amenity_type": "pub", "cuisine": None},  # same NAME, wrong place
        {"name": "Some Other Pub", "lat": 51.0905, "lon": -0.7005, "amenity_type": "pub", "cuisine": None},  # right area, wrong name
        {"name": "Bib Spot", "lat": 51.0911, "lon": -0.7011, "amenity_type": "restaurant", "cuisine": "pub fare"},
    ]
    enriched = attach_michelin_matches(fake_osm_venues, test_bbox)
    crown_nearby = next(v for v in enriched if v["name"] == "The Crown" and v["lat"] > 50)
    crown_far = next(v for v in enriched if v["name"] == "The Crown" and v["lat"] < 50)
    other_pub = next(v for v in enriched if v["name"] == "Some Other Pub")
    bib_spot = next(v for v in enriched if v["name"] == "Bib Spot")

    assert "michelin" in crown_nearby and crown_nearby["michelin"]["award"] == "1 Star"
    assert "michelin" not in crown_far, "Same name but a venue on the other side of the world must NOT match"
    assert "michelin" not in other_pub, "Right area, but no real name match -- must NOT match on proximity alone"
    assert "michelin" in bib_spot and bib_spot["michelin"]["award"] == "Bib Gourmand"
    print("PASSED — matches require BOTH proximity and a real name match; neither alone is enough, "
          "so a same-named venue elsewhere and a different-named venue nearby both correctly fail to match\n")

    os.remove(tmp_path)
    print("All Michelin module offline tests passed.")
