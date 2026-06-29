"""
Byway — ScenicOrNot Scenicness Scoring
=========================================

What this does, in plain terms:
Uses real, crowd-sourced "how scenic is this place" ratings (1-10
scale) from ScenicOrNot — 217,000 geotagged photos covering ~95% of
Great Britain's 1km grid squares, 1.5 million+ human ratings — as a
direct scenicness signal, rather than trying to reconstruct "is this
place pretty/dramatic" purely from proxies like water/forest/elevation
proximity.

WHY THIS EXISTS: real-route testing (Winnats Pass, Peak District)
found that NONE of the existing scenery signals (water, forest,
elevation-as-net-climbing, village/Conservation-Area proximity)
capture dramatic landscape character — a road through a narrow
limestone gorge with towering crags either side. Checking ScenicOrNot
directly for this exact road confirmed a real, consistent cluster of
high human ratings nearby (6.2–8.25/10 across 7 separate photos, vs
ScenicOrNot's own ~4-5/10 typical baseline) — exactly the kind of
signal none of this project's existing categories would ever surface.
This is literally the project's own previously-cited research lineage
(Seresinhe, Preis & Moat, building on Quercia's earlier work) — direct
human aesthetic judgment, not a proxy we're approximating.

DATA SOURCE: http://scenicornot.datasciencelab.co.uk/votes.tsv
Licensed under the Open Database Licence (ODbL) — same license family
as OpenStreetMap itself, confirmed directly from the dataset's current
maintainers (Data Science Lab, Warwick Business School, via their own
FAQ page). Commercial use is fine under ODbL's attribution/share-alike
terms — the same kind of obligation this project already has for OSM
data, not a new category of constraint.

HONEST LIMITATIONS:
- The downloadable snapshot covers votes through February 2015.
  Landscape doesn't move, so this is a non-issue for the geological/
  topographic features (gorges, hills, water, dramatic terrain) that
  mostly drive scenicness — but it wouldn't reflect anything that's
  changed since (new development, a since-removed eyesore, etc.).
- ~95% grid coverage, not 100% — a "no rating nearby" result means "no
  signal," not "this place definitely isn't scenic," the same honest
  caveat already applied to Conservation Areas' incomplete coverage.
- Each rating is for ONE PHOTO from ONE SPOT within a ~1km grid square
  — a real, human-judged sample of that square's character, not a
  guarantee every road within the square shares that exact view.

PERFORMANCE NOTE: unlike elevation, this needs only ONE network call
EVER (the initial dataset download, ~20-30MB) — after that, every bbox
query is pure in-memory Python, no rate limit, no per-point cost. Safe
to compute for an entire graph's worth of ways in a single PHASE 1
pass (see 07_score_graph_enjoyment.py) — no phase-2 deferral needed,
unlike elevation's genuinely rate-limited per-point fetches.

Network note: needs real internet access for the one-time download —
will only work somewhere with a real connection (e.g. GitHub
Codespaces), not inside Claude's sandboxed tool environment. Once
downloaded, works entirely offline.
"""

import os
import csv
import math
import urllib.request


VOTES_URL = "http://scenicornot.datasciencelab.co.uk/votes.tsv"
LOCAL_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".scenicornot_votes.tsv",
)
# Stored at the project root (one level up from scoring/), matching
# the same convention as scoring/elevation.py's .elevation_cache.json
# — a one-time download cached locally, never re-fetched once present.

MIN_VOTES_TO_TRUST = 3
# ScenicOrNot's own FAQ states every photo is rated at least 3 times
# "to help limit abuse, or the impact of people with terrible taste" —
# the public votes.tsv already only includes photos meeting this bar,
# but we check again defensively here in case that ever changes
# upstream, rather than silently trusting it forever.

_dataset_cache = None  # in-memory cache across calls within one run/process


def _ensure_dataset_downloaded():
    if not os.path.exists(LOCAL_CACHE_PATH):
        print("  Downloading ScenicOrNot scenicness dataset (one-time, ~20-30MB)...")
        urllib.request.urlretrieve(VOTES_URL, LOCAL_CACHE_PATH)
        print(f"  Downloaded and cached at {LOCAL_CACHE_PATH} — every future run reuses this, no re-download.")


def _load_full_dataset():
    """
    Loads the WHOLE ScenicOrNot dataset into memory once per process
    (not once per bbox query) — ~217,000 rows of plain floats/strings,
    a few tens of MB, entirely reasonable to hold for the lifetime of
    a single route-scoring run, the same spirit as this project's
    existing whole-area region-data fetches.
    """
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    _ensure_dataset_downloaded()

    records = []
    with open(LOCAL_CACHE_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 6:
                continue
            try:
                lat, lon, avg_rating = float(row[1]), float(row[2]), float(row[3])
                votes = row[5].split(",") if row[5] else []
            except (ValueError, IndexError):
                continue
            if len(votes) < MIN_VOTES_TO_TRUST:
                continue
            records.append({"lat": lat, "lon": lon, "avg_rating": avg_rating, "num_votes": len(votes)})

    _dataset_cache = records
    return records


def fetch_scenicornot_in_bbox(bbox):
    """
    Returns every ScenicOrNot rating whose coordinates fall within the
    given bounding box (same {min_lat, max_lat, min_lon, max_lon}
    shape used everywhere else in this codebase — see
    scoring/proximity.py's route_bounding_box()).

    Formatted to match the {"name", "lat", "lon", "weight"} shape
    score_proximity() now expects (see scoring/proximity.py) — "weight"
    is the rating normalised to 0-1 (avg_rating / 10), so an 8/10 spot
    contributes meaningfully more than a 4/10 spot, not just
    "present vs absent" the way Conservation Areas or water/forest
    features do.

    No network call happens here beyond the ONE-TIME dataset download
    (cached locally on first use) — filtering to a bbox is pure
    in-memory Python, so this is safe to call for an entire graph's
    worth of ways in one PHASE 1 pass, unlike elevation.
    """
    records = _load_full_dataset()
    results = []
    for r in records:
        if (bbox["min_lat"] <= r["lat"] <= bbox["max_lat"]
                and bbox["min_lon"] <= r["lon"] <= bbox["max_lon"]):
            results.append({
                "name": f"Scenic spot ({r['avg_rating']}/10, {r['num_votes']} votes)",
                "lat": r["lat"], "lon": r["lon"],
                "weight": round(r["avg_rating"] / 10, 3),
            })
    return results


if __name__ == "__main__":
    # Self-test using a small FAKE local dataset (no internet needed)
    # to verify bbox filtering and weight calculation — the real
    # download/parsing is exercised for real whenever this module is
    # actually used (e.g. via 15_check_scenicornot_winnats.py, already
    # confirmed working against real data this session).
    import tempfile

    fake_tsv_rows = [
        ["1", "53.3408", "-1.8000", "8.25", "0.5", "10,7,9,7", "http://example.com/1"],   # inside bbox, high rating
        ["2", "53.3450", "-1.8050", "4.0", "0.5", "4,4,4,4", "http://example.com/2"],      # inside bbox, average rating
        ["3", "51.0000", "0.5000", "9.0", "0.5", "9,9,9", "http://example.com/3"],          # FAR outside bbox
        ["4", "53.3410", "-1.8010", "7.0", "0.5", "7,7", "http://example.com/4"],           # inside bbox, but only 2 votes -- should be excluded
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, newline="") as tmp:
        writer = csv.writer(tmp, delimiter="\t")
        for row in fake_tsv_rows:
            writer.writerow(row)
        tmp_path = tmp.name

    # Point this module at the fake file instead of the real cache.
    LOCAL_CACHE_PATH = tmp_path
    _dataset_cache = None

    print("--- Test 1: bbox filtering excludes far-away points ---")
    test_bbox = {"min_lat": 53.30, "max_lat": 53.38, "min_lon": -1.85, "max_lon": -1.75}
    results = fetch_scenicornot_in_bbox(test_bbox)
    print(f"Found {len(results)} result(s) within bbox.")
    found_ids_by_lat = {r["lat"] for r in results}
    assert 51.0 not in found_ids_by_lat, "The far-away point (lat 51.0) must be excluded by bbox filtering"
    print("PASSED — far-away point correctly excluded\n")

    print("--- Test 2: low-vote-count points are excluded (MIN_VOTES_TO_TRUST) ---")
    assert 53.3410 not in found_ids_by_lat, (
        "The 2-vote point must be excluded even though it's within the bbox -- "
        "doesn't meet MIN_VOTES_TO_TRUST"
    )
    print(f"Results after filtering: {len(results)} (expected 2, not 3 -- the 2-vote point excluded)")
    assert len(results) == 2
    print("PASSED — low-confidence (under 3 votes) points correctly excluded\n")

    print("--- Test 3: weight is correctly normalised to 0-1 from the 1-10 rating ---")
    high_rated = next(r for r in results if r["lat"] == 53.3408)
    avg_rated = next(r for r in results if r["lat"] == 53.3450)
    print(f"8.25/10 rating -> weight={high_rated['weight']}")
    print(f"4.0/10 rating -> weight={avg_rated['weight']}")
    assert high_rated["weight"] == 0.825
    assert avg_rated["weight"] == 0.4
    assert high_rated["weight"] > avg_rated["weight"]
    print("PASSED — weight correctly reflects the real rating magnitude, not just presence\n")

    os.remove(tmp_path)
    print("All ScenicOrNot module tests passed.")