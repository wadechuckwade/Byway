"""
Byway — Isolated test of the real OS OpenMap Local classifier
==================================================================

WHY THIS EXISTS: classify_points_os() (scoring/os_landcover.py) is
built from a CONFIRMED real schema, but has never actually been run
against the live file — geopandas/shapely aren't available in the
environment that wrote it. Before wiring this into the main scoring
pipeline (replacing CORINE in scoring/landcover.py), check it against
a small number of points where the right answer is obvious and
checkable by eye:
  - The middle of a known town/city centre -> should classify "urban"
  - A point in open countryside, away from any town -> "agriculture"
    (the elimination default) or "forest_natural" if it happens to
    land in woodland
  - A point on a lake or river -> "water"

If any of these come back wrong, that's real, useful evidence about
what needs fixing (geometry/CRS handling, layer selection, etc.) —
exactly the same "test small before trusting it everywhere" approach
already used for every other new data source this session.

Run from the project root (needs data/opmplc_gb.gpkg and a real
geopandas install).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.os_landcover import classify_points_os, GEOPANDAS_AVAILABLE


GEOPACKAGE_PATH = "data/opmplc_gb.gpkg"

# Known, checkable test points — (name, lon, lat, expected_category).
# Coordinates are deliberately well inside each area, not near a
# boundary, to avoid an "edge case got the wrong side" result being
# mistaken for "the whole approach is wrong."
#
# HONEST CAVEAT: these coordinates come from general knowledge, not
# map verification (no live map access while writing this) — Guildford
# town centre is the one I'm genuinely confident about (a major town,
# dense building coverage, low risk of landing wrong). The others
# carry real uncertainty: Ashdown Forest is famously more heathland
# than dense woodland in many parts (the name is a historical "royal
# hunting ground" term, not a literal tree-cover description) — a
# mismatch there could mean either a real classifier problem OR just
# an imprecise pin. Read mismatches on those with that caveat in mind,
# not as automatic proof of a bug.
TEST_POINTS = [
    ("Guildford town centre (should be urban — high confidence)", -0.5704, 51.2362, "urban"),
    ("Open countryside south of Lurgashall, away from the village itself "
     "(should be agriculture or forest, NOT urban)", -0.660, 51.015, None),
    ("Bewl Water reservoir, Kent/Sussex border (should be water — "
     "coordinate not map-verified, moderate confidence)", 0.4156, 51.0648, "water"),
    ("Ashdown Forest (should be forest_natural OR could be heathland/"
     "agriculture — much of it isn't densely wooded; low confidence "
     "this specific point lands on tree cover)", 0.0500, 51.0500, None),
]


if __name__ == "__main__":
    if not GEOPANDAS_AVAILABLE:
        print("geopandas not available — install with: pip install geopandas --break-system-packages")
        sys.exit(1)

    if not os.path.exists(GEOPACKAGE_PATH):
        print(f"No file found at {GEOPACKAGE_PATH} — adjust GEOPACKAGE_PATH at the top of this script "
              f"to wherever you actually saved it.")
        sys.exit(1)

    print(f"Testing {len(TEST_POINTS)} known points against {GEOPACKAGE_PATH}, ONE AT A TIME, "
          f"each with its own small local bbox.")
    print("(FIXED from the first version of this script, which built ONE bbox spanning ALL points "
          "combined — since these test points are scattered across ~70km of South East England, "
          "that meant loading huge swaths of building/woodland data unnecessarily. Real usage, e.g. "
          "corridor-sampled points along one route, would share a naturally tight bbox already — "
          "this was a test-script-only problem, not a real usage pattern.)\n")

    all_as_expected = True
    for name, lon, lat, expected in TEST_POINTS:
        print(f"--- {name} ---")
        print(f"  Loading relevant layers for a small bbox around ({lon}, {lat})...", flush=True)
        import time as _time
        t_start = _time.time()

        # A TIGHT local bbox (roughly 2km buffer) -- fast regardless
        # of the GeoPackage's total size, since bbox-filtering happens
        # at the READ level (see _load_layer_for_bbox).
        buffer = 0.02
        local_bbox = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)
        results = classify_points_os([(lon, lat)], GEOPACKAGE_PATH, bbox=local_bbox, verbose=True, debug_tiles=True)
        actual = results.get((lon, lat))

        elapsed = round(_time.time() - t_start, 1)
        print(f"  Done in {elapsed}s.")

        if expected is None:
            status = "(no single expected answer — just check this looks plausible)"
        elif actual == expected:
            status = "MATCHES expected"
        else:
            status = f"DOES NOT MATCH expected ({expected})"
            all_as_expected = False
        print(f"  -> classified as: {actual}  {status}\n")

    print("=" * 60)
    if all_as_expected:
        print("All points with a clear expected answer matched. This is real evidence")
        print("the classifier is working correctly, not yet a guarantee for every")
        print("possible location — but a solid basis for wiring it into the main pipeline.")
    else:
        print("At least one point didn't match its expected category — worth checking")
        print("WHY before trusting this further: wrong CRS handling? Wrong layer? The")
        print("point itself landing closer to a boundary than expected? Real evidence")
        print("either way, not a guess.")
    print("=" * 60)
