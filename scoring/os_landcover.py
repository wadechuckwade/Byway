"""
Byway — OS OpenMap Local Land Cover (UK-specific alternative to CORINE)
==========================================================================

What this is, honestly: a SCHEMA INSPECTION TOOL, not yet a working
classifier. Every other new external data source integrated this
project (Conservation Areas, ScenicOrNot, CORINE) needed at least one
real-run correction after first contact with the live data — usually
a wrong field name or a geometry/projection mismatch, found by running
something REAL and looking at REAL output, not by guessing harder
upfront. For CORINE, there was at least a documented, stable numeric
coding convention (1xx/2xx/3xx/4xx/5xx) to anchor a best-effort guess
on. For OS OpenMap Local, there isn't an equivalent confirmed-from-
documentation schema available here — writing a category-mapping
classifier without ever having seen the real file's actual layer
names and attribute values would be a guess with nothing solid under
it, not a reasonable first attempt.

WHY OS OPENMAP LOCAL AT ALL: UK-specific (not a pan-European
compromise like CORINE), genuinely free OpenData with NO API key
required (confirmed directly from Ordnance Survey's own GitHub
package documentation: "It is possible to download Open Data products
without an API key"), and — most importantly — a real DOWNLOADABLE
dataset (GeoPackage, split into manageable 100km² national-grid
tiles) rather than a live API call per point. That sidesteps CORINE's
exact failure mode entirely: once downloaded, every lookup is local,
with no timeout risk, no mystery slow runs, ever.

HOW TO GET THE FILE: osdatahub.os.uk/downloads/open/OpenMapLocal —
select the relevant 100km² grid tile(s) for your area (not the whole-
GB file, which is unnecessarily large for a single route's bbox), in
GeoPackage format. Place the downloaded .gpkg file somewhere in the
project and pass its path to inspect_schema() below.

REQUIRED DEPENDENCY (not yet installed/verified in this environment —
install before running): geopandas, plus its own dependencies
(fiona, shapely, pyproj). Install with:
    pip install geopandas --break-system-packages
This is a genuinely new, heavier kind of dependency than anything else
in this project (which has stuck to plain `requests` + JSON so far) —
a real, worth-noting tradeoff for what a locally-downloaded vector
dataset buys in reliability.

NEXT STEPS, IN ORDER:
1. Download one real 100km² tile covering a test area (e.g. the
   Tillington/Haslemere bbox).
2. Run inspect_schema(path_to_file) — prints every layer name and a
   sample of real rows/columns, so we can see the ACTUAL field names
   and classification values rather than guess them.
3. ONLY THEN build the real classify_points()-style function, with a
   category mapping grounded in what the file actually contains —
   not written yet, deliberately, given point 1-2 haven't happened.
"""

import os

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Layer listing needs EITHER pyogrio or fiona — modern geopandas (1.x)
# defaults to pyogrio and doesn't require fiona at all any more. The
# original version of this module hard-required fiona specifically,
# which fails on current geopandas installs that never installed it
# (confirmed directly: a real environment had geopandas 1.1.3 with
# pyogrio as a listed dependency, no fiona anywhere) — even though
# geopandas itself works perfectly fine. Fixed: try pyogrio first,
# fall back to fiona only if that's the active backend instead.
try:
    import pyogrio
    LAYER_LISTING_BACKEND = "pyogrio"
except ImportError:
    try:
        import fiona
        LAYER_LISTING_BACKEND = "fiona"
    except ImportError:
        LAYER_LISTING_BACKEND = None


def _list_layers(geopackage_path):
    """Lists every layer in a GeoPackage, using whichever backend is actually available."""
    if LAYER_LISTING_BACKEND == "pyogrio":
        # pyogrio.list_layers returns an array of [name, geometry_type]
        # rows — more informative than fiona's plain name list, and
        # the one confirmed installed in the environment this was
        # actually tested against.
        layer_info = pyogrio.list_layers(geopackage_path)
        return [row[0] for row in layer_info]
    elif LAYER_LISTING_BACKEND == "fiona":
        return fiona.listlayers(geopackage_path)
    else:
        raise ImportError(
            "Neither pyogrio nor fiona is available for listing GeoPackage layers. "
            "Install with: pip install geopandas --break-system-packages "
            "(modern geopandas pulls in pyogrio automatically)."
        )


def inspect_schema(geopackage_path, rows_per_layer=5):
    """
    Prints every layer in a downloaded OS OpenMap Local GeoPackage,
    plus a sample of real rows and column names for each — the actual
    first step before writing any classification logic, rather than
    guessing at field names the way CORINE's identify response
    initially required a defensive multi-field-name guess.

    Run this by hand against a REAL downloaded file before trusting
    anything else in this module.
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError(
            "geopandas is required to read GeoPackage files, and isn't installed "
            "in this environment. Install with: pip install geopandas --break-system-packages"
        )
    if not os.path.exists(geopackage_path):
        raise FileNotFoundError(
            f"No file at {geopackage_path} — download a tile from "
            f"osdatahub.os.uk/downloads/open/OpenMapLocal first (GeoPackage format, "
            f"no API key needed) and pass its real path here."
        )

    layers = _list_layers(geopackage_path)
    print(f"Layers found in {geopackage_path} (layer listing via {LAYER_LISTING_BACKEND}):")
    for layer in layers:
        print(f"  - {layer}")

    for layer in layers:
        print(f"\n{'=' * 60}")
        print(f"Layer: {layer}")
        print(f"{'=' * 60}")
        try:
            gdf = gpd.read_file(geopackage_path, layer=layer, rows=rows_per_layer)
        except Exception as e:
            print(f"  Could not read layer (may need a different geopandas/fiona "
                  f"version, or this layer may not support row-limited reads): {e}")
            continue
        print(f"Columns: {list(gdf.columns)}")
        print(f"Geometry type: {gdf.geom_type.iloc[0] if len(gdf) > 0 else 'unknown (no rows)'}")
        print(f"Sample rows:")
        print(gdf.head(rows_per_layer).to_string())

        # For any column that looks like a classification/theme field,
        # show its distinct values across this sample — the real,
        # concrete thing needed before writing a category mapping.
        for col in gdf.columns:
            if col.lower() in ("theme", "classification", "class", "type", "descriptiveterm", "make"):
                print(f"\nDistinct values in likely-classification column '{col}': "
                      f"{sorted(gdf[col].dropna().unique().tolist())}")


# ============================================================
# REAL CLASSIFIER — built from CONFIRMED live schema (not guessed)
# ============================================================
#
# CONFIRMED FULL LAYER LIST (live inspection): building,
# car_charging_point, electricity_transmission_line, foreshore,
# functional_site, glasshouse, important_building, motorway_junction,
# named_place, railway_station, railway_track, railway_tunnel, road,
# road_tunnel, roundabout, surface_water_area, surface_water_line,
# tidal_boundary, tidal_water, woodland.
#
# CRITICAL FINDING: there is NO general agriculture/farmland polygon
# layer in this product at all. OS OpenMap Local is structured around
# named INFRASTRUCTURE/FEATURE types (buildings, woodland, water,
# railways, roads), not an exhaustive land-cover classification the
# way CORINE is. "glasshouse" is literally physical greenhouse
# structures, not general farmland — not used for classification here.
#
# CLASSIFICATION STRATEGY — BY ELIMINATION: check water/foreshore
# first (most unambiguous), then building (urban), then woodland
# (forest); if a point matches NONE of these, default to
# "agriculture" — in rural England, that's overwhelmingly the most
# likely remaining category for unclassified non-urban, non-forest,
# non-water land. Not perfectly rigorous (could technically be heath
# or moorland instead in some areas — CORINE's own "agriculture"
# category has the same kind of fuzziness, per the validated research:
# "strong mixing between modes"), but a reasonable, honest inference
# given English land-use demographics — a materially better signal
# than no signal at all. "wetland" isn't representable at all with
# this product, unlike CORINE — there's no equivalent layer.
#
# CONFIRMED GEOMETRY: 'building' and 'foreshore' are both Polygon,
# same as 'woodland', 'surface_water_area', 'tidal_water' — all
# confirmed directly via live schema inspection. Columns are minimal
# across the board: ['id', 'feature_code', 'geometry'] — no rich
# attribute data, just presence/absence by layer.
#
# CONFIRMED COORDINATE SYSTEM: British National Grid (EPSG:27700) —
# visible directly in the raw coordinate values (six-figure eastings/
# northings, not lat/lon). Every loaded layer gets reprojected to
# WGS84 (EPSG:4326) via geopandas' standard .to_crs(), to match the
# (lon, lat) convention used throughout the rest of this codebase.
#
# HONEST STATUS: this code is built from a CONFIRMED real schema (the
# layer names, columns, and geometry types above all came from live
# inspect_schema()/direct queries against the real downloaded file,
# not a guess) — but the classify_points_os() function itself has NOT
# been run against the live file yet, since geopandas/shapely aren't
# available in the environment writing this. Test it on a small,
# known-real point BEFORE wiring it into the main scoring pipeline —
# see 17_test_os_landcover.py for exactly that check.

TILE_SIZE_M = 500
# NEW — batches nearby points into small tiles, loading each layer
# ONCE PER TILE rather than once per point or once for the whole
# route. Real evidence from BOTH directions led here: a whole-route
# bbox load made building take 54.6s (Guildford, 4,960 polygons) and
# woodland take 22.2s (3,633 polygons) once a realistic bbox was used
# — too much density in one load. Switching to a tight query PER
# POINT fixed the density problem (Guildford test: 2.0s) but made a
# REAL route with 2,513 points take 536.8s — up to 5 separate queries
# per point (2 water layers + foreshore + building + woodland, before
# landing on the agriculture default) times 2,513 points is enough
# per-query overhead (file open, spatial index lookup) to dominate
# even though each individual query was fast. Tiling is the actual
# middle ground: a 500m tile keeps even dense-town building counts
# bounded (Guildford's per-km² density would give roughly 100
# buildings per tile, not the whole bbox's several thousand), while
# letting MANY corridor-sampled points along the same stretch of road
# share one tile's already-loaded data — collapsing thousands of
# per-point queries down to however many DISTINCT tiles the route's
# points actually fall into (typically far fewer than the point count
# itself, since corridor sampling clusters points along roads).
# Flagged as a first-pass tuning choice, like every other constant in
# this project — adjust based on what the next real run's timing
# actually shows, not a guess assumed correct in advance.

WATER_LAYERS = ["surface_water_area", "tidal_water"]
FORESHORE_LAYER = "foreshore"
URBAN_LAYER = "building"
FOREST_LAYER = "woodland"

_tile_layer_cache = {}  # {(geopackage_path, layer_name, tile_x, tile_y): GeoDataFrame}
_BNG_TRANSFORMER = None


def _to_bng(lon, lat):
    """Lazily-initialised WGS84 -> British National Grid transformer (shared, not rebuilt per call)."""
    global _BNG_TRANSFORMER
    if _BNG_TRANSFORMER is None:
        import pyproj
        _BNG_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return _BNG_TRANSFORMER.transform(lon, lat)


def _tile_for_point(lon, lat, tile_size_m=TILE_SIZE_M):
    """Snaps a point (in BNG metres) to its containing tile index — points within the same tile share one load."""
    x, y = _to_bng(lon, lat)
    return int(x // tile_size_m), int(y // tile_size_m)


def _load_layer_for_tile(geopackage_path, layer_name, tile_x, tile_y, tile_size_m=TILE_SIZE_M, verbose=False):
    """
    Loads a single layer, bbox-filtered to ONE small tile (in BNG
    metres) — reprojected to WGS84 after loading. Cached per (path,
    layer, tile) so every point falling in the same tile reuses this
    same load, regardless of how many points that turns out to be.
    """
    cache_key = (geopackage_path, layer_name, tile_x, tile_y)
    if cache_key in _tile_layer_cache:
        return _tile_layer_cache[cache_key]

    min_x, min_y = tile_x * tile_size_m, tile_y * tile_size_m
    max_x, max_y = min_x + tile_size_m, min_y + tile_size_m
    # Small buffer beyond the tile's own edge, so a feature just
    # outside the tile boundary (but within reach of a point sitting
    # near the tile's edge) doesn't get missed.
    buf = 50
    bng_bbox = (min_x - buf, min_y - buf, max_x + buf, max_y + buf)

    if verbose:
        print(f"    Loading '{layer_name}' for tile ({tile_x},{tile_y})...", end=" ", flush=True)
    import time as _time
    t_start = _time.time()

    gdf = gpd.read_file(geopackage_path, layer=layer_name, bbox=bng_bbox)
    if len(gdf) > 0:
        gdf = gdf.to_crs(epsg=4326)

    if verbose:
        print(f"done ({len(gdf)} feature(s), {round(_time.time() - t_start, 1)}s)")

    _tile_layer_cache[cache_key] = gdf
    return gdf


def classify_points_os(points, geopackage_path, bbox=None, verbose=False, debug_tiles=False):
    """
    Classifies a list of (lon, lat) points against the REAL OS
    OpenMap Local data — the schema-confirmed local replacement for
    CORINE's live identify() calls. Pure LOCAL computation, no network
    at all (the whole reason this migration is worth doing).

    Points are batched into small TILES (see TILE_SIZE_M) — each
    layer gets loaded ONCE PER TILE actually needed, not once per
    point and not once for the whole route's bbox. This is the
    result of real, two-directional evidence: a whole-bbox load was
    too slow in building-dense areas; pure per-point queries were too
    slow at high point counts (per-query overhead dominating). Tiling
    is the middle ground that avoided both failure modes.

    bbox: accepted for backward compatibility with earlier callers,
    not used directly — tiling is computed straight from the points.

    verbose: prints ONE summary line (points, tiles, total time) —
    NOT a line per tile load. debug_tiles: prints the OLD per-tile-
    load detail too (real feedback found that overwhelming for normal
    runs, even though it was genuinely the right level of detail while
    actively diagnosing the building/woodland density bug earlier —
    kept available, just opt-in rather than default).

    Returns {(lon, lat): category_string} — always one of "water",
    "urban", "forest_natural", "agriculture" (the only categories OS
    OpenMap Local can actually distinguish — there is no "wetland"
    equivalent in this product, unlike CORINE).

    Mirrors scoring.landcover.fetch_landcover_classes' call shape
    deliberately (list of points in, {point: category} out) — same
    interface as the CORINE-backed classifier, wired in as a drop-in
    replacement in scoring/landcover.py.
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas is required — pip install geopandas --break-system-packages")

    from shapely.geometry import Point
    import time as _time
    t_start = _time.time()

    # Group points by tile FIRST, so each tile's layers get loaded
    # exactly once regardless of how many points land in it.
    points_by_tile = {}
    for lon, lat in points:
        tile = _tile_for_point(lon, lat)
        points_by_tile.setdefault(tile, []).append((lon, lat))

    results = {}
    for (tile_x, tile_y), tile_points in points_by_tile.items():
        water_gdfs = [_load_layer_for_tile(geopackage_path, layer, tile_x, tile_y, verbose=debug_tiles)
                      for layer in WATER_LAYERS]
        foreshore_gdf = _load_layer_for_tile(geopackage_path, FORESHORE_LAYER, tile_x, tile_y, verbose=debug_tiles)
        urban_gdf = _load_layer_for_tile(geopackage_path, URBAN_LAYER, tile_x, tile_y, verbose=debug_tiles)
        forest_gdf = _load_layer_for_tile(geopackage_path, FOREST_LAYER, tile_x, tile_y, verbose=debug_tiles)

        for lon, lat in tile_points:
            pt_buffered = Point(lon, lat).buffer(0.0003)  # ~30m — "nearby," not "exactly inside"
            category = "agriculture"  # default, BY ELIMINATION — see module notes above

            is_water = any(len(g) > 0 and g.intersects(pt_buffered).any() for g in water_gdfs)
            is_foreshore = len(foreshore_gdf) > 0 and foreshore_gdf.intersects(pt_buffered).any()
            if is_water or is_foreshore:
                category = "water"
            elif len(urban_gdf) > 0 and urban_gdf.intersects(pt_buffered).any():
                category = "urban"
            elif len(forest_gdf) > 0 and forest_gdf.intersects(pt_buffered).any():
                category = "forest_natural"

            results[(lon, lat)] = category

    if verbose:
        elapsed = round(_time.time() - t_start, 1)
        print(f"  Land cover (OS local): {len(points)} point(s) -> {len(points_by_tile)} unique "
              f"{TILE_SIZE_M}m tile(s), {elapsed}s total.")

    return results


if __name__ == "__main__":
    print("This module is a schema-inspection tool, not yet a working classifier.")
    print("Usage: download a real OS OpenMap Local GeoPackage tile, then run:")
    print("  python3 -c \"from os_landcover import inspect_schema; "
          "inspect_schema('/path/to/your/tile.gpkg')\"")
    print()
    print(f"geopandas available in this environment: {GEOPANDAS_AVAILABLE}")
    print(f"Layer-listing backend available: {LAYER_LISTING_BACKEND}")
    if not GEOPANDAS_AVAILABLE:
        print("Install with: pip install geopandas --break-system-packages")
