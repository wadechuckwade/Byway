"""
Byway — Patch script: wire Michelin + Google Places into 08_three_route_system.py
======================================================================================

What this does: applies two surgical, EXACT string replacements to
graph_search/08_three_route_system.py --

  1. JOURNEY_PREFERENCE_DEFAULTS gets two new flags
     ("enable_food_drink", "enable_google_places_ratings") and one new
     tuning constant ("num_google_places_targets").
  2. The food/drink block is restructured so venue fetching + the free
     Michelin cross-reference run whenever enable_food_drink is True
     (default), INDEPENDENT of enable_food_drink_ai (the AI narration
     sub-step, unchanged, still its own explicit opt-in) -- with the
     new, also-opt-in Google Places enrichment step wired in alongside
     it, capped to a small per-route candidate set.

WHY A PATCH SCRIPT, NOT A HEREDOC OVERWRITE OF THE WHOLE FILE: this
file is 1447 lines and only two specific regions are changing -- a
full-file heredoc rewrite would require trusting a manual
transcription of over a thousand lines this session never even saw in
full. This script instead does an EXACT literal-string replace with a
uniqueness assertion (fails loudly if the anchor text isn't found
exactly once) -- the same safety property str_replace gives, and a
direct, deliberate avoidance of the exact bug class already in this
project's own logs (str_replace boundary issues silently eating a
neighbouring line). If either assertion fails, NOTHING is written --
the script aborts cleanly, telling you the file has likely changed
since this patch was written, rather than guessing and corrupting it.

WHAT CHANGES BEHAVIOUR, NOT JUST ADDS TO IT -- READ THIS BEFORE RUNNING:
previously, food/drink venues were ONLY EVER fetched if
enable_food_drink_ai was True -- meaning the free OSM fetch itself was
gated behind the AI flag, even though fetching itself costs nothing.
After this patch, with the new enable_food_drink defaulting to True,
venues (+ the free Michelin match) are fetched by default even with
AI off. This is a deliberate, positive change matching the whole point
of this session's research -- but it IS a behaviour change from what
existed before, not a pure no-op addition. If some other code (e.g.
app.py) assumed "no food_drink-related work happens unless
enable_food_drink_ai is True," check it before deploying this.

Run from the Codespace root (or adjust TARGET_PATH below to wherever
08_three_route_system.py actually lives).
"""

import sys

TARGET_PATH = "graph_search/08_three_route_system.py"

OLD_DEFAULTS = """    "enable_food_drink_ai": False,
}"""

NEW_DEFAULTS = """    "enable_food_drink_ai": False,
    # NEW -- separate from enable_food_drink_ai: turns on venue
    # fetching + the free Michelin cross-reference even when the AI
    # narration step is off. Defaults to True since this part is now
    # genuinely free (one cached Michelin download, then in-memory
    # matching) -- same "free signals are always-on" convention
    # already used for ScenicOrNot/Historic England/Conservation
    # Areas. Set to False to skip food/drink entirely (e.g. for raw
    # routing-only runs).
    "enable_food_drink": True,
    # NEW -- a SECOND real-cost feature, independent of the AI step.
    # Defaults OFF, same as enable_food_drink_ai -- see scoring/
    # google_places_circuit_breaker.py for the hard monthly free-tier
    # cap that applies regardless.
    "enable_google_places_ratings": False,
    "num_google_places_targets": 5,
}"""

OLD_BLOCK = """    if prefs.get("enable_food_drink_ai") and node_coords is not None:
        _report("Fetching food/drink venues and getting AI recommendations...")
        try:
            from scoring.food_drink import fetch_food_drink_venues_in_bbox
            from scoring.food_drink_ai import get_food_drink_recommendations
            from scoring.proximity import _min_distance_to_route_m

            lats = [lat for lat, lon in node_coords.values()]
            lons = [lon for lat, lon in node_coords.values()]
            fd_bbox = {"min_lat": min(lats), "max_lat": max(lats), "min_lon": min(lons), "max_lon": max(lons)}
            all_venues = fetch_food_drink_venues_in_bbox(fd_bbox)

            for label in list(results.keys()):
                if label.startswith("_"):
                    continue
                r = results[label]
                route_points = []
                seen_way_ids_fd = set()
                for w in r["way_info"]:
                    way_id = w["way_id"]
                    if way_id in seen_way_ids_fd:
                        continue
                    seen_way_ids_fd.add(way_id)
                    if way_id in ways:
                        route_points.extend(ways[way_id]["points"])

                nearby_venues = [
                    v for v in all_venues
                    if route_points and _min_distance_to_route_m(v["lat"], v["lon"], route_points) <= 1000
                ]
                r["food_drink"] = get_food_drink_recommendations(
                    nearby_venues,
                    route_context=f"a {round(r['real_distance_m'] / 1000, 1)}km scenic drive",
                    verbose=verbose,
                )
        except Exception as e:
            if verbose:
                print(f"  Food/drink AI step failed ({type(e).__name__}: {e}) — continuing without it, "
                      f"core route results are unaffected.")"""

NEW_BLOCK = """    if prefs.get("enable_food_drink", True) and node_coords is not None:
        _report("Fetching food/drink venues...")
        try:
            from scoring.food_drink import fetch_food_drink_venues_in_bbox
            from scoring.michelin import attach_michelin_matches
            from scoring.proximity import _min_distance_to_route_m

            lats = [lat for lat, lon in node_coords.values()]
            lons = [lon for lat, lon in node_coords.values()]
            fd_bbox = {"min_lat": min(lats), "max_lat": max(lats), "min_lon": min(lons), "max_lon": max(lons)}
            all_venues = fetch_food_drink_venues_in_bbox(fd_bbox)

            # Free, always-on (no flag, no per-call cost): cross-
            # reference against the Michelin Guide mirror -- mutates
            # all_venues' own dicts in place, so every route's
            # distance-filtered view below inherits any match
            # automatically, with zero extra fetches. See scoring/
            # michelin.py for why this replaced the original (wrong-
            # proxy) CAMRA Historic Pub Interiors idea.
            attach_michelin_matches(all_venues, fd_bbox, verbose=verbose)

            for label in list(results.keys()):
                if label.startswith("_"):
                    continue
                r = results[label]
                route_points = []
                seen_way_ids_fd = set()
                for w in r["way_info"]:
                    way_id = w["way_id"]
                    if way_id in seen_way_ids_fd:
                        continue
                    seen_way_ids_fd.add(way_id)
                    if way_id in ways:
                        route_points.extend(ways[way_id]["points"])

                # CHANGED -- keep each venue's real distance to THIS
                # route, then sort by it. The original version only
                # filtered (<=1000m) and kept whatever order OSM
                # happened to return -- fine when the AI step (which
                # does its own ranking anyway) was the only consumer,
                # but the NEW Google enrichment step below needs a
                # principled "nearest first" order, since it only
                # looks up a small handful, not the whole list.
                venues_with_distance = []
                for v in all_venues:
                    if not route_points:
                        continue
                    d = _min_distance_to_route_m(v["lat"], v["lon"], route_points)
                    if d <= 1000:
                        venues_with_distance.append((d, v))
                venues_with_distance.sort(key=lambda pair: pair[0])
                nearby_venues = [v for _, v in venues_with_distance]

                # NEW -- free factual venue data (OSM + any Michelin
                # match) for this specific route, always populated
                # when enable_food_drink is on, independent of whether
                # AI narration or Google ratings (below) are also
                # enabled. Capped at 20 purely to keep the payload
                # reasonable -- NOT the same cap as the AI step's own
                # MAX_VENUES_TO_OFFER_MODEL.
                r["food_drink_venues"] = nearby_venues[:20]

                # NEW -- Google Places ratings: a SECOND real-cost
                # feature, independent of the AI step, opt-in, OFF by
                # default. Only ever looks up the nearest
                # num_google_places_targets venues for THIS route --
                # never the whole nearby list -- see scoring/
                # google_places.py's module docstring for why.
                if prefs.get("enable_google_places_ratings"):
                    try:
                        from scoring.google_places import enrich_venues_with_google_ratings
                        num_targets = prefs.get("num_google_places_targets", 5)
                        enrich_venues_with_google_ratings(
                            nearby_venues[:num_targets],
                            enable=True, max_venues=num_targets, verbose=verbose,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"  Google Places enrichment failed ({type(e).__name__}: {e}) — "
                                  f"continuing without it, core route results are unaffected.")

                # UNCHANGED -- AI narration, still its own explicit
                # opt-in flag (separate from enable_food_drink, which
                # now just gates the free fetch/Michelin step above).
                if prefs.get("enable_food_drink_ai"):
                    from scoring.food_drink_ai import get_food_drink_recommendations
                    r["food_drink"] = get_food_drink_recommendations(
                        nearby_venues,
                        route_context=f"a {round(r['real_distance_m'] / 1000, 1)}km scenic drive",
                        verbose=verbose,
                    )
        except Exception as e:
            if verbose:
                print(f"  Food/drink step failed ({type(e).__name__}: {e}) — continuing without it, "
                      f"core route results are unaffected.")"""


def apply_patch(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    defaults_count = content.count(OLD_DEFAULTS)
    block_count = content.count(OLD_BLOCK)

    if defaults_count != 1:
        print(f"ABORTING -- expected the JOURNEY_PREFERENCE_DEFAULTS anchor exactly once, found {defaults_count}. "
              f"The file has likely changed since this patch was written -- nothing was written, "
              f"investigate before re-running.")
        sys.exit(1)

    if block_count != 1:
        print(f"ABORTING -- expected the food/drink block anchor exactly once, found {block_count}. "
              f"The file has likely changed since this patch was written -- nothing was written, "
              f"investigate before re-running.")
        sys.exit(1)

    content = content.replace(OLD_DEFAULTS, NEW_DEFAULTS)
    content = content.replace(OLD_BLOCK, NEW_BLOCK)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Patched {path} successfully:")
    print("  - JOURNEY_PREFERENCE_DEFAULTS: added enable_food_drink (default True), "
          "enable_google_places_ratings (default False), num_google_places_targets (default 5)")
    print("  - Food/drink block: Michelin cross-reference now wired in (always-on, free); "
          "Google Places enrichment wired in (opt-in, capped); AI narration logic UNCHANGED, "
          "just re-gated under its own explicit check now that the outer condition changed.")
    print()
    print("VERIFY before trusting this further (per this project's own standing practice -- "
          "py_compile success alone isn't proof of correctness):")
    print(f"  python3 -c \"import ast; ast.parse(open('{path}').read())\"   # confirms it's syntactically valid")
    print(f"  grep -n '^def \\|^    \"enable_' {path}   # confirm every def is still top-level, "
          f"and the 3 new/changed prefs keys are really there")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET_PATH
    apply_patch(target)
