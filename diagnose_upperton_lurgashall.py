"""
Byway — Diagnostic: score the ACTUAL known-nicer route (via Upperton
and Lurgashall), using Milestone 1's hand-specified-waypoint method —
the same approach already proven on Witney/Dunwich/Mam Tor.

WHY THIS EXISTS, SEPARATE FROM 09_diagnose_known_nicer_route.py:
That script re-derives "the nicer route" by running plain-distance
Dijkstra and assuming the result matches what the user meant. It
doesn't — the user has clarified the real route goes via Upperton and
Lurgashall specifically. Rather than trust an assumption, this script
scores that EXACT route by hand-specifying both villages as waypoints,
the same proven method used for every Milestone 1 test case.

Run from the repo root: python diagnose_upperton_lurgashall.py
Requires real internet access (Codespaces), not Claude's sandbox.
"""

from milestone_1_score_route import score_route, compare_routes

TILLINGTON = {"name": "Tillington", "lat": 50.9896, "lon": -0.6313}
HASLEMERE = {"name": "Haslemere", "lat": 51.089, "lon": -0.710}

# Approximate centroids from listed-building/OS-grid records — close,
# but worth confirming against Google Maps before trusting precision,
# since OSRM needs these close enough to the real lane to route through
# it rather than nearby footpaths or the wrong fork.
UPPERTON = {"name": "Upperton", "lat": 50.996, "lon": -0.637}
LURGASHALL = {"name": "Lurgashall", "lat": 51.035, "lon": -0.6655}

if __name__ == "__main__":
    direct_result = score_route(
        TILLINGTON, HASLEMERE, route_label="Direct (fastest) route"
    )

    scenic_result = score_route(
        TILLINGTON, HASLEMERE,
        waypoints=[UPPERTON, LURGASHALL],
        route_label="Route via Upperton / Lurgashall",
    )

    compare_routes(direct_result, scenic_result, alt_label="Route via Upperton / Lurgashall")

    print(f"\n{'=' * 60}")
    print("Specifically worth checking in the output above:")
    print("- Does the Upperton/Lurgashall route actually get routed")
    print("  through country lanes, or does OSRM quietly snap part of")
    print("  it back onto A-roads near Petworth?")
    print("- Does the Conservation Areas check find Upperton at all?")
    print("  (It's a real, named Conservation Area on this exact road —")
    print("  if it still shows 0 matches, that's evidence of a fetch/")
    print("  match bug, not just a Historic England coverage gap.)")
    print(f"{'=' * 60}")