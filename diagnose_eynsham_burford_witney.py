"""
Byway — Re-check: how big is the real Eynsham -> Burford via Witney
detour, using the CURRENT (patched) scoring pipeline?

WHY THIS EXISTS: milestone_1_score_route.py's __main__ block now runs
the Tillington/Haslemere test case (changed earlier this session), so
the original three-test-case comparison that first found this route's
"clear win on all 3 categories" no longer runs by default. This script
re-derives that one specific comparison via score_route()/
compare_routes(), without touching the main file's __main__ block
again — same pattern as diagnose_upperton_lurgashall.py.

Directly answers the open question from the discovery test: is the
modest result (penalty_scale_used=60, +0.1 min, +0.006 enjoyment) an
honest "there's no big hidden detour here" finding, or is it hiding a
real, substantial detour that the search still can't reach even at
16000s/km?

Run from the repo root: python diagnose_eynsham_burford_witney.py
Requires real internet access (Codespaces), not Claude's sandbox.
"""

from milestone_1_score_route import score_route, compare_routes

EYNSHAM = {"name": "Eynsham", "lat": 51.7808, "lon": -1.3745}
BURFORD = {"name": "Burford", "lat": 51.8071, "lon": -1.6368}
WITNEY = {"name": "Witney", "lat": 51.7864, "lon": -1.4848}

if __name__ == "__main__":
    direct_result = score_route(
        EYNSHAM, BURFORD, route_label="Direct (fastest) route"
    )

    scenic_result = score_route(
        EYNSHAM, BURFORD, waypoints=[WITNEY],
        route_label="Route via Witney (explicit waypoint)"
    )

    compare_routes(direct_result, scenic_result, alt_label="Route via Witney (explicit waypoint)")

    print(f"\n{'=' * 60}")
    print("Specifically worth checking in the output above:")
    print("- How big is 'Extra distance' / 'Extra time' actually? If")
    print("  it's a km or two and a couple of minutes, today's modest")
    print("  discovery result (penalty_scale_used=60, +0.1 min) is")
    print("  likely an honest, correct finding -- there's no big hidden")
    print("  detour being missed.")
    print("- If it's substantial (several km / many minutes) yet the")
    print("  score improvement is real, that confirms the structural")
    print("  ceiling flagged in 04_pathfinding.py's docstring: this")
    print("  route's own (distance x enjoyment-deficit) doesn't beat")
    print("  the direct route's, so no penalty_scale, however large,")
    print("  will ever select it -- the real next step would be a")
    print("  genuinely different algorithm, not a bigger number.")
    print(f"{'=' * 60}")