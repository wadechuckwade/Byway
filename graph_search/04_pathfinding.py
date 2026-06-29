"""
Byway — Milestone 1.5, Step 3: Find nearest nodes + run our own Dijkstra
============================================================================

What this does, in plain terms:
Three things, deliberately kept separate and simple:
1. Given an arbitrary lat/lon (e.g. "Tillington"), find the nearest
   actual graph node — since real-world coordinates essentially never
   land exactly on an OSM node.
2. Run a standard Dijkstra shortest-path search over the graph built
   in Step 2, using plain distance as the cost (no enjoyment scoring
   yet).
3. Search over that same Dijkstra search for the route with the BEST
   score reachable within a TIME BUDGET — see
   find_best_route_within_time_budget's docstring for why this exists
   and why it changed AGAIN in this version (the first version had its
   own hidden ceiling — see below).

WHY PLAIN DISTANCE FIRST: before ever introducing our own enjoyment
scoring into the cost function, we need to know our own pathfinding
implementation is CORRECT — that it finds genuinely valid, sensible
routes through the graph at all. If we jumped straight to enjoyment-
weighted search and got a weird result, we wouldn't know whether the
weirdness came from the scoring blend or from a bug in the search
itself. Proving plain-distance Dijkstra works first isolates that
risk before adding any complexity.
"""

import heapq
import math


def _haversine_distance_m(lat1, lon1, lat2, lon2):
    """Same formula used throughout this project, for consistency."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_nearest_node(target_lat, target_lon, node_coords):
    """
    Given a target lat/lon and the {node_id: (lat, lon)} dict from
    graph construction, find the single nearest actual graph node.

    This is a simple linear scan — fine for a bounded regional graph
    (thousands of nodes, not millions), and consistent with this
    project's existing "simple first, optimize later only if proven
    necessary" approach (e.g. proximity.py's distance checks are
    similarly unoptimized linear scans, and have been fast enough at
    every scale tested so far).

    Returns (nearest_node_id, distance_m).
    """
    best_node_id = None
    best_distance = float("inf")

    for node_id, (lat, lon) in node_coords.items():
        dist = _haversine_distance_m(target_lat, target_lon, lat, lon)
        if dist < best_distance:
            best_distance = dist
            best_node_id = node_id

    return best_node_id, best_distance


JUNCTION_PENALTY_SECONDS = 8
# Small fixed time cost applied whenever the route transitions from
# one OSM way to a DIFFERENT one — approximating the real delay of
# slowing down, checking, and pulling away at a junction/give-way.
# WHY THIS EXISTS: real-route testing (Tillington -> Haslemere) found
# our estimated time (15.9 min) notably faster than Google's (~25
# min) for a route confirmed correct. Research confirmed this is a
# known, structural limitation of static/traffic-oblivious routing
# engines generally (not just our implementation) — they have no
# inherent concept of junction delay, which adds up significantly on
# a route with many short way segments (this route has 59 distinct
# ways). 8 seconds is a deliberately modest, round-number starting
# assumption — not derived from a specific study — flagged as
# tuneable, same spirit as every other first-pass weight/threshold in
# this project.
#
# Note: this deliberately only counts transitions between DIFFERENT
# way_ids, not every single graph edge — most ways are subdivided
# into many small edges internally (e.g. a long A-road has dozens of
# edges along its length), and penalising every edge would wildly
# over-penalise long straight roads that have no real junctions along
# their interior points at all.


ENJOYMENT_PENALTY_SECONDS_PER_KM = 90
# Scale factor used ONLY when a caller uses the older blend_factor
# mechanism directly (see dijkstra_shortest_path) — kept for backward
# compatibility with existing tests/callers. find_best_route_within_
# time_budget() no longer uses this at all; see its docstring and
# DEFAULT_PENALTY_SCALE_CANDIDATES_S_PER_KM below for why.
#
# REPLACES an earlier multiplicative approach (cost = time * (1 -
# blend*enjoyment)) that was tested against the real Tillington ->
# Haslemere graph and found to be MECHANICALLY TOO WEAK to ever
# produce real route divergence — see git history / decisions log for
# the full numbers. This additive-per-km version fixed that specific
# problem, but introduced a DIFFERENT one, discovered this session —
# see below.
#
# THE CEILING PROBLEM (found this session, real-route testing on
# Eynsham -> Burford via Witney): the first version of
# find_best_route_within_time_budget() swept blend_factor from 0 to
# MAX_BLEND_FACTOR (1.0), meaning the maximum possible "credit" any
# detour could ever receive was capped at
# MAX_BLEND_FACTOR * ENJOYMENT_PENALTY_SECONDS_PER_KM = 1.0 * 90 =
# 90 seconds/km — REGARDLESS of how generous the time budget was. This
# is the exact same number already proven too weak for Tillington/
# Haslemere (needed 900-1850s/km there). Sweeping blend_factor
# couldn't ever escape that ceiling, because blend_factor only ever
# multiplies this ONE fixed constant. Confirmed analytically: a
# constructed test case (Test 14 below) shows a real, findable detour
# that needs ~294s/km of credit to be selected — unreachable at
# blend_factor<=1.0, found immediately once the sweep covers higher
# values directly (Test 15).
#
# FIX: find_best_route_within_time_budget() now sweeps
# penalty_scale_s_per_km DIRECTLY (see DEFAULT_PENALTY_SCALE_
# CANDIDATES_S_PER_KM below), decoupled from blend_factor and this
# constant entirely. This constant and blend_factor remain here,
# unchanged, for any caller that wants the older fixed-multiplier
# behaviour directly.


MAX_BLEND_FACTOR = 1.0
# No longer needs a sub-1.0 safety cap: unlike the old multiplicative
# formula (where blend_factor=1.0 combined with enjoyment=1.0 made a
# road literally FREE), the additive formula can never make a road's
# cost negative or zero — its time_s floor is always there
# regardless of blend_factor or enjoyment, only the SIZE of the added
# penalty changes. Kept as a named constant (rather than removed
# entirely) so the safety-capping mechanism remains in the code and
# easy to reintroduce if a future formula change reopens this risk.
# NOTE: this only bounds the OLD blend_factor mechanism — it has no
# effect on find_best_route_within_time_budget's direct penalty_scale
# sweep, which is the whole point of the fix above.


def dijkstra_shortest_path(graph, start_node_id, end_node_id,
                            enjoyment_scores=None, blend_factor=0.0,
                            penalty_scale_s_per_km=None):
    """
    Standard Dijkstra shortest-path search over the graph built in
    Step 2, using TIME as the cost (not raw distance) — see
    02_build_graph.py's module docstring for why this matters: plain-
    distance search was found (via real-route testing) to favour
    short-but-slow minor lanes over longer-but-faster A-roads, which
    is the wrong behaviour for a genuine "fastest route" baseline.

    Also applies a JUNCTION_PENALTY_SECONDS cost whenever the path
    changes from one way_id to a different one, approximating the
    real-world delay of navigating a junction — see constant's
    docstring above for why this was added.

    ENJOYMENT BLENDING — TWO WAYS TO SPECIFY THE SAME UNDERLYING
    PENALTY, kept both for backward compatibility:
        cost = time_s + penalty_scale * (distance_m / 1000) * (1 - enjoyment_score)
    where penalty_scale is EITHER:
      - penalty_scale_s_per_km, if explicitly provided (takes
        priority) — used directly, with NO cap. This is what
        find_best_route_within_time_budget() uses to sweep a wide
        range of real values; or
      - blend_factor * ENJOYMENT_PENALTY_SECONDS_PER_KM, otherwise —
        the older mechanism, still capped at
        MAX_BLEND_FACTOR * ENJOYMENT_PENALTY_SECONDS_PER_KM = 90s/km.
        Kept for any caller (and the self-tests below) that wants the
        simpler 0-1 dial rather than specifying raw seconds/km.

    graph shape: {node_id: [(neighbor_id, distance_m, time_s, way_info), ...]}

    Returns:
        path_node_ids: list of node IDs from start to end (inclusive),
                        or None if no path exists
        total_time_s: total REAL TIME of the path in seconds (NOT
                       discounted by enjoyment — this is always the
                       genuine time cost of the path actually chosen,
                       for honest reporting to the user), or None
        total_distance_m: total DISTANCE of the path in metres, or None
        path_way_info: list of way_info dicts, one per edge traversed
    """
    blend_factor = max(0.0, min(blend_factor, MAX_BLEND_FACTOR))
    enjoyment_scores = enjoyment_scores or {}

    # Standard Dijkstra using a min-heap priority queue. costs tracks
    # the best known EFFECTIVE cost to each node (what's actually
    # being minimised — may be enjoyment-discounted); real_times
    # tracks the genuine, undiscounted time along that same best path,
    # for honest reporting. distances_so_far tracks real distance,
    # also for reporting only. last_way_id tracks which way_id we
    # arrived at each node via, needed to detect junction transitions.
    costs = {start_node_id: 0}
    real_times = {start_node_id: 0}
    distances_so_far = {start_node_id: 0}
    last_way_id = {start_node_id: None}
    previous = {}
    previous_way_info = {}
    previous_edge_distance = {}
    # NEW: tracks the real distance of the SPECIFIC edge used to reach
    # each node — needed so the reconstructed path can report each
    # way_info entry's own edge length, for proper length-weighted
    # averaging downstream (see 08_three_route_system.py's
    # average_enjoyment_along_path). way_info objects are SHARED
    # across many edges of the same way, so we can't just store the
    # distance ON way_info itself — that would corrupt every other
    # edge referencing the same shared dict.
    visited = set()

    priority_queue = [(0, start_node_id)]

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == end_node_id:
            break

        for neighbor_id, edge_distance, edge_time, way_info in graph.get(current_node, []):
            if neighbor_id in visited:
                continue

            # Apply the junction penalty only when this edge's way_id
            # differs from the way we arrived at current_node on —
            # i.e. only at a genuine transition to a different road,
            # not for every internal edge along a single long way.
            current_way_id = last_way_id.get(current_node)
            edge_way_id = way_info["way_id"]
            penalty = JUNCTION_PENALTY_SECONDS if (
                current_way_id is not None and current_way_id != edge_way_id
            ) else 0

            real_edge_time = edge_time + penalty

            # ENJOYMENT BLENDING — see docstring above for the two
            # ways this can be specified. penalty_scale_s_per_km, if
            # given, takes priority and is used with NO cap.
            enjoyment = enjoyment_scores.get(edge_way_id, 0.0)
            distance_km = edge_distance / 1000
            effective_scale = (
                penalty_scale_s_per_km if penalty_scale_s_per_km is not None
                else blend_factor * ENJOYMENT_PENALTY_SECONDS_PER_KM
            )
            enjoyment_penalty = effective_scale * distance_km * (1 - enjoyment)
            effective_edge_cost = edge_time + enjoyment_penalty + penalty

            new_cost = current_cost + effective_edge_cost
            if new_cost < costs.get(neighbor_id, float("inf")):
                costs[neighbor_id] = new_cost
                real_times[neighbor_id] = real_times[current_node] + real_edge_time
                distances_so_far[neighbor_id] = distances_so_far[current_node] + edge_distance
                last_way_id[neighbor_id] = edge_way_id
                previous[neighbor_id] = current_node
                previous_way_info[neighbor_id] = way_info
                previous_edge_distance[neighbor_id] = edge_distance
                heapq.heappush(priority_queue, (new_cost, neighbor_id))

    if end_node_id not in costs:
        return None, None, None, None

    # Reconstruct the path by walking backwards from end to start.
    # Each way_info entry gets a SHALLOW COPY with "_edge_distance_m"
    # added, rather than mutating the shared original dict (the same
    # way_info object is referenced by many different edges of the
    # same way, each with its own distance — mutating the shared
    # object in place would corrupt all of them).
    path_node_ids = [end_node_id]
    path_way_info = []
    node = end_node_id
    while node != start_node_id:
        way_info_with_distance = dict(previous_way_info[node])
        way_info_with_distance["_edge_distance_m"] = previous_edge_distance[node]
        path_way_info.append(way_info_with_distance)
        node = previous[node]
        path_node_ids.append(node)
    path_node_ids.reverse()
    path_way_info.reverse()

    return path_node_ids, real_times[end_node_id], distances_so_far[end_node_id], path_way_info


def _average_enjoyment_for_path(way_info_list, enjoyment_scores):
    """
    Average enjoyment across the UNIQUE ways used in a path (length-
    unweighted for now — same simple first pass used in
    08_three_route_system.py's average_enjoyment_along_path, kept
    consistent rather than introducing a second, slightly different
    metric for the same thing).
    """
    way_ids = set(w["way_id"] for w in way_info_list)
    scores = [enjoyment_scores.get(wid, {}).get("enjoyment_score", 0.0)
              if isinstance(enjoyment_scores.get(wid), dict)
              else enjoyment_scores.get(wid, 0.0)
              for wid in way_ids]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


DEFAULT_MAX_TIME_MULTIPLIER = 1.5
# Fallback used only if a caller asks for a time-budget search without
# specifying either a multiplier or an absolute extra-time preference.
# Not meant to be a carefully-tuned constant — it's a reasonable
# default ("up to 50% longer") for when no real user preference exists
# yet.

DEFAULT_PENALTY_SCALE_CANDIDATES_S_PER_KM = [
    0, 30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000,
]
# NEW — candidate penalty-per-km values find_best_route_within_time_
# budget() sweeps directly, REPLACING the old blend_factor x
# ENJOYMENT_PENALTY_SECONDS_PER_KM approach, which had a hidden,
# unintentional ceiling of 90s/km baked in (see
# ENJOYMENT_PENALTY_SECONDS_PER_KM's docstring above for the full
# story — that ceiling is exactly what caused a real, known-better
# detour on the Eynsham/Burford test to never be found, no matter how
# generous the time budget was).
#
# Roughly-doubling spacing covers several orders of magnitude cheaply:
# real-route testing has now seen required scales ranging from ~90
# (this project's original default) up to 900-1850 (Tillington/
# Haslemere's minor-lane detour) — and this session's analysis showed
# the Eynsham/Burford Witney detour likely needs somewhere around
# 90-300, i.e. BETWEEN two values this project had separately used as
# "the" constant at different points, which is itself evidence no
# single fixed value was ever going to be right. 16000 is a
# deliberately generous top end, comfortably beyond anything seen so
# far, with real margin.
#
# HONEST LIMITATION, not yet hit in practice but worth stating
# precisely: this is a SWEEP over a LINEAR cost combination
# (time + scale*distance*deficit), not a true resource-constrained
# search. Algebraically, a slower route can only ever be preferred
# over a faster one, AT ANY scale, if the slower route's own
# (distance_km * enjoyment_deficit) product is SMALLER than the
# faster route's. If a real "nicer" detour is long enough that this
# isn't true even with genuinely better enjoyment, NO scale value —
# however large — will ever select it. Widening this candidate list
# fixes "the ceiling was too low to reach the right answer"; it does
# NOT fix that deeper case. If a route already confirmed better by
# Milestone 1's direct scoring still isn't found even with this much
# wider sweep, that's the signal this deeper limitation is the real
# blocker, and the correct next step is a genuinely different
# algorithm (a resource-constrained / Pareto-frontier search treating
# time and enjoyment as two separate dimensions, rather than
# collapsing them into one linear combination) — flagged here rather
# than built speculatively before confirming it's actually needed.


def time_budget_multiplier_from_preference(direct_time_s, extra_time_minutes=None,
                                            max_time_multiplier=None):
    """
    Converts a user-facing journey preference into the single internal
    multiplier find_best_route_within_time_budget() actually uses, so
    the search function itself only ever deals with one
    representation, while callers (eventually: a real "this specific
    journey" UI) can express the preference in whichever unit makes
    sense to a person.

    Supports two ways someone might naturally express their patience
    for a detour:
      - extra_time_minutes: "I've got an extra 20 minutes" — absolute,
        probably the more natural unit for a person to actually type
        or set with a slider.
      - max_time_multiplier: "I'm willing to go up to 50% longer" —
        relative, the form the search itself uses internally.
    Provide at most one; if neither is given, falls back to
    DEFAULT_MAX_TIME_MULTIPLIER. If direct_time_s is 0 or missing,
    extra_time_minutes can't be converted to a ratio — falls back to
    the multiplier (or default) instead of dividing by zero.
    """
    if extra_time_minutes is not None and direct_time_s:
        return 1.0 + (extra_time_minutes * 60) / direct_time_s
    if max_time_multiplier is not None:
        return max_time_multiplier
    return DEFAULT_MAX_TIME_MULTIPLIER


def find_best_route_within_time_budget(graph, start_node_id, end_node_id,
                                        enjoyment_scores, direct_time_s,
                                        max_time_multiplier=None,
                                        extra_time_minutes=None,
                                        penalty_scale_candidates=None):
    """
    SUPERSEDED for "find the best AVERAGE enjoyment route within a
    budget" — see find_best_average_enjoyment_route() below, which
    replaces this for that purpose. Real-route testing (Eynsham ->
    Burford via Witney) proved this function has a genuine structural
    blind spot, not just a calibration gap: this is a SWEEP over a
    LINEAR cost combination (time + scale*distance*deficit). A slower
    route can only ever be preferred over a faster one, AT ANY scale,
    if the slower route's own (distance_km * enjoyment_deficit) is
    SMALLER than the faster route's. The real Witney detour's own
    product (12.1, including the final stretch) was LARGER than the
    direct route's (10.1) — meaning no scale, however large, could
    ever have selected it. Confirmed by widening the sweep to 16000s/km
    (180x the original ceiling) with no change in result, then
    confirmed analytically against the real numbers.

    Kept in place, unchanged, for any caller that specifically wants
    this sweep-based approach (e.g. for comparison/regression testing
    against the new function) — search over penalty_scale_s_per_km
    (NOT blend_factor — see ENJOYMENT_PENALTY_SECONDS_PER_KM's and
    DEFAULT_PENALTY_SCALE_CANDIDATES_S_PER_KM's docstrings above for
    why that changed) to find the route with the HIGHEST average
    enjoyment whose REAL (undiscounted) time stays within a time
    budget.

    WHY A DIRECT PENALTY-SCALE SWEEP, NOT blend_factor: the previous
    version of this function swept blend_factor 0 to 1.0, which can
    only ever reach a maximum effective penalty of
    MAX_BLEND_FACTOR * ENJOYMENT_PENALTY_SECONDS_PER_KM = 90s/km — a
    hidden ceiling that meant a real, known-better detour (Eynsham/
    Burford via Witney, already confirmed better by Milestone 1's
    direct scoring) could never be found, no matter how generous the
    time budget was. Sweeping penalty_scale_s_per_km directly across
    DEFAULT_PENALTY_SCALE_CANDIDATES_S_PER_KM removes that ceiling.

    This also matches how a real user actually thinks about the
    tradeoff: "show me the best score reachable within however much
    extra time I'm willing to spend" — the user owns the time/score
    tradeoff, the app's job is to find the best score actually
    achievable at each time budget, not to silently decide the
    tradeoff for them (or, as it turns out, to silently under-deliver
    because of an internal cap the user never knew existed).

    direct_time_s: the real time of the FASTEST route (blend_factor=0
    / penalty_scale=0), used as the baseline the budget is measured
    against. Pass in the already-computed direct-route time rather
    than recomputing it here, since callers already have it.

    max_time_multiplier / extra_time_minutes: see
    time_budget_multiplier_from_preference() above — exactly one
    should be provided; this function converts via that helper so
    every caller goes through the same single conversion path. This
    is the intended seam for a future "this specific journey"
    preference input.

    penalty_scale_candidates: override the default sweep list — e.g.
    pass a narrower/finer list if profiling shows the default sweep
    is too slow on a much larger graph, or a wider one if a route is
    suspected to need even more than the current default's top end.

    Returns a dict:
        {"path": [...], "real_time_s": ..., "real_distance_m": ...,
         "way_info": [...], "penalty_scale_used": float,
         "avg_enjoyment": float, "within_budget": bool}
    or None if no path exists between start_node_id and end_node_id at
    all (distinct from "no candidate fit the budget," which instead
    falls back to the fastest available candidate — see within_budget).
    """
    multiplier = time_budget_multiplier_from_preference(
        direct_time_s, extra_time_minutes=extra_time_minutes,
        max_time_multiplier=max_time_multiplier,
    )
    time_budget_s = direct_time_s * multiplier

    scales_to_try = penalty_scale_candidates or DEFAULT_PENALTY_SCALE_CANDIDATES_S_PER_KM

    candidates = []
    for scale in scales_to_try:
        path, real_time_s, real_distance_m, way_info = dijkstra_shortest_path(
            graph, start_node_id, end_node_id,
            enjoyment_scores=enjoyment_scores, penalty_scale_s_per_km=scale,
        )
        if path is None:
            continue
        avg_enjoyment = _average_enjoyment_for_path(way_info, enjoyment_scores)
        candidates.append({
            "penalty_scale_used": scale,
            "path": path,
            "real_time_s": real_time_s,
            "real_distance_m": real_distance_m,
            "way_info": way_info,
            "avg_enjoyment": avg_enjoyment,
        })

    if not candidates:
        return None

    within_budget = [c for c in candidates if c["real_time_s"] <= time_budget_s]

    if within_budget:
        # Among everything that fits the budget, take the one with the
        # HIGHEST enjoyment. Ties broken by lowest penalty_scale (the
        # simplest route that achieves that enjoyment level), purely
        # for determinism — not a meaningful preference either way.
        best = max(within_budget, key=lambda c: (c["avg_enjoyment"], -c["penalty_scale_used"]))
        best["within_budget"] = True
    else:
        # No candidate fits — shouldn't normally happen, since
        # penalty_scale=0 (the fastest possible route) trivially has
        # real_time_s == direct_time_s, which always fits any budget
        # with multiplier >= 1.0. Kept as an explicit, honest fallback
        # (closest-to-budget candidate) rather than crashing or
        # silently returning something arbitrary.
        best = min(candidates, key=lambda c: c["real_time_s"])
        best["within_budget"] = False

    return best


def _add_pareto_label(label_list, new_time, new_value):
    """
    Try to insert (new_time, new_value) into label_list, maintaining
    only NON-DOMINATED labels — (t1, v1) dominates (t2, v2) if
    t1 <= t2 AND v1 >= v2 (reaching the same point no slower AND no
    less valuable). This is the standard pruning rule for Pareto-
    optimal label-setting/multi-criteria shortest-path algorithms: a
    dominated label can never lead to a better final result than the
    label dominating it, via the same or better continuation, so it's
    always safe to discard.

    Returns True if the label was actually added (i.e. not itself
    dominated by something already present); mutates label_list in
    place, removing anything the new label itself dominates.
    """
    for (lt, lv) in label_list:
        if lt <= new_time and lv >= new_value:
            return False
    label_list[:] = [
        (lt, lv) for (lt, lv) in label_list
        if not (new_time <= lt and new_value >= lv)
    ]
    label_list.append((new_time, new_value))
    return True


def _pareto_search_within_time_budget(graph, start_node_id, end_node_id,
                                       edge_weight_fn, time_budget_s,
                                       max_labels_per_node=50):
    """
    NEW — exact (up to the max_labels_per_node safety cap) multi-
    criteria search: finds the path from start to end MAXIMISING total
    edge_weight_fn-derived value, subject to a HARD constraint that
    real time used never exceeds time_budget_s.

    WHY THIS EXISTS, AND WHY IT'S A GENUINELY DIFFERENT ALGORITHM from
    dijkstra_shortest_path / find_best_route_within_time_budget above:
    those collapse time and enjoyment into ONE linear cost
    (time + scale*deficit) and minimise it — which, as proven on the
    real Eynsham/Burford case (see find_best_route_within_time_budget's
    docstring), can structurally fail to find a real, better route no
    matter how the scale is tuned. This function instead tracks time
    as a genuine HARD CONSTRAINT (not folded into the cost at all) and
    treats "value" (whatever edge_weight_fn computes) as the thing to
    maximise within that constraint — a real resource-constrained
    shortest path, not a linear approximation of one.

    HOW IT WORKS: each node maintains a small set of non-dominated
    (time_used, value_so_far) labels — Pareto-optimal tradeoffs between
    "how much time has this path used" and "how much value has it
    earned." Labels are processed in order of increasing time (valid
    because time only increases along any path, the same property that
    makes plain Dijkstra correct). At the end, among all labels at the
    destination with time <= budget, the one with the highest value is
    the genuinely optimal answer — not an approximation.

    KNOWN SIMPLIFICATIONS, both deliberate and worth revisiting if they
    turn out to matter:
    - Does NOT apply JUNCTION_PENALTY_SECONDS (would require tracking
      last_way_id as part of every label, multiplying the state space
      by the number of distinct way_ids reachable at each node, for a
      secondary ~8-second effect). Real time reported here is therefore
      slightly optimistic versus dijkstra_shortest_path's junction-
      aware time.
    - max_labels_per_node caps how many Pareto-optimal tradeoffs are
      kept per node, pruning the weakest-by-value ones if exceeded.
      This is a practical safety valve against pathological label
      explosion on dense graphs, not a proven bound — real road graphs
      in this project's testing have stayed in the thousands of edges,
      not millions, so this hasn't been stress-tested at larger scale.
    - UNTESTED ON REAL GRAPHS by me — verified here only against small
      synthetic graphs (see self-tests below), since this sandbox has
      no network access to run it against real Overpass/elevation
      data. Runtime on a real 50,000+ edge graph is genuinely unknown
      until profiled for real.

    edge_weight_fn(distance_m, time_s, way_info) -> float: computes
    the value contributed by traversing one edge.

    Returns: (best_value, real_time_s, real_distance_m, path_node_ids,
    path_way_info), or (None, None, None, None, None) if no path
    exists within the budget at all.
    """
    import heapq

    labels = {start_node_id: [(0.0, 0.0)]}
    parent = {}  # (node, time, value) -> (prev_node, prev_time, prev_value, way_info, edge_distance_m)
    pq = [(0.0, 0.0, start_node_id)]
    settled = set()

    while pq:
        t, v, node = heapq.heappop(pq)

        # PERFORMANCE FIX: a label can be pushed onto the queue, then
        # later DOMINATED and removed from labels[node] by an even
        # better label discovered via a different path, before this
        # one is ever popped. Without this check, the search would
        # still fully expand the now-useless stale label anyway — the
        # `settled` check below only prevents re-processing the exact
        # same (node, t, v) twice, it does nothing to stop a label
        # that's already been proven dominated from being expanded
        # once. On a real ~50,000-edge graph this caused the search to
        # keep re-exploring from labels already known to be pointless,
        # compounding combinatorially — almost certainly the actual
        # cause of a real run taking 10+ minutes without finishing.
        # Skipping stale labels here restores the algorithm to
        # expanding each genuinely-useful label once, which is what
        # makes Pareto label-setting tractable in practice at all.
        if (t, v) not in labels.get(node, []):
            continue

        key = (node, t, v)
        if key in settled:
            continue
        settled.add(key)

        if node == end_node_id:
            continue  # nothing further to gain by expanding past the destination

        for neighbor_id, edge_distance, edge_time, way_info in graph.get(node, []):
            new_t = t + edge_time
            if new_t > time_budget_s:
                continue
            new_v = v + edge_weight_fn(edge_distance, edge_time, way_info)

            neighbor_labels = labels.setdefault(neighbor_id, [])
            if _add_pareto_label(neighbor_labels, new_t, new_v):
                if len(neighbor_labels) > max_labels_per_node:
                    neighbor_labels.sort(key=lambda lab: -lab[1])
                    del neighbor_labels[max_labels_per_node:]
                parent[(neighbor_id, new_t, new_v)] = (node, t, v, way_info, edge_distance)
                heapq.heappush(pq, (new_t, new_v, neighbor_id))

    if end_node_id not in labels or not labels[end_node_id]:
        return None, None, None, None, None

    best_t, best_v = max(labels[end_node_id], key=lambda lab: lab[1])

    # Reconstruct the path by walking backwards via the parent chain.
    path_node_ids = [end_node_id]
    path_way_info = []
    real_distance_m = 0.0
    cur = (end_node_id, best_t, best_v)
    while not (cur[0] == start_node_id and cur[1] == 0.0 and cur[2] == 0.0):
        prev_node, prev_t, prev_v, way_info, edge_distance = parent[cur]
        path_way_info.append(way_info)
        real_distance_m += edge_distance
        path_node_ids.append(prev_node)
        cur = (prev_node, prev_t, prev_v)
    path_node_ids.reverse()
    path_way_info.reverse()

    return best_v, best_t, real_distance_m, path_node_ids, path_way_info


def find_best_average_enjoyment_route(graph, start_node_id, end_node_id,
                                       enjoyment_scores, time_budget_s,
                                       lambda_iterations=14,
                                       max_labels_per_node=50):
    """
    SUPERSEDED for real-sized graphs — confirmed via direct stress
    testing (not guesswork): on synthetic grid graphs shaped like a
    real road network (low-degree, many genuinely non-dominated routes
    between distant points), this took 1.2s at 100 nodes, 23s at 400
    nodes, 144s at 900 nodes — clearly superlinear, and this project's
    real graphs run 26,000+ nodes. The exact Pareto-frontier approach
    is mathematically correct but has a known, real property on grid-
    like topology: the set of non-dominated (time, enjoyment) routes
    between two distant points can grow explosively, which is exactly
    what happened here. This isn't a bug to patch — it's the wrong
    algorithm for a graph this size in Python.

    Replaced by find_best_route_via_candidates() below, which trades
    "provably optimal" for "bounded, fast, and targeted at where a
    good route is actually likely to be" — see its docstring.

    Kept here, unchanged, for small graphs or for anyone who
    specifically wants the exact (if slow) answer to compare against.

    HOW: this is the classic "maximum mean weight path subject to a
    resource constraint" problem, solved via the standard, mathematically
    exact technique — parametric (binary) search on the target average
    enjoyment λ. For a candidate λ, define each edge's value as
    (enjoyment - λ) * distance_km, and ask _pareto_search_within_time_
    budget(): is there a path within the time budget whose TOTAL value
    is >= 0? If yes, a path exists with average enjoyment >= λ within
    budget, so λ is achievable — search higher. If no, search lower.
    Because "is λ achievable" is monotonic (harder to satisfy as λ
    rises), binary search converges to the true best achievable average
    within lambda_iterations steps — this is the textbook-correct
    method, not a heuristic.

    WHY THIS REPLACES find_best_route_within_time_budget(): that
    function's sweep-the-scale approach has a proven structural blind
    spot (see its docstring) — it can fail to find a real, better route
    no matter how the scale constant is tuned, whenever the better
    route's own (distance x deficit) happens to exceed the faster
    route's. This function has no such blind spot: the time budget is
    a genuine hard constraint, never traded off against anything, so
    a route either fits or it doesn't — there's no constant to get
    wrong.

    time_budget_s: a HARD cap on real time, typically computed as
    direct_time_s * some_multiplier (e.g. 1.2 for a modest "up to 20%
    longer" Compromise tier, 1.8 for a more generous Max Enjoyment
    tier) — see 08_three_route_system.py's JOURNEY_PREFERENCE_DEFAULTS
    for where these tiers are actually set, and
    time_budget_multiplier_from_preference() above for converting a
    user preference (either a multiplier or an absolute extra-minutes
    value) into this single number. There is deliberately NO continuous
    per-minute-over-budget penalty here — a route either fits within
    the tier's cap or it doesn't; "how much of a detour is too much"
    is controlled entirely by WHICH TIER'S cap is being searched
    against, not by a separate penalty curve layered on top.

    Returns a dict (same shape as find_best_route_within_time_budget's,
    for drop-in compatibility with existing callers):
        {"path": [...], "real_time_s": ..., "real_distance_m": ...,
         "way_info": [...], "lambda_used": float, "avg_enjoyment": float,
         "within_budget": bool}
    or None if no path exists between start and end within the budget
    at all.
    """
    def _make_weight_fn(lam):
        def weight_fn(edge_distance_m, edge_time_s, way_info):
            enjoyment = enjoyment_scores.get(way_info["way_id"], 0.0)
            return (enjoyment - lam) * (edge_distance_m / 1000)
        return weight_fn

    lo, hi = 0.0, 1.0

    # Seed with lambda=0 BEFORE bisecting: since enjoyment scores are
    # always >= 0, average enjoyment >= 0 is trivially true for ANY
    # path that exists at all within the budget — but bisection's
    # midpoints (0.5, 0.25, 0.125, ...) never test exactly 0 itself,
    # so without this seed, a graph where the only feasible route
    # happens to have exactly zero average enjoyment would never see
    # ANY successful test and would wrongly return None, even though
    # a perfectly valid (if unenjoyable) route genuinely exists within
    # budget. This also doubles as the "does ANY path exist within
    # budget at all" check.
    baseline_value, baseline_time, baseline_distance, baseline_path, baseline_way_info = (
        _pareto_search_within_time_budget(
            graph, start_node_id, end_node_id, _make_weight_fn(0.0), time_budget_s,
            max_labels_per_node=max_labels_per_node,
        )
    )
    if baseline_path is None:
        return None  # no path exists within the budget at all

    best_found = (baseline_path, baseline_way_info, baseline_time, baseline_distance)

    for _ in range(lambda_iterations):
        mid = (lo + hi) / 2
        value, real_time_s, real_distance_m, path_node_ids, path_way_info = _pareto_search_within_time_budget(
            graph, start_node_id, end_node_id, _make_weight_fn(mid), time_budget_s,
            max_labels_per_node=max_labels_per_node,
        )
        if value is not None and value >= -1e-9:
            lo = mid
            best_found = (path_node_ids, path_way_info, real_time_s, real_distance_m)
        else:
            hi = mid

    path_node_ids, path_way_info, real_time_s, real_distance_m = best_found
    avg_enjoyment = _average_enjoyment_for_path(path_way_info, enjoyment_scores)

    return {
        "path": path_node_ids,
        "way_info": path_way_info,
        "real_time_s": real_time_s,
        "real_distance_m": real_distance_m,
        "avg_enjoyment": avg_enjoyment,
        "lambda_used": round(lo, 4),
        "penalty_scale_used": None,  # not applicable to this search; kept for dict-shape compatibility
        "within_budget": True,  # enforced as a hard constraint by construction
    }


def _representative_nodes_per_way(graph):
    """
    One-time O(E) scan building {way_id: {node_ids touching this way}}.
    Collects every distinct node touching each way (not just the
    first one encountered) so the caller can pick whichever one isn't
    the start/end node, rather than being stuck if the first candidate
    happens to collide — e.g. a single-edge way directly off the start
    node would otherwise have its only recorded node BE the start
    node, silently skipping a perfectly good candidate.
    """
    rep = {}
    for node_id, edges in graph.items():
        for neighbor_id, _, _, way_info in edges:
            node_set = rep.setdefault(way_info["way_id"], set())
            node_set.add(node_id)
            node_set.add(neighbor_id)
    return rep


def _route_way_id_set(way_info_list):
    return set(w["way_id"] for w in way_info_list)


def _jaccard_similarity(way_info_a, way_info_b):
    """
    Fraction of roads shared between two routes, by way_id —
    0.0 = completely different roads, 1.0 = identical road sets. Used
    to GUARANTEE genuine diversity between kept candidates: no two
    candidates this module keeps will ever share more than
    similarity_threshold of their roads (see
    generate_candidate_routes() below).
    """
    set_a = _route_way_id_set(way_info_a)
    set_b = _route_way_id_set(way_info_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _combine_path_legs(leg1, leg2):
    """
    Combine two consecutive dijkstra_shortest_path results (each a
    (path, time_s, distance_m, way_info) tuple) into one continuous
    route, for the start -> via_node -> end waypoint-forcing pattern
    below. Returns (None, None, None, None) if either leg has no path.
    """
    path1, time1, dist1, wi1 = leg1
    path2, time2, dist2, wi2 = leg2
    if path1 is None or path2 is None:
        return None, None, None, None
    # path1's last node IS path2's first node (the via-node) — drop
    # the duplicate when joining.
    combined_path = path1 + path2[1:]
    combined_way_info = wi1 + wi2
    return combined_path, time1 + time2, dist1 + dist2, combined_way_info


DEFAULT_CANDIDATE_PENALTY_SCALES_S_PER_KM = [0, 250, 1000, 4000, 16000]
# Smaller, coarser version of DEFAULT_PENALTY_SCALE_CANDIDATES_S_PER_KM
# used specifically by generate_candidate_routes() below — still spans
# the same overall range, just with fewer points, since each one costs
# a full Dijkstra call and real-world testing showed per-call cost on
# an actual ~53,000-edge road graph is higher than synthetic testing
# suggested (see find_three_routes's docstring in
# 08_three_route_system.py for the real numbers that motivated this).


WAYPOINT_LEG_PENALTY_SCALE_S_PER_KM = 1000
# NEW — direct feedback found waypoint-forced and feature-forced
# candidates producing "shaggy dog" detours that almost never won,
# often via major roads. Root cause, confirmed in the code itself:
# both sources' connecting legs (start->via, via->end) were computed
# with PLAIN dijkstra_shortest_path() calls — no enjoyment_scores, no
# penalty_scale — meaning pure-TIME shortest paths. On a real road
# network, the fastest way to reach a distant point is very often a
# trunk/A-road, which then gets hammered by the road-class multiplier
# once scored (trunk 0.35x, primary 0.55x) — the forced waypoint
# itself might be excellent, but the JOURNEY to and from it was never
# enjoyment-aware at all, only "caring" about the one forced point.
# Fixed: legs now use the SAME enjoyment-blended cost function as the
# blend-scale sweep (source 3) — a real, well-tested mechanism, not a
# new one. 1000 s/km is a middle value from that existing spectrum
# ([0, 250, 1000, 4000, 16000]) — strong enough to meaningfully avoid
# majors when a decent alternative exists, without being so extreme
# it forces an absurd detour just to dodge any major-road segment at
# all costs.


def generate_candidate_routes(graph, start_node_id, end_node_id, enjoyment_scores,
                               time_budget_s, num_waypoint_candidates=15,
                               penalty_scale_candidates=None,
                               similarity_threshold=0.85, verbose=False,
                               node_coords=None, target_features=None):
    """
    Generates a BOUNDED set of genuinely diverse candidate routes
    within the time budget, replacing find_best_average_enjoyment_
    route()'s exact-but-impractically-slow Pareto search (see that
    function's docstring for why it doesn't scale).

    FOUR SOURCES of candidates, each cheap (ordinary single-criterion
    Dijkstra calls — no label explosion possible, since each call only
    ever tracks ONE best path, not a frontier of them):

    1. The plain fastest route — always included as the baseline.

    2. WAYPOINT-FORCED candidates: take the top num_waypoint_candidates
       highest-enjoyment WAYS anywhere in the graph, and for each,
       force a route through it (start -> a node on that way -> end).
       This directly targets the most likely location of a genuinely
       good route. This is the automated equivalent of what Milestone
       1 did by hand (forcing through Witney, Upperton, Mam Tor) — now
       driven by the graph's own scores instead of personal knowledge
       of the area.

       CHANGED — the connecting legs (start->via, via->end) now use
       enjoyment-BLENDED Dijkstra (see WAYPOINT_LEG_PENALTY_SCALE_S_
       PER_KM), not pure time. Direct feedback found these candidates
       producing "shaggy dog" detours via major roads that almost
       never won — the root cause was that the legs were computed by
       pure-TIME shortest path, with zero regard for enjoyment, so the
       fastest way to reach a forced waypoint very often defaulted to
       a trunk/A-road regardless of how excellent the waypoint itself
       was. The forced point was enjoyment-aware; the journey to reach
       it wasn't. Now both are.

    3. PENALTY-SCALE SWEEP candidates: the existing blend sweep
       directly from start to end, no waypoint forcing — catches broad
       route shifts that aren't anchored on any single standout road.

    4. FEATURE-FORCED candidates (NEW): given target_features (a list
       of {"name", "lat", "lon"} point-features — e.g. the highest-
       rated ScenicOrNot spots, Conservation Areas, named historic
       sites), forces a route through the graph node NEAREST to each
       one directly, rather than only relying on the top-N WAYS
       ranking. WHY THIS MATTERS: a way can score only moderately on
       its own blended formula (e.g. mediocre curvature) even when it
       happens to be the closest road to something genuinely
       excellent — that way would never make the top-N cut on its own
       score, so source (2) alone would never find it. Going directly
       to the best individual FEATURES and finding whichever real road
       is nearest to each one targets them more reliably than hoping
       they happen to lift their road's overall score enough.
       Requires node_coords (for find_nearest_node) — if not provided,
       or target_features is empty/None, this source is simply
       skipped, no error, fully backward compatible.

    PERFORMANCE NOTE: each new source item costs exactly 2 Dijkstra
    calls (same as waypoint-forcing), so keep target_features bounded
    (e.g. 10-20 per category) rather than passing every single rated
    spot in a bbox — real-route testing found per-call cost on actual
    road graphs higher than synthetic tests suggested, so this adds up
    faster than it looks on a small example.

    verbose: if True, prints elapsed time for each phase.

    DIVERSITY GUARANTEE: every kept candidate is checked against every
    OTHER kept candidate via Jaccard similarity (shared fraction of
    roads, by way_id) — if two candidates share MORE than
    similarity_threshold of their roads, only the better-scoring one
    is kept.

    HONEST LIMITATION: this is NOT guaranteed to find the mathematically
    optimal route. It's deliberately targeted (test the graph's actual
    best material AND its best individual features) rather than
    exhaustive.

    Returns a list of dicts: {"path", "way_info", "real_time_s",
    "real_distance_m", "avg_enjoyment", "source"} — only candidates
    that fit within time_budget_s are included at all.
    """
    import time as _time
    candidates = []

    def _try_add(path, way_info, real_time_s, real_distance_m, source):
        if path is None or real_time_s is None:
            return
        if real_time_s > time_budget_s:
            return  # hard constraint, enforced immediately — no exceptions
        avg_enjoyment = _average_enjoyment_for_path(way_info, enjoyment_scores)
        candidate = {
            "path": path, "way_info": way_info, "real_time_s": real_time_s,
            "real_distance_m": real_distance_m, "avg_enjoyment": avg_enjoyment,
            "source": source,
        }
        for i, existing in enumerate(candidates):
            if _jaccard_similarity(existing["way_info"], way_info) >= similarity_threshold:
                if avg_enjoyment > existing["avg_enjoyment"]:
                    candidates[i] = candidate
                return
        candidates.append(candidate)

    t_phase = _time.time()

    # 1. Plain fastest route.
    path, t, d, wi = dijkstra_shortest_path(graph, start_node_id, end_node_id)
    _try_add(path, wi, t, d, "direct")
    if verbose:
        print(f"    [timing] direct route: {round(_time.time() - t_phase, 2)}s")
        t_phase = _time.time()

    # 2. Waypoint-forced candidates from the graph's own best material.
    rep_nodes = _representative_nodes_per_way(graph)
    top_way_ids = sorted(enjoyment_scores.items(), key=lambda kv: -kv[1])[:num_waypoint_candidates]
    for way_id, score in top_way_ids:
        candidate_nodes = rep_nodes.get(way_id, set())
        via_node = next((n for n in candidate_nodes if n != start_node_id and n != end_node_id), None)
        if via_node is None:
            continue
        leg1 = dijkstra_shortest_path(
            graph, start_node_id, via_node,
            enjoyment_scores=enjoyment_scores, penalty_scale_s_per_km=WAYPOINT_LEG_PENALTY_SCALE_S_PER_KM,
        )
        leg2 = dijkstra_shortest_path(
            graph, via_node, end_node_id,
            enjoyment_scores=enjoyment_scores, penalty_scale_s_per_km=WAYPOINT_LEG_PENALTY_SCALE_S_PER_KM,
        )
        combined_path, combined_time, combined_dist, combined_wi = _combine_path_legs(leg1, leg2)
        _try_add(combined_path, combined_wi, combined_time, combined_dist, f"via_way_{way_id}")
    if verbose:
        print(f"    [timing] {len(top_way_ids)} waypoint-forced candidates "
              f"({2 * len(top_way_ids)} Dijkstra calls): {round(_time.time() - t_phase, 2)}s")
        t_phase = _time.time()

    # 3. Penalty-scale sweep candidates (cheap, complementary).
    scales = penalty_scale_candidates or DEFAULT_CANDIDATE_PENALTY_SCALES_S_PER_KM
    for scale in scales:
        path, t, d, wi = dijkstra_shortest_path(
            graph, start_node_id, end_node_id, enjoyment_scores=enjoyment_scores,
            penalty_scale_s_per_km=scale,
        )
        _try_add(path, wi, t, d, f"blend_scale_{scale}")
    if verbose:
        print(f"    [timing] {len(scales)} blend-scale candidates: "
              f"{round(_time.time() - t_phase, 2)}s")
        t_phase = _time.time()

    # 4. Feature-forced candidates (NEW): target real point-features directly.
    if node_coords is not None and target_features:
        feature_count = 0
        for feature in target_features:
            via_node, _ = find_nearest_node(feature["lat"], feature["lon"], node_coords)
            if via_node is None or via_node in (start_node_id, end_node_id):
                continue
            leg1 = dijkstra_shortest_path(
                graph, start_node_id, via_node,
                enjoyment_scores=enjoyment_scores, penalty_scale_s_per_km=WAYPOINT_LEG_PENALTY_SCALE_S_PER_KM,
            )
            leg2 = dijkstra_shortest_path(
                graph, via_node, end_node_id,
                enjoyment_scores=enjoyment_scores, penalty_scale_s_per_km=WAYPOINT_LEG_PENALTY_SCALE_S_PER_KM,
            )
            combined_path, combined_time, combined_dist, combined_wi = _combine_path_legs(leg1, leg2)
            label = feature.get("source_label") or feature.get("name", "feature")
            _try_add(combined_path, combined_wi, combined_time, combined_dist, f"via_feature_{label}")
            feature_count += 1
        if verbose:
            print(f"    [timing] {feature_count} feature-forced candidates "
                  f"({2 * feature_count} Dijkstra calls): {round(_time.time() - t_phase, 2)}s")

    return candidates


def select_best_candidate_within_budget(candidates, time_budget_s, min_time_s=None):
    """
    Cheap, NO-Dijkstra step: pick the best (highest avg_enjoyment)
    candidate from an ALREADY-GENERATED candidate list that fits a
    given time budget.

    WHY THIS EXISTS: lets multiple different time budgets (e.g.
    Compromise's modest tier and Max Enjoyment's more generous one)
    share ONE expensive generation pass instead of each calling
    generate_candidate_routes() separately. Real-route testing showed
    regenerating per tier roughly doubled total runtime for no real
    benefit — the same roads exist in the graph regardless of which
    tier's budget is being checked, so there's no reason to re-run all
    those Dijkstra calls twice. Generate ONCE using the WIDEST budget
    needed across all tiers, then filter down per tier with this cheap
    function.

    NEW — min_time_s (real-route testing found Compromise and Max
    Enjoyment silently picking the IDENTICAL candidate whenever the
    single best-average route happened to also fit the tighter
    budget — defeating the point of offering three genuinely different
    choices): if provided, only considers candidates with
    real_time_s STRICTLY GREATER than min_time_s. Pass the lower
    tier's own chosen real_time_s here when selecting a higher tier,
    so the higher tier is never allowed to silently collapse onto the
    lower one — see find_three_routes() in 08_three_route_system.py
    for the actual usage (Max Enjoyment is selected with
    min_time_s=compromise's real_time_s).

    Returns a dict shaped like find_best_route_via_candidates's return
    (with "within_budget" and "num_candidates_considered" added), or
    None if nothing in the candidate list fits this particular budget
    (including, if min_time_s is set, nothing that's ALSO genuinely
    longer than the lower tier).
    """
    fitting = [c for c in candidates if c["real_time_s"] <= time_budget_s]
    if min_time_s is not None:
        fitting = [c for c in fitting if c["real_time_s"] > min_time_s]
    if not fitting:
        return None
    best = dict(max(fitting, key=lambda c: c["avg_enjoyment"]))
    best["within_budget"] = True
    best["num_candidates_considered"] = len(fitting)
    return best


def find_best_route_via_candidates(graph, start_node_id, end_node_id, enjoyment_scores,
                                    time_budget_s, num_waypoint_candidates=15,
                                    penalty_scale_candidates=None,
                                    similarity_threshold=0.85, verbose=False,
                                    node_coords=None, target_features=None):
    """
    Single-budget convenience wrapper around generate_candidate_routes()
    + select_best_candidate_within_budget() — see both for the full
    design and honest limitations. For finding routes across MULTIPLE
    time budgets from the same graph (e.g. this project's Compromise +
    Max Enjoyment tiers), prefer calling generate_candidate_routes()
    ONCE at the widest needed budget and select_best_candidate_within_
    budget() per tier instead — see find_three_routes() in
    08_three_route_system.py for exactly that pattern.

    node_coords / target_features: passed straight through to
    generate_candidate_routes() — see its docstring (source 4).

    Returns None only if no path exists between start and end within
    the budget at all.
    """
    candidates = generate_candidate_routes(
        graph, start_node_id, end_node_id, enjoyment_scores, time_budget_s,
        num_waypoint_candidates=num_waypoint_candidates,
        penalty_scale_candidates=penalty_scale_candidates,
        similarity_threshold=similarity_threshold, verbose=verbose,
        node_coords=node_coords, target_features=target_features,
    )
    return select_best_candidate_within_budget(candidates, time_budget_s)


if __name__ == "__main__":
    # Self-tests using small fake graphs (no internet needed).

    print("--- Test 1: find_nearest_node picks the genuinely closest point ---")
    node_coords = {
        1: (51.0, -0.7),
        2: (51.01, -0.71),    # further away
        3: (51.001, -0.701),  # genuinely closest to our target
    }
    nearest_id, dist = find_nearest_node(51.0005, -0.7005, node_coords)
    print(f"Nearest node: {nearest_id}, distance: {round(dist, 1)}m")
    assert nearest_id == 3, f"Expected node 3 to be nearest, got {nearest_id}"
    print("PASSED\n")

    print("--- Test 2: Dijkstra finds the path with the shorter TIME (not necessarily shorter distance) ---")
    test_graph = {
        1: [(2, 100, 100, {"way_id": 1, "name": "Long Road A"}), (3, 10, 10, {"way_id": 2, "name": "Short Road A"})],
        2: [(4, 100, 100, {"way_id": 1, "name": "Long Road B"})],
        3: [(4, 10, 10, {"way_id": 2, "name": "Short Road B"})],
        4: [],
    }
    path, total_time, total_dist, way_info = dijkstra_shortest_path(test_graph, 1, 4)
    print(f"Path: {path}, total time: {total_time}s, total distance: {total_dist}m")
    assert path == [1, 3, 4], f"Expected path via node 3, got {path}"
    assert total_time == 20, f"Expected total time 20s (same way_id throughout, no junction penalty), got {total_time}"
    print("PASSED\n")

    print("--- Test 3: Dijkstra respects one-way restrictions (no edge back) ---")
    oneway_graph = {
        1: [(2, 50, 10, {"way_id": 1, "name": "One Way"})],
        2: [],
    }
    path2, time2, dist2, _ = dijkstra_shortest_path(oneway_graph, 2, 1)
    print(f"Path from 2 to 1 (should be impossible): {path2}")
    assert path2 is None, "Should not find a path against a one-way street"
    print("PASSED\n")

    print("--- Test 4: no path exists between disconnected components ---")
    disconnected_graph = {
        1: [(2, 50, 10, {"way_id": 1, "name": "Road A"})],
        2: [(1, 50, 10, {"way_id": 1, "name": "Road A"})],
        99: [],
    }
    path3, time3, dist3, _ = dijkstra_shortest_path(disconnected_graph, 1, 99)
    print(f"Path to disconnected node: {path3}")
    assert path3 is None
    print("PASSED\n")

    print("--- Test 5: time-based search picks a DIFFERENT (and correct) path than distance-based would ---")
    real_scenario_graph = {
        1: [
            (2, 5000, 212, {"way_id": 10, "name": "Fast A-road"}),
            (3, 3000, 432, {"way_id": 20, "name": "Slow minor lane"}),
        ],
        2: [(4, 100, 5, {"way_id": 11, "name": "final stretch A"})],
        3: [(4, 100, 5, {"way_id": 21, "name": "final stretch B"})],
        4: [],
    }
    path5, time5, dist5, way_info5 = dijkstra_shortest_path(real_scenario_graph, 1, 4)
    print(f"Path: {path5}, total time: {time5}s, total distance: {dist5}m")
    print(f"Roads used: {[w['name'] for w in way_info5]}")
    assert path5 == [1, 2, 4], (
        "BUG: time-based search should prefer the longer-but-faster "
        "route, not the shorter-but-slower one"
    )
    assert dist5 == 5100, "Distance should reflect the actual path taken (the longer one)"
    print("PASSED — correctly prefers the faster route despite it being longer in distance\n")

    print("--- Test 6: junction penalty is applied exactly once per way_id transition, not per edge ---")
    same_way_graph = {
        1: [(2, 100, 10, {"way_id": 1, "name": "Road A"})],
        2: [(3, 100, 10, {"way_id": 1, "name": "Road A"})],
        3: [(4, 100, 10, {"way_id": 1, "name": "Road A"})],
        4: [],
    }
    path6, time6, dist6, _ = dijkstra_shortest_path(same_way_graph, 1, 4)
    print(f"Same way_id throughout: time = {time6}s (3 edges x 10s, no junctions)")
    assert time6 == 30, f"Expected exactly 30s with zero junction penalty, got {time6}"
    print("PASSED — no penalty applied when staying on the same way\n")

    switching_way_graph = {
        1: [(2, 100, 10, {"way_id": 1, "name": "Road A"})],
        2: [(3, 100, 10, {"way_id": 2, "name": "Road B"})],
        3: [(4, 100, 10, {"way_id": 2, "name": "Road B"})],
        4: [],
    }
    path7, time7, dist7, _ = dijkstra_shortest_path(switching_way_graph, 1, 4)
    expected_time7 = 30 + JUNCTION_PENALTY_SECONDS
    print(f"One way_id transition: time = {time7}s (30s travel + {JUNCTION_PENALTY_SECONDS}s for the one junction)")
    assert time7 == expected_time7, f"Expected exactly {expected_time7}s (one penalty), got {time7}"
    print("PASSED — exactly one penalty applied for the one genuine transition\n")

    print("--- Test 8: with blend_factor=0, enjoyment scores should have ZERO effect (backward compatible) ---")
    enjoyment_test_graph = {
        1: [
            (2, 5000, 250, {"way_id": 100, "name": "Fast but dull"}),
            (3, 5000, 500, {"way_id": 200, "name": "Slow but lovely"}),
        ],
        2: [(4, 100, 5, {"way_id": 101, "name": "end A"})],
        3: [(4, 100, 5, {"way_id": 201, "name": "end B"})],
        4: [],
    }
    fake_enjoyment = {100: 0.0, 200: 1.0}
    path8, time8, dist8, _ = dijkstra_shortest_path(
        enjoyment_test_graph, 1, 4, enjoyment_scores=fake_enjoyment, blend_factor=0.0
    )
    print(f"blend_factor=0.0: path = {path8}")
    assert path8 == [1, 2, 4], "With blend_factor=0, should still pick the fast path regardless of enjoyment"
    print("PASSED — enjoyment correctly has zero effect when blend_factor is 0\n")

    print("--- Test 9: with a high blend_factor, the search should prefer the MORE ENJOYABLE path ---")
    path9, time9, dist9, way_info9 = dijkstra_shortest_path(
        enjoyment_test_graph, 1, 4, enjoyment_scores=fake_enjoyment, blend_factor=0.8
    )
    print(f"blend_factor=0.8: path = {path9}, roads used: {[w['name'] for w in way_info9]}")
    assert path9 == [1, 3, 4], (
        "With a high blend_factor, the search should now prefer the "
        "more enjoyable (even though slower) path"
    )
    print("PASSED — high blend_factor correctly favours the more enjoyable route\n")

    print("--- Test 10: blend_factor is silently capped, never reaching 'free road' territory ---")
    path10, time10, dist10, _ = dijkstra_shortest_path(
        enjoyment_test_graph, 1, 4, enjoyment_scores=fake_enjoyment, blend_factor=5.0
    )
    print(f"blend_factor=5.0 (way beyond cap): path = {path10}, real time = {time10}s")
    assert path10 == [1, 3, 4], "Should still prefer the enjoyable path, just capped, not broken"
    assert time10 is not None and time10 > 0, "Real time should still be a sensible positive number"
    print("PASSED — extreme blend_factor values are safely capped, not left uncontrolled\n")

    print("--- Test 11: time-budget search picks the FAST path when the budget is tight ---")
    direct_path, direct_time, _, _ = dijkstra_shortest_path(enjoyment_test_graph, 1, 4)
    result_tight = find_best_route_within_time_budget(
        enjoyment_test_graph, 1, 4, fake_enjoyment, direct_time_s=direct_time,
        max_time_multiplier=1.0,
    )
    print(f"Direct time: {direct_time}s. Tight-budget result: penalty_scale_used="
          f"{result_tight['penalty_scale_used']}, real_time_s={result_tight['real_time_s']}, "
          f"within_budget={result_tight['within_budget']}")
    assert result_tight["real_time_s"] == direct_time, (
        "With zero extra time tolerated, the result should be exactly "
        "the fast path's real time"
    )
    assert result_tight["within_budget"] is True, "The fast path itself should always fit its own time"
    print("PASSED — tight budget correctly falls back to the fast path\n")

    print("--- Test 12: time-budget search picks the ENJOYABLE path once the budget allows it ---")
    result_generous = find_best_route_within_time_budget(
        enjoyment_test_graph, 1, 4, fake_enjoyment, direct_time_s=direct_time,
        max_time_multiplier=2.5,
    )
    print(f"Generous-budget result: penalty_scale_used={result_generous['penalty_scale_used']}, "
          f"real_time_s={result_generous['real_time_s']}, avg_enjoyment={result_generous['avg_enjoyment']}, "
          f"within_budget={result_generous['within_budget']}")
    assert result_generous["path"] == [1, 3, 4], (
        "With a generous time budget, the search should find the more "
        "enjoyable (slower) path, not just the fastest one"
    )
    assert result_generous["within_budget"] is True
    print("PASSED — generous budget correctly finds the more enjoyable route\n")

    print("--- Test 13: extra_time_minutes preference converts correctly to the same result ---")
    result_minutes = find_best_route_within_time_budget(
        enjoyment_test_graph, 1, 4, fake_enjoyment, direct_time_s=direct_time,
        extra_time_minutes=10,
    )
    print(f"extra_time_minutes=10 result: penalty_scale_used={result_minutes['penalty_scale_used']}, "
          f"real_time_s={result_minutes['real_time_s']}, within_budget={result_minutes['within_budget']}")
    assert result_minutes["path"] == [1, 3, 4], (
        "10 extra minutes is far more than enough to afford this route's "
        "real detour cost — should pick the enjoyable path"
    )
    print("PASSED — extra_time_minutes preference converts and behaves correctly\n")

    print("--- Test 14 (NEW): the OLD blend_factor ceiling (90s/km max) FAILS to find a real, findable detour ---")
    # Constructed so the detour needs ~294s/km of credit to be
    # selected (worked out algebraically — see
    # ENJOYMENT_PENALTY_SECONDS_PER_KM's docstring): a detour that's
    # 500m longer, with only a MODEST enjoyment improvement (0.0 ->
    # 0.4, not a dramatic one), but costs 500s more real time. This is
    # deliberately NOT an extreme case — it's the kind of real,
    # plausible "longer but only somewhat nicer" tradeoff this project
    # has actually seen (Eynsham/Burford), used here in a controlled
    # form so the exact threshold can be verified precisely.
    ceiling_test_graph = {
        1: [
            (2, 5000, 250, {"way_id": 100, "name": "Fast but dull"}),
            (3, 5500, 750, {"way_id": 200, "name": "Slow but only modestly lovely"}),
        ],
        2: [(4, 100, 5, {"way_id": 101, "name": "end A"})],
        3: [(4, 100, 5, {"way_id": 201, "name": "end B"})],
        4: [],
    }
    ceiling_enjoyment = {100: 0.0, 200: 0.4}

    # Even at the OLD mechanism's absolute maximum (blend_factor=1.0,
    # i.e. the full 90s/km), the detour should NOT be selected.
    path_old_max, time_old_max, _, way_info_old_max = dijkstra_shortest_path(
        ceiling_test_graph, 1, 4, enjoyment_scores=ceiling_enjoyment, blend_factor=1.0,
    )
    print(f"OLD mechanism at its absolute max (blend_factor=1.0 = 90s/km): "
          f"path = {[w['name'] for w in way_info_old_max]}")
    assert path_old_max == [1, 2, 4], (
        "BUG IN TEST SETUP, not the code: this detour was constructed to need "
        "MORE than 90s/km — if the old ceiling already finds it, the test "
        "numbers need adjusting to actually demonstrate the limitation"
    )
    print("CONFIRMED — even at its absolute maximum, the old blend_factor "
          "mechanism cannot find this real, constructed detour\n")

    print("--- Test 15 (NEW): the NEW wide penalty_scale sweep FINDS that same detour ---")
    # Generous time budget (3.5x) so budget itself isn't the
    # constraint being tested here — purely testing whether the wider
    # SCALE range can reach the threshold this detour actually needs.
    direct_path_ceiling, direct_time_ceiling, _, _ = dijkstra_shortest_path(ceiling_test_graph, 1, 4)
    result_wide_sweep = find_best_route_within_time_budget(
        ceiling_test_graph, 1, 4, ceiling_enjoyment, direct_time_s=direct_time_ceiling,
        max_time_multiplier=3.5,
    )
    print(f"Direct time: {direct_time_ceiling}s. Wide-sweep result: "
          f"penalty_scale_used={result_wide_sweep['penalty_scale_used']}, "
          f"real_time_s={result_wide_sweep['real_time_s']}, "
          f"avg_enjoyment={result_wide_sweep['avg_enjoyment']}")
    assert result_wide_sweep["path"] == [1, 3, 4], (
        "The wider penalty_scale sweep should find this detour, which the "
        "old blend_factor-capped mechanism (Test 14) could not reach"
    )
    assert result_wide_sweep["penalty_scale_used"] > ENJOYMENT_PENALTY_SECONDS_PER_KM, (
        "The scale actually needed to find this route should exceed the "
        "old mechanism's absolute maximum (90s/km) — confirming this test "
        "genuinely demonstrates escaping the old ceiling, not coincidence"
    )
    print(f"PASSED — found the detour using penalty_scale={result_wide_sweep['penalty_scale_used']}s/km, "
          f"genuinely beyond the old mechanism's {ENJOYMENT_PENALTY_SECONDS_PER_KM}s/km ceiling\n")

    print("--- Test 16 (NEW): a case PROVEN impossible for the OLD method at ANY scale ---")
    # Mirrors the real Eynsham/Burford failure mode: a route that's
    # roughly DOUBLE the distance, with a real but modest enjoyment
    # improvement throughout (0.0 -> 0.4) — not a dramatic contrast,
    # just genuinely nicer. Confirmed analytically (see comments
    # below) that the slower route's own (distance x deficit) EXCEEDS
    # the direct route's, meaning NO penalty_scale value, however
    # large, can ever select it via the old additive-cost mechanism.
    deep_limit_graph = {
        1: [
            (2, 10000, 600, {"way_id": 100, "name": "Direct fast road"}),
            (3, 20000, 1500, {"way_id": 200, "name": "Long but genuinely nicer road"}),
        ],
        2: [(4, 100, 5, {"way_id": 101, "name": "end A"})],
        3: [(4, 100, 5, {"way_id": 201, "name": "end B"})],
        4: [],
    }
    deep_limit_enjoyment = {100: 0.0, 200: 0.4}

    # Confirm the impossibility analytically before testing the code,
    # so this test is provably checking what it claims to check:
    # direct total "badness" (distance_km * (1 - enjoyment), summed
    # over edges) = 10*(1-0) + 0.1*(1-0) = 10.1
    # slow total "badness" = 20*(1-0.4) + 0.1*(1-0) = 12.1
    # Since slow's badness (12.1) EXCEEDS direct's (10.1), the old
    # mechanism's inequality (extra_time < scale * (direct_badness -
    # slow_badness)) can never be satisfied for ANY scale >= 0, since
    # the bracket is negative — confirmed below, not just asserted.
    direct_badness = 10 * (1 - 0.0) + 0.1 * (1 - 0.0)
    slow_badness = 20 * (1 - 0.4) + 0.1 * (1 - 0.0)
    print(f"Direct badness: {direct_badness}, Slow badness: {slow_badness} "
          f"(slow > direct: {slow_badness > direct_badness} -- confirms impossibility)")
    assert slow_badness > direct_badness, (
        "Test setup bug: this test is meant to demonstrate the deep structural "
        "limitation, which requires the slower route's badness to EXCEED the "
        "faster route's -- adjust the numbers if this assertion fails"
    )

    # Verify the OLD method really does fail, exhaustively, across its
    # entire valid range (every candidate scale in its own default
    # sweep list) — not just at one spot-checked value.
    old_method_ever_succeeded = False
    for scale in DEFAULT_PENALTY_SCALE_CANDIDATES_S_PER_KM:
        p, _, _, _ = dijkstra_shortest_path(
            deep_limit_graph, 1, 4, enjoyment_scores=deep_limit_enjoyment,
            penalty_scale_s_per_km=scale,
        )
        if p == [1, 3, 4]:
            old_method_ever_succeeded = True
            break
    print(f"OLD method selected the nicer route at ANY tested scale (0 to 16000s/km)? "
          f"{old_method_ever_succeeded}")
    assert not old_method_ever_succeeded, (
        "BUG IN TEST SETUP: the old method was expected to fail at EVERY scale "
        "for this constructed case — if it succeeded anywhere, the numbers above "
        "don't actually demonstrate the deep structural limitation"
    )
    print("CONFIRMED — the old method cannot find this route at ANY scale in its entire range\n")

    print("--- Test 17 (NEW): the NEW Pareto + lambda search finds it directly ---")
    direct_path_dl, direct_time_dl, _, _ = dijkstra_shortest_path(deep_limit_graph, 1, 4)
    # Generous budget (3.0x) so budget itself isn't the constraint
    # being tested — purely testing whether the new search can find a
    # route the old one is mathematically incapable of, ever.
    result_new_search = find_best_average_enjoyment_route(
        deep_limit_graph, 1, 4, deep_limit_enjoyment,
        time_budget_s=direct_time_dl * 3.0,
    )
    print(f"Direct time: {direct_time_dl}s. New search result: path={result_new_search['path']}, "
          f"real_time_s={result_new_search['real_time_s']}, "
          f"avg_enjoyment={result_new_search['avg_enjoyment']}, "
          f"lambda_used={result_new_search['lambda_used']}")
    assert result_new_search["path"] == [1, 3, 4], (
        "The new Pareto+lambda search should find the nicer route that the old "
        "method is mathematically incapable of finding at any scale"
    )
    assert result_new_search["avg_enjoyment"] > 0.0, (
        "The selected route's average enjoyment should be meaningfully positive, "
        "reflecting the real 0.4 enjoyment value on its main edge"
    )
    print("PASSED — new search succeeds exactly where the old method is proven, "
          "mathematically, to always fail\n")

    print("--- Test 18 (NEW): the new search respects a TIGHT budget too (doesn't ignore time) ---")
    # Same graph, but a budget too tight to ever afford the nicer
    # route's real time (1505s) — must fall back to the direct path,
    # proving this isn't "always pick the nicer-looking road
    # regardless of cost," it's a genuine hard constraint.
    result_tight_new = find_best_average_enjoyment_route(
        deep_limit_graph, 1, 4, deep_limit_enjoyment,
        time_budget_s=direct_time_dl * 1.0,
    )
    print(f"Tight budget result: path={result_tight_new['path']}, "
          f"real_time_s={result_tight_new['real_time_s']}")
    assert result_tight_new["path"] == [1, 2, 4], (
        "With zero extra time tolerated, the search must fall back to the "
        "direct path -- the time budget is a HARD constraint, not a suggestion"
    )
    print("PASSED — tight budget correctly forces the direct path, confirming "
          "the time constraint is genuinely enforced, not bypassed\n")

    print("--- Test 19 (NEW): find_best_route_via_candidates finds the same proven-hard route ---")
    # Same deep_limit_graph as Tests 16/17 — confirms the new, cheaper
    # approach finds what the exact-but-impractical search also found.
    result_candidates = find_best_route_via_candidates(
        deep_limit_graph, 1, 4, deep_limit_enjoyment, time_budget_s=direct_time_dl * 3.0,
    )
    print(f"Candidates result: path={result_candidates['path']}, source={result_candidates['source']}, "
          f"avg_enjoyment={result_candidates['avg_enjoyment']}, "
          f"num_candidates_considered={result_candidates['num_candidates_considered']}")
    assert result_candidates["path"] == [1, 3, 4], (
        "The candidate-based search should also find the nicer route here"
    )
    print("PASSED — candidate-based search finds the same route as the exact search, much faster\n")

    print("--- Test 20 (NEW): diversity guarantee — near-duplicate candidates don't both survive ---")
    # A graph with two routes that are 100% identical in roads used
    # (literally the same path found twice via different generation
    # methods) should never appear twice in the candidate list.
    dup_graph = {
        1: [(2, 1000, 100, {"way_id": 900, "name": "Only road", "highway": "residential"})],
        2: [(3, 1000, 100, {"way_id": 901, "name": "Only road part 2", "highway": "residential"})],
        3: [],
    }
    dup_enjoyment = {900: 0.3, 901: 0.3}
    dup_candidates = generate_candidate_routes(
        dup_graph, 1, 3, dup_enjoyment, time_budget_s=1000,
        num_waypoint_candidates=5,
    )
    print(f"Candidates found for a graph with only ONE possible route: {len(dup_candidates)}")
    assert len(dup_candidates) == 1, (
        "A graph with exactly one possible route should produce exactly ONE "
        "candidate, however many generation methods are tried — duplicates "
        "must be collapsed, not listed separately"
    )
    print("PASSED — duplicate/identical routes correctly collapse to one candidate\n")

    print("--- Test 21 (NEW): the candidate search also respects a TIGHT budget ---")
    result_tight_candidates = find_best_route_via_candidates(
        deep_limit_graph, 1, 4, deep_limit_enjoyment, time_budget_s=direct_time_dl * 1.0,
    )
    print(f"Tight budget result: path={result_tight_candidates['path']}, source={result_tight_candidates['source']}")
    assert result_tight_candidates["path"] == [1, 2, 4], (
        "With zero extra time tolerated, must fall back to the direct path"
    )
    print("PASSED — tight budget correctly enforced\n")

    print("--- Test 22 (NEW): STRESS TEST at a scale comparable to this project's real graphs ---")
    print("(This is the test that matters most — the exact search looked correct on tiny")
    print("graphs too, right before failing catastrophically at real scale. Verifying THIS")
    print("approach at a comparable size before handing it over, not after.)")
    import random as _random
    import time as _time

    def _build_grid_graph(width, height, seed=42):
        _random.seed(seed)
        g = {}
        way_id_counter = 0
        for y in range(height):
            for x in range(width):
                node_id = y * width + x
                g[node_id] = []
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor_id = ny * width + nx
                        way_id_counter += 1
                        distance_m = _random.uniform(80, 400)
                        time_s = distance_m / _random.uniform(8, 20)
                        g[node_id].append((
                            neighbor_id, distance_m, time_s,
                            {"way_id": way_id_counter, "name": f"road_{way_id_counter}", "highway": "residential"}
                        ))
        return g

    def _build_enjoyment(g, seed=43):
        _random.seed(seed)
        e = {}
        for edges in g.values():
            for _, _, _, way_info in edges:
                e[way_info["way_id"]] = round(_random.uniform(0.0, 0.5), 3)
        return e

    # 165x165 ~= 27,225 nodes -- comparable to this project's real
    # Eynsham/Burford graph (26,773 nodes, 53,440 edges).
    stress_size = 165
    stress_graph = _build_grid_graph(stress_size, stress_size)
    stress_enjoyment = _build_enjoyment(stress_graph)
    stress_start, stress_end = 0, stress_size * stress_size - 1
    stress_direct_path, stress_direct_time, _, _ = dijkstra_shortest_path(stress_graph, stress_start, stress_end)
    stress_edges = sum(len(e) for e in stress_graph.values())

    t0 = _time.time()
    stress_result = find_best_route_via_candidates(
        stress_graph, stress_start, stress_end, stress_enjoyment,
        time_budget_s=stress_direct_time * 1.8,
    )
    elapsed = _time.time() - t0
    print(f"{stress_size}x{stress_size} grid ({stress_size*stress_size} nodes, {stress_edges} edges): "
          f"{round(elapsed, 2)}s, avg_enjoyment={stress_result['avg_enjoyment']}, "
          f"source={stress_result['source']}, "
          f"num_candidates_considered={stress_result['num_candidates_considered']}")
    assert elapsed < 60, (
        f"Stress test took {round(elapsed, 1)}s on a graph comparable to this "
        f"project's real graphs -- if this fails, this approach ALSO isn't fast "
        f"enough and needs further work before being handed over again"
    )
    print(f"PASSED — completed in {round(elapsed, 2)}s on a graph the same order of "
          f"magnitude as this project's real Eynsham/Burford graph\n")

    print("--- Test 23 (NEW): per-edge distance tracking is correct and sums to total distance ---")
    # Reuse the deep_limit_graph -- check that path_way_info entries
    # each carry their own correct "_edge_distance_m", and that
    # summing them matches the route's reported total distance.
    direct_path_23, direct_time_23, direct_dist_23, direct_wi_23 = dijkstra_shortest_path(
        deep_limit_graph, 1, 4
    )
    summed_distance = sum(w["_edge_distance_m"] for w in direct_wi_23)
    print(f"Reported total distance: {direct_dist_23}m, sum of per-edge distances: {summed_distance}m")
    assert summed_distance == direct_dist_23, (
        "Sum of each way_info entry's own _edge_distance_m must exactly match "
        "the route's total reported distance"
    )
    # Also confirm the ORIGINAL shared way_info dict (e.g. in the graph
    # itself) was never mutated -- each path entry must be its OWN copy.
    original_way_info_unmodified = "_edge_distance_m" not in deep_limit_graph[1][0][3]
    assert original_way_info_unmodified, (
        "BUG: the shared way_info dict in the graph itself must NOT be mutated -- "
        "each path entry should be an independent copy"
    )
    print("PASSED — per-edge distances are correct and the shared graph data is untouched\n")

    print("--- Test 24 (NEW): feature-forced candidates find a route via a specific point, "
          "even when that road wouldn't make the top-N ways cut ---")
    # A graph with TWO genuinely separate paths from 1 to 3: a fast
    # "direct" route via node 4, and a slower, mediocre-scoring route
    # via node 2 — where the target feature happens to sit. Confirms
    # generate_candidate_routes' source 4 finds the node-2 route
    # directly via coordinates, not via either road's own score (both
    # are deliberately low-scoring, so neither would make a top-N cut).
    feature_test_graph = {
        1: [
            (4, 5000, 300, {"way_id": 100, "name": "Direct dull road", "highway": "unclassified"}),
            (2, 5000, 500, {"way_id": 102, "name": "Mediocre slow approach", "highway": "unclassified"}),
        ],
        4: [(3, 100, 10, {"way_id": 103, "name": "direct exit", "highway": "unclassified"})],
        2: [(3, 200, 50, {"way_id": 101, "name": "mediocre slow exit", "highway": "unclassified"})],
        3: [],
    }
    feature_test_coords = {
        1: (51.000, -0.700), 4: (51.000, -0.650), 3: (51.0015, -0.649),
        2: (51.001, -0.650),
    }
    feature_test_enjoyment = {100: 0.05, 103: 0.0, 102: 0.05, 101: 0.0}  # all deliberately low/unremarkable

    near_node_2_feature = [{"name": "Hidden gem viewpoint", "lat": 51.0011, "lon": -0.6502}]

    candidates_without_feature = generate_candidate_routes(
        feature_test_graph, 1, 3, feature_test_enjoyment, time_budget_s=1000,
        num_waypoint_candidates=0,  # disable source 2 entirely for this test
        node_coords=feature_test_coords, target_features=None,
    )
    candidates_with_feature = generate_candidate_routes(
        feature_test_graph, 1, 3, feature_test_enjoyment, time_budget_s=1000,
        num_waypoint_candidates=0,
        node_coords=feature_test_coords, target_features=near_node_2_feature,
    )
    print(f"Candidates without feature targeting: {[c['source'] for c in candidates_without_feature]}")
    print(f"Candidates with feature targeting:    {[c['source'] for c in candidates_with_feature]}")
    assert not any(c["source"].startswith("via_feature_") for c in candidates_without_feature), (
        "Without target_features, no feature-forced candidate should appear at all"
    )
    feature_candidate = next((c for c in candidates_with_feature if c["source"].startswith("via_feature_")), None)
    assert feature_candidate is not None, (
        "With target_features provided, a feature-forced candidate should be generated"
    )
    assert feature_candidate["path"] == [1, 2, 3], (
        f"The feature-forced candidate should route via node 2 (where the feature sits), "
        f"got path {feature_candidate['path']}"
    )
    print("PASSED — feature-forced candidates correctly appear only when explicitly requested, "
          "targeting real coordinates directly rather than relying on either road's own score\n")

    print("--- Test 25 (NEW): min_time_s prevents a higher tier from collapsing onto a lower one ---")
    # Three candidates: a cheap mediocre one, and TWO that tie for the
    # best avg_enjoyment, one cheap and one expensive. Without
    # min_time_s, Compromise and Max Enjoyment would BOTH pick the
    # cheap best-scoring one (since it satisfies both budgets) --
    # exactly the real-world collapse found in testing. With
    # min_time_s=Compromise's own pick, Max Enjoyment must look beyond
    # it for something genuinely longer.
    tier_test_candidates = [
        {"source": "mediocre", "real_time_s": 100, "avg_enjoyment": 0.3},
        {"source": "best_cheap", "real_time_s": 200, "avg_enjoyment": 0.9},
        {"source": "best_expensive", "real_time_s": 500, "avg_enjoyment": 0.85},
    ]
    compromise_pick = select_best_candidate_within_budget(tier_test_candidates, time_budget_s=300)
    print(f"Compromise (budget=300s): picks '{compromise_pick['source']}' "
          f"(time={compromise_pick['real_time_s']}s, avg_enjoyment={compromise_pick['avg_enjoyment']})")
    assert compromise_pick["source"] == "best_cheap", "Compromise should pick the best-scoring option that fits its budget"

    max_enjoyment_naive = select_best_candidate_within_budget(tier_test_candidates, time_budget_s=600)
    print(f"Max Enjoyment WITHOUT min_time_s (budget=600s): picks '{max_enjoyment_naive['source']}' "
          f"-- SAME as Compromise, exactly the real-world collapse bug")
    assert max_enjoyment_naive["source"] == "best_cheap", (
        "Confirming the bug: without min_time_s, Max Enjoyment naively picks the same "
        "candidate as Compromise, since it's still the best-scoring option within the wider budget too"
    )

    max_enjoyment_fixed = select_best_candidate_within_budget(
        tier_test_candidates, time_budget_s=600, min_time_s=compromise_pick["real_time_s"]
    )
    print(f"Max Enjoyment WITH min_time_s={compromise_pick['real_time_s']}: picks "
          f"'{max_enjoyment_fixed['source']}' -- genuinely different from Compromise")
    assert max_enjoyment_fixed["source"] == "best_expensive", (
        "With min_time_s set, Max Enjoyment must select a genuinely longer candidate, "
        "even though it scores slightly lower than the one Compromise already offers"
    )
    print("PASSED — min_time_s correctly prevents the two tiers from silently collapsing\n")

    print("All pathfinding tests passed.")
