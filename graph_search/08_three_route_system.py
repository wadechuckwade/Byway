"""
Byway — Milestone 1.5, Step 5: The actual three-route system
==================================================================

What this does, in plain terms:
Builds the scored graph for a trip's bounding box, then finds THREE
routes from it:
  - Direct: pure fastest route (no enjoyment weighting, unchanged).
  - Compromise: the best candidate route found within a modest time
    budget (e.g. up to 20% longer than Direct).
  - Max Enjoyment: the same, within a more generous time budget.

NEW — LENGTH-WEIGHTED AVERAGING (bug fix): average_enjoyment_along_
path() used to average UNWEIGHTED across a route's unique way_ids — a
50m connector street counted exactly as much as a 5km scenic road.
Milestone 1 fixed this exact issue early on (its "Overall (length-
weighted average)" output) — the graph-search pipeline never
inherited that fix. Now uses each edge's real "_edge_distance_m" (see
04_pathfinding.py's dijkstra_shortest_path) to properly weight by
actual distance travelled on each way, not just how many distinct
ways happen to be in the list.

NEW — DELIBERATE FEATURE TARGETING: on top of the existing top-N-WAYS
candidate mechanism, find_three_routes() now builds a target_features
list — the highest-rated ScenicOrNot spots, every Conservation Area,
every named historic site found in the graph's bbox — and passes it to
generate_candidate_routes() (04_pathfinding.py, source 4) to force
routes directly through the nearest real road to EACH one. WHY: a road
can score only moderately on its own blended formula even when it's
the closest road to something genuinely excellent — going straight to
the best individual features targets them more reliably than hoping
they lift their road's overall score enough to make a top-N-by-score
cut.

TWO-PHASE SCORING (see 07_score_graph_enjoyment.py): candidates are
generated using PROVISIONAL (no-elevation) scores, then the small set
of ways those candidates actually use gets refined with real elevation
(now gradient-based, not a fixed absolute-climb threshold) before
final ranking.

DISPLAY: every printed score shows out of 10 to one decimal place via
score_module.to_ten(). Internal math is unchanged (still 0-1).

Network note: needs real internet access — run in Codespaces.
"""

import os
import math
import importlib.util


def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_this_dir = os.path.dirname(__file__)
fetch_module = _import_from_path("fetch_road_network", os.path.join(_this_dir, "01_fetch_road_network.py"))
build_module = _import_from_path("build_graph", os.path.join(_this_dir, "02_build_graph.py"))
path_module = _import_from_path("pathfinding", os.path.join(_this_dir, "04_pathfinding.py"))
score_module = _import_from_path("score_graph_enjoyment", os.path.join(_this_dir, "07_score_graph_enjoyment.py"))
map_export_module = _import_from_path("map_export", os.path.join(_this_dir, "map_export.py"))


# --- Journey preference defaults -------------------------------------
JOURNEY_PREFERENCE_DEFAULTS = {
    "compromise_extra_time_minutes": None,
    "compromise_max_time_multiplier": 1.2,      # up to 20% longer than Direct
    "max_enjoyment_extra_time_minutes": None,
    "max_enjoyment_max_time_multiplier": 1.8,   # used only if escalation is explicitly disabled (set to None)
    "max_enjoyment_multiplier_escalation": [1.8, 2.5, 3.5, 5.0],
    # NEW — FLEXIBLE ceiling, tried in order: real-route testing found
    # Max Enjoyment reverting to repeat Compromise even when a
    # genuinely better detour existed, just beyond the single fixed
    # 1.8x cutoff. Rather than pick one ceiling and give up if nothing
    # beats Compromise within it, try progressively wider budgets,
    # STOPPING at the first one that finds something genuinely better
    # — so a real improvement just beyond 1.8x still gets found,
    # without defaulting to the most generous (5x) ceiling when a
    # smaller widening would already do. Candidates are generated ONCE
    # at the widest value in this list (5.0x here), then checked
    # smallest-first for selection — no extra Dijkstra cost for trying
    # multiple ceilings, since the candidate pool itself doesn't change.
    "num_waypoint_candidates": None,
    "similarity_threshold": None,
    "elevation_target_spacing_m": None,
    # NEW — CAPPED, REALLOCATED feature-target budget (direct
    # feedback: 299 feature-targets, or whatever an uncapped category
    # union produces, is more search breadth than needed, costing real
    # Dijkstra time for diminishing returns). ~50 total, split across
    # categories.
    #
    # CHANGED — land-cover grid targeting now defaults to 0 (off),
    # not 20. This was a real, evidence-backed hypothesis a few turns
    # back ("scatters cells across the WHOLE bbox, no observed route
    # has ever won via a landcover_grid source") — now PROVEN, not
    # just suspected: a real run showed grid targeting taking 341.5s
    # out of a 425.2s total (80% of runtime), because
    # generate_landcover_grid_targets classifies EVERY cell in the
    # bbox (728 of them, for this trip's wider bbox) just to rank and
    # keep the best 20 — throwing away 97% of what it computed. Worse,
    # this scales UP as trips get longer (bigger bbox, more cells),
    # right when Milestone 2 is about to add real cost on top. Kept
    # available (set this above 0 to re-enable) in case a smarter
    # sampling strategy is worth revisiting later — e.g. classifying a
    # random subset of cells rather than the whole grid — but off by
    # default given the proven cost and zero observed benefit so far.
    "num_landcover_grid_targets": 0,
    "num_scenicornot_targets": 10,
    "num_conservation_area_targets": 10,
    "num_historic_site_targets": 10,
    "landcover_grid_cell_size_km": 1.0,
    # NEW — corridor search (see search_within_corridor): runs AFTER
    # Compromise/Max Enjoyment are picked, searching for genuine LOCAL
    # alternatives within a geographic buffer around each, rather than
    # only ever generating radically-different far-flung detours.
    # Deliberately lean by default — 2 penalty scales, not the full
    # sweep — kept cheap on purpose given real concern about total
    # runtime, especially with Milestone 2 about to add cost on top.
    "corridor_search_enabled": True,
    "corridor_buffer_km": 1.2,
    "corridor_penalty_scales": [1000, 4000],
    # NEW (Milestone 2) — food/drink AI recommendations. Deliberately
    # OFF by default — this is the one part of the pipeline that costs
    # real money per call (everything else is free data sources), so
    # it should never run unless explicitly asked for, not silently
    # fire on every route search. See scoring/ai_circuit_breaker.py
    # for the hard daily spend cap that applies regardless.
    "enable_food_drink_ai": False,
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
}


def summarize_road_sequence(way_info_list):
    """Same ref-grouping display logic proven in 05_full_pathfinding_test.py."""
    road_sequence = []
    for w in way_info_list:
        ref = w.get("ref", "")
        name = w["name"]
        if ref:
            group_key = ref
            display = ref
        else:
            group_key = name or f"(unnamed {w['highway']})"
            display = group_key

        if not road_sequence or road_sequence[-1]["key"] != group_key:
            road_sequence.append({"key": group_key, "display": display, "local_names": set()})
        if name:
            road_sequence[-1]["local_names"].add(name)

    lines = []
    for road in road_sequence:
        local_names = sorted(road["local_names"] - {road["display"]})
        if local_names:
            lines.append(f"{road['display']} (locally: {', '.join(local_names)})")
        else:
            lines.append(road["display"])
    return lines


def average_enjoyment_along_path(way_info_list, enjoyment_scores):
    """
    LENGTH-WEIGHTED average enjoyment score across a path — weights
    each edge's contribution by its own real distance
    ("_edge_distance_m", set by dijkstra_shortest_path's path
    reconstruction), not just "how many distinct ways did this route
    use," which previously let a route's score get diluted by however
    many short connector ways it happened to need, regardless of how
    little distance they actually covered.

    Falls back to the OLD unweighted-unique-ways behaviour if
    "_edge_distance_m" isn't present on any entry (e.g. way_info lists
    built by something other than dijkstra_shortest_path's
    reconstruction) — kept for backward compatibility, but every
    candidate this project's own search produces DOES carry this field
    now, so the weighted path is what actually runs in practice.

    Returns a 0-1 value (internal scale) — convert with
    score_module.to_ten() at display time.
    """
    total_distance = sum(w.get("_edge_distance_m", 0) for w in way_info_list)
    if total_distance <= 0:
        way_ids = set(w["way_id"] for w in way_info_list)
        scores = [enjoyment_scores.get(wid, {}).get("enjoyment_score", 0.0) for wid in way_ids]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    weighted_sum = sum(
        enjoyment_scores.get(w["way_id"], {}).get("enjoyment_score", 0.0) * w.get("_edge_distance_m", 0)
        for w in way_info_list
    )
    return round(weighted_sum / total_distance, 3)


def best_stretch_enjoyment_along_path(way_info_list, enjoyment_scores, best_stretch_fraction=0.3):
    """
    NEW — length-weighted average enjoyment over just the BEST
    best_stretch_fraction of the route's total distance (by each
    edge's own way's enjoyment score, highest first). Answers "how
    good does this route get at its best," distinct from
    average_enjoyment_along_path's "what's this route like typically."

    WHY THIS EXISTS: Milestone 1 had exactly this metric (its "Best
    30% of route, peak-weighted" output) specifically because a length-
    weighted AVERAGE structurally punishes a route for needing SOME
    unremarkable connecting distance to reach something genuinely
    great — which is most real "longer but better" detours, almost by
    definition. The graph-search pipeline never inherited this metric,
    so there was no way to see whether a route's peak moments were
    excellent even when its average looked unremarkable (e.g. the
    Eynsham/Burford Leafield detour, which spends real distance on
    Leafield's own residential streets to reach the rated viewpoint).

    Returns a 0-1 value (internal scale) — convert with
    score_module.to_ten() at display time.
    """
    total_distance = sum(w.get("_edge_distance_m", 0) for w in way_info_list)
    if total_distance <= 0:
        return 0.0

    scored_edges = [
        (enjoyment_scores.get(w["way_id"], {}).get("enjoyment_score", 0.0), w.get("_edge_distance_m", 0))
        for w in way_info_list
    ]
    scored_edges.sort(key=lambda x: -x[0])

    target_distance = total_distance * best_stretch_fraction
    running_distance = 0.0
    weighted_sum = 0.0
    for score, distance in scored_edges:
        if running_distance >= target_distance:
            break
        # PRORATE the boundary-crossing edge: only take however much of
        # it is needed to exactly fill the remaining target window,
        # rather than including it in full. Without this, a single
        # large, lower-scoring edge that happens to cross the boundary
        # could dominate the whole "best stretch" sample (e.g. one 8km
        # mediocre way added in full just to reach a 30% target,
        # swamping a genuinely excellent but much shorter segment) —
        # found via testing, not assumed.
        remaining_needed = target_distance - running_distance
        distance_to_use = min(distance, remaining_needed)
        weighted_sum += score * distance_to_use
        running_distance += distance_to_use

    return round(weighted_sum / running_distance, 3) if running_distance > 0 else 0.0


ROUTE_ELEVATION_WEIGHT = 0.175
# NEW — elevation is no longer blended into any individual way's own
# driving_enjoyment/scenery (see 07_score_graph_enjoyment.py's
# refine_scores_with_elevation docstring for why: SRTM's ~10m vertical
# noise becomes a huge relative error over a short 100-300m way, and
# "this road climbs dramatically" is a property of a sustained
# stretch, not any single short way). Instead, ONE elevation score is
# computed per CANDIDATE ROUTE (route_level_elevation_score, below)
# and blended in at the route level. 0.175 preserves elevation's PRIOR
# overall contribution to the final enjoyment_score under the old
# formula (0.4 weight within driving x 0.25 driving-weight, plus 0.15
# weight within scenery x 0.5 scenery-weight = 0.10 + 0.075 = 0.175) --
# a deliberate choice to keep elevation's overall IMPORTANCE unchanged
# while fixing WHERE/HOW it's measured, not a fresh guess.


def route_level_elevation_score(way_info_list, enjoyment_scores):
    """
    NEW — computes ONE gradient-based elevation score for a WHOLE
    candidate route, rather than averaging together many individual
    ways' own (noise-prone, short-distance) elevation scores.

    Aggregates each way's own total_ascent_m (already computed per-way
    in refine_scores_with_elevation, from its real interpolated
    elevation profile) PROPORTIONALLY by how much of that way's total
    length this specific edge represents -- e.g. if an edge covers
    half of its way's total length, it contributes half that way's
    total ascent. This avoids needing to re-derive which exact points
    of a way's elevation profile correspond to one specific edge
    (most graph-search ways are short enough that this proportional
    approximation is reasonable, not a precision loss that matters in
    practice).

    Returns a 0-1 value (internal scale) — blend into avg_enjoyment/
    best_stretch_enjoyment via ROUTE_ELEVATION_WEIGHT, not used alone.
    """
    total_distance = 0.0
    total_ascent = 0.0
    for w in way_info_list:
        edge_distance = w.get("_edge_distance_m", 0)
        total_distance += edge_distance
        way_data = enjoyment_scores.get(w["way_id"], {})
        way_total_ascent = way_data.get("total_ascent_m", 0.0) or 0.0
        way_total_length = way_data.get("_total_length_m", 0.0) or 0.0
        if way_total_length > 0:
            total_ascent += way_total_ascent * (edge_distance / way_total_length)

    if total_distance <= 0:
        return 0.0

    gradient = total_ascent / total_distance
    reasonable_max_gradient = 0.08  # matches scoring/elevation.py's own threshold
    return min(gradient / reasonable_max_gradient, 1.0)


LANDCOVER_CATEGORIES = ("urban", "agriculture", "forest_natural", "wetland", "water")


def landcover_composition_along_path(way_info_list, enjoyment_scores):
    """
    NEW — length-weighted average land-cover composition across a
    whole route, mirroring average_enjoyment_along_path's weighting
    exactly. Each way's own "landcover_composition" (from
    07_score_graph_enjoyment.py's refine_scores_with_elevation, where
    it's computed per-way from sampled CORINE points) gets weighted by
    that edge's real distance, so a long stretch of forest contributes
    more to the ROUTE's overall composition than a short stretch of
    farmland, not equally regardless of length.

    WHY THIS EXISTS: direct product requirement — users should be able
    to see WHAT a route's land actually consists of (e.g. "62% Forest
    & natural, 28% Agriculture, 10% Urban"), not just an opaque single
    landcover_score number, if they choose to view a score breakdown.

    Returns {"urban": 0-1, "agriculture": 0-1, "forest_natural": 0-1,
    "wetland": 0-1, "water": 0-1} (fractions sum to ~1.0), or all-zero
    if no way in the route has composition data (e.g. ways under 3
    points, which never get refined).
    """
    total_distance = sum(w.get("_edge_distance_m", 0) for w in way_info_list)
    if total_distance <= 0:
        return {k: 0.0 for k in LANDCOVER_CATEGORIES}

    weighted_totals = {k: 0.0 for k in LANDCOVER_CATEGORIES}
    for w in way_info_list:
        distance = w.get("_edge_distance_m", 0)
        composition = enjoyment_scores.get(w["way_id"], {}).get("landcover_composition")
        if not composition:
            continue
        for category in LANDCOVER_CATEGORIES:
            weighted_totals[category] += composition.get(category, 0.0) * distance

    return {k: round(v / total_distance, 3) for k, v in weighted_totals.items()}


def _shannon_entropy_normalized(composition):
    """
    Shannon entropy of a composition dict's values, normalised to 0-1
    by the MAXIMUM possible entropy for however many distinct
    categories the dict's own keys define (an even split across ALL
    of them) — not however many happen to be non-zero for one
    particular route, which would wrongly let a route using only 2 of
    4 possible categories, evenly split, score as "maximally diverse."
    Used by compute_experimental_scores' Uniqueness formula.
    """
    values = [v for v in composition.values() if v > 0]
    if not values:
        return 0.0
    entropy = -sum(v * math.log(v) for v in values)
    max_entropy = math.log(len(composition)) if len(composition) > 1 else 1.0
    return round(min(1.0, entropy / max_entropy), 3) if max_entropy > 0 else 0.0


def compute_experimental_scores(way_info_list, final_scores, landcover_composition, route_elevation_score,
                                 villages_count=0, historic_count=0):
    """
    NEW (Milestone 2) — composite EXPERIMENTAL scores: Drama, Charm,
    Uniqueness. Built entirely from data this project already
    computes — no new fetching needed for any of these three.

    GROUNDING, HONESTLY STATED: the development plan names Quercia,
    Schifanella & Aiello's "beautiful/quiet/happy" routing research as
    a reference point. Real, validated findings from that actual work
    support these as genuine categories, not arbitrary inventions —
    "places offering distinctive experiences are considered happy"
    (-> Uniqueness) and "the presence of charming...historical
    elements" was found to measurably balance out otherwise negative
    impressions of busy roads (-> Charm). What does NOT transfer:
    their model was trained on crowdsourced PAIRWISE HUMAN COMPARISONS
    of street photos, at real scale (thousands of volunteers). We have
    no access to anything like that — these are principled PROXIES
    built from existing signals, not independently validated
    predictors of real human perception. Hence EXPERIMENTAL, labelled
    as such everywhere this surfaces (terminal output, map popups, the
    app's route cards) — see the plan's own suggested validation step
    (informal pairwise comparison testing) before trusting these
    further.

    DRAMA — dramatic terrain + technically intense driving + a water
    bonus (gorges/coastal drama). Uses RAW curvature (not curvature
    already discounted by road class) deliberately — a dramatic bend
    on a trunk road is still dramatic to experience, even though its
    overall "enjoyment" is correctly penalised elsewhere for being a
    major road.

    CHARM — DENSITY of villages + historic sites per km, capped at a
    reasonable ceiling (3+ qualifying features/km treated as maximal).
    Deliberately different from "interest" (which uses noisy-OR,
    rewarding ANY one strong signal without requiring more) — Charm
    specifically rewards having MANY charming things along the way,
    not just one.

    UNIQUENESS — Shannon entropy of the route's own land-cover mix
    (see _shannon_entropy_normalized) — a route through a genuinely
    varied blend of landscapes scores higher than one that's
    monotonously a single type, regardless of whether that one type
    is "good." A deliberate, defensible INTERPRETATION of
    "distinctiveness," not the only possible one.

    villages_count/historic_count: total qualifying features within
    each category's own HIGHLIGHT_DISTANCE_THRESHOLDS_M radius — NOT
    capped at 5 like find_route_highlights returns, since Charm needs
    the real count, not just the best 5 overall.

    Returns {"drama": 0-1, "charm": 0-1, "uniqueness": 0-1} — convert
    with score_module.to_ten() at display time, same as every other
    score in this project.
    """
    total_distance_m = sum(w.get("_edge_distance_m", 0) for w in way_info_list)
    if total_distance_m <= 0:
        return {"drama": 0.0, "charm": 0.0, "uniqueness": 0.0}

    raw_curvature_weighted = sum(
        final_scores.get(w["way_id"], {}).get("_curvature_score", 0.0) * w.get("_edge_distance_m", 0)
        for w in way_info_list
    ) / total_distance_m
    water_pct = landcover_composition.get("water", 0.0)
    drama = round(min(1.0, 0.5 * route_elevation_score + 0.35 * raw_curvature_weighted + 0.15 * water_pct), 3)

    distance_km = total_distance_m / 1000
    feature_density_per_km = (villages_count + historic_count) / max(distance_km, 1.0)
    charm = round(min(1.0, feature_density_per_km / 3.0), 3)

    uniqueness = _shannon_entropy_normalized(landcover_composition)

    return {"drama": drama, "charm": charm, "uniqueness": uniqueness}


# Distance thresholds for "does this route pass near this named
# feature" — deliberately different per category, matching how each
# one is actually experienced. Villages are AREAS (worth a wider catch
# radius — you can tell you're near a village before you're right on
# top of its centroid). ScenicOrNot ratings represent a ~1km grid
# square's character, matching the wider threshold already used for
# scoring it elsewhere (scoring/proximity.py). Historic sites/listed
# buildings are specific points — a tighter radius means "actually
# passed this," not "was vaguely in the same area."
HIGHLIGHT_DISTANCE_THRESHOLDS_M = {
    "villages": 600,
    "scenicornot": 800,
    "historic": 300,
}


def find_route_highlights(way_info_list, ways, region_features, max_highlights=5):
    """
    NEW — finds the SPECIFIC NAMED features (villages, ScenicOrNot-
    rated viewpoints, historic sites/listed buildings) a route
    actually passes near, not just an aggregate score.

    WHY THIS EXISTS: direct feedback — score alone doesn't say WHAT
    makes a route good. This surfaces the concrete, named reasons
    (e.g. "passes within 180m of Lurgashall, a Conservation Area in
    the South Downs National Park," or "Grade I listed church within
    220m") so a route's enjoyment score has real, checkable substance
    behind it, not just a number.

    CHANGED — returns the best max_highlights OVERALL (ranked by
    significance — each category's own "weight" field, which is
    already a comparable 0-1 scale across all three sources — then by
    proximity as a tiebreaker), not the best N per category. Real
    feedback found showing up to 5 per category (15 total) genuinely
    overwhelming; what's actually wanted is "the highlights of the
    trip," not an exhaustive list.

    Reuses scoring.proximity._min_distance_to_route_m directly — the
    SAME distance calculation already trusted throughout this
    codebase for scoring, not a second, independent implementation
    that could quietly drift from it.

    way_info_list: a route's "way_info" (as returned by find_three_
    routes) — used to reconstruct the route's actual point sequence.
    ways: {way_id: {"points": [...], ...}} from score_all_ways_
    provisional — needed to look up each way's real geometry.
    region_features: the same dict score_all_ways_provisional returns
    (conservation_areas, scenicornot, historic) — the raw, named
    feature lists to check the route against.

    Returns a single list of up to max_highlights dicts (each tagged
    with "category": "village"/"scenicornot"/"historic", plus
    "distance_m" and whatever fields the source list already carried
    — name/weight/grade/lpa/etc.), sorted best-first. An empty list
    just means nothing nearby qualified — not an error.
    """
    from scoring.proximity import _min_distance_to_route_m

    route_points = []
    seen_way_ids = set()
    for w in way_info_list:
        way_id = w["way_id"]
        if way_id in seen_way_ids:
            continue
        seen_way_ids.add(way_id)
        if way_id in ways:
            route_points.extend(ways[way_id]["points"])

    if not route_points:
        return []

    def _check_category(features, threshold_m, category_label):
        hits = []
        for f in features:
            if "lat" not in f or "lon" not in f:
                continue
            dist_m = _min_distance_to_route_m(f["lat"], f["lon"], route_points)
            if dist_m <= threshold_m:
                hits.append({**f, "distance_m": round(dist_m, 1), "category": category_label})
        return hits

    all_hits = (
        _check_category(region_features.get("conservation_areas", []),
                         HIGHLIGHT_DISTANCE_THRESHOLDS_M["villages"], "village")
        + _check_category(region_features.get("scenicornot", []),
                           HIGHLIGHT_DISTANCE_THRESHOLDS_M["scenicornot"], "scenicornot")
        + _check_category(region_features.get("historic", []),
                           HIGHLIGHT_DISTANCE_THRESHOLDS_M["historic"], "historic")
    )
    # Best-first: highest weight (significance) wins; closer breaks ties.
    all_hits.sort(key=lambda h: (-h.get("weight", 0), h["distance_m"]))
    return all_hits[:max_highlights]


def count_qualifying_features(way_info_list, ways, features, threshold_m):
    """
    NEW (Milestone 2) — counts how many features in a list qualify as
    "near this route" (same distance-check logic as find_route_
    highlights' _check_category), WITHOUT capping at any maximum —
    used by compute_experimental_scores' Charm formula, which needs
    the real total count (how MANY charming things), not just the
    best 5 overall the way highlights display does.
    """
    from scoring.proximity import _min_distance_to_route_m

    route_points = []
    seen_way_ids = set()
    for w in way_info_list:
        way_id = w["way_id"]
        if way_id in seen_way_ids:
            continue
        seen_way_ids.add(way_id)
        if way_id in ways:
            route_points.extend(ways[way_id]["points"])

    if not route_points:
        return 0

    count = 0
    for f in features:
        if "lat" not in f or "lon" not in f:
            continue
        if _min_distance_to_route_m(f["lat"], f["lon"], route_points) <= threshold_m:
            count += 1
    return count


def format_highlights_text(highlights):
    """
    NEW — turns find_route_highlights' flat, ranked list into short,
    readable lines for printing/display — e.g.:
      "Lurgashall (village, South Downs National Park Authority) — 180m"
      "St Peter's Church (historic site, Grade I listed) — 220m"
    Returns a list of strings, in the SAME best-first order
    find_route_highlights already established — empty if nothing
    qualified at all.
    """
    lines = []
    for h in highlights:
        category = h.get("category")
        if category == "village":
            lpa_note = f", {h['lpa']}" if h.get("lpa") else ""
            lines.append(f"{h.get('name', 'Unnamed village')} (village{lpa_note}) — {round(h['distance_m'])}m")
        elif category == "scenicornot":
            rating_10 = round(h.get("weight", 0) * 10, 1)
            lines.append(f"{h.get('name', 'Scenic viewpoint')} (ScenicOrNot rated {rating_10}/10) — {round(h['distance_m'])}m")
        elif category == "historic":
            grade_note = f", Grade {h['grade']} listed" if h.get("grade") else ""
            lines.append(f"{h.get('name', 'Historic site')} (historic site{grade_note}) — {round(h['distance_m'])}m")
    return lines


def build_target_features(region_features, prefs, bbox=None):
    """
    NEW — builds the list of point-features to deliberately force
    candidate routes through (04_pathfinding.py's generate_candidate_
    routes, source 4), from the raw feature lists score_all_ways_
    provisional() fetched, PLUS a NEW land-cover grid category.

    - Top N ScenicOrNot spots, sorted by rating (their real "weight").
    - Top N Conservation Areas, NOW sorted by weight (villages.py's
      LPA-based weighting — National Park/AONB-administered areas
      rank first), not first-found order.
    - Top N historic sites, NOW sorted by weight (historic_england.py's
      official Grade — Grade I ranks above Grade II* above Grade II;
      plain OSM-tagged sites with no Historic England grade default to
      weight 0 for THIS ranking/cap purpose specifically — they still
      contribute normally to scoring via score_proximity's own
      default, just aren't prioritised for the limited target slots
      when genuinely graded sites are available).
    - NEW: top N land-cover grid cells (scoring.landcover.
      generate_landcover_grid_targets) — land cover has no pre-
      existing list of "best spots" the way the other three
      categories do (it's a continuous surface, not a finite list of
      places), so this GRIDS the bbox and ranks cells directly. Only
      built if bbox is provided — without it, falls back to the
      other three categories alone (e.g. if a caller doesn't have a
      bbox handy).

    Each entry gets a "source_label" for clearer diagnostics in
    generate_candidate_routes' "via_feature_..." source tags.

    Total target count is now CAPPED and DELIBERATELY allocated across
    categories (default ~50 total, see JOURNEY_PREFERENCE_DEFAULTS) —
    direct feedback that an uncapped union of every category was more
    search breadth than needed, costing real Dijkstra time for
    diminishing returns.
    """
    targets = []

    scenicornot = sorted(
        region_features.get("scenicornot", []), key=lambda f: -f.get("weight", 0)
    )[:prefs["num_scenicornot_targets"]]
    for f in scenicornot:
        targets.append({**f, "source_label": f"scenicornot_{f.get('weight', 0)}"})

    conservation_areas = sorted(
        region_features.get("conservation_areas", []), key=lambda f: -f.get("weight", 0)
    )[:prefs["num_conservation_area_targets"]]
    for f in conservation_areas:
        targets.append({**f, "source_label": f"village_{f.get('name', '')}_w{f.get('weight', 0)}"})

    historic = sorted(
        region_features.get("historic", []), key=lambda f: -f.get("weight", 0)
    )[:prefs["num_historic_site_targets"]]
    for f in historic:
        grade = f.get("grade", "")
        targets.append({**f, "source_label": f"historic_{f.get('name', '')}{f'_Grade{grade}' if grade else ''}"})

    if bbox is not None and prefs.get("num_landcover_grid_targets", 0) > 0:
        from scoring.landcover import generate_landcover_grid_targets, fetch_landcover_classes
        import time as _time
        # NEW — explicit timing around grid-targeting's land-cover
        # cost specifically, separate from corridor-refinement's own
        # cost (printed later, in refine_scores_with_elevation). Why:
        # real performance review raised a concrete, testable
        # hypothesis — this mechanism scatters cells across the WHOLE
        # bbox (including any town centres in it), deliberately
        # touching tiles candidate-road-following corridor refinement
        # would otherwise never need to load, and no real run so far
        # has shown a winning route actually sourced from a land-cover
        # grid target. This print turns that hypothesis into a
        # measurable answer on the next run, rather than continued
        # reasoning about it in the abstract.
        t_grid_start = _time.time()
        landcover_targets = generate_landcover_grid_targets(
            bbox, fetch_landcover_classes,
            cell_size_km=prefs.get("landcover_grid_cell_size_km", 1.0),
            top_n=prefs["num_landcover_grid_targets"],
        )
        print(f"  [timing] land-cover grid targeting ({prefs['num_landcover_grid_targets']} cells): "
              f"{round(_time.time() - t_grid_start, 1)}s")
        for f in landcover_targets:
            targets.append({**f, "source_label": f"landcover_grid_w{f.get('weight', 0)}"})

    return targets


def fully_scored_route(way_info_list, enjoyment_scores):
    """
    Computes a route's avg_enjoyment and best_stretch_enjoyment with
    the route-level elevation blend correctly applied (see
    route_level_elevation_score's docstring for why elevation is
    blended at the route level rather than per-way).

    IMPORTANT: this is the CANONICAL way to score a route's
    avg_enjoyment/best_stretch_enjoyment outside find_three_routes —
    e.g. in diagnostic scripts. Calling average_enjoyment_along_path/
    best_stretch_enjoyment_along_path directly gives the WAY-LEVEL-
    ONLY score, missing the route-level elevation blend entirely —
    this used to be a closure local to find_three_routes for exactly
    this calculation, and a diagnostic script written before that
    blend existed would have silently produced numbers inconsistent
    with what find_three_routes actually reports for the same route.
    Promoted to a top-level function specifically to prevent that
    class of inconsistency recurring in future diagnostic scripts.

    Returns (avg_enjoyment, best_stretch_enjoyment, route_elevation_score)
    — all 0-1 internal scale; convert with score_module.to_ten() for
    display.
    """
    way_avg = average_enjoyment_along_path(way_info_list, enjoyment_scores)
    way_best = best_stretch_enjoyment_along_path(way_info_list, enjoyment_scores)
    route_elev = route_level_elevation_score(way_info_list, enjoyment_scores)
    blended_avg = (1 - ROUTE_ELEVATION_WEIGHT) * way_avg + ROUTE_ELEVATION_WEIGHT * route_elev
    blended_best = (1 - ROUTE_ELEVATION_WEIGHT) * way_best + ROUTE_ELEVATION_WEIGHT * route_elev
    return round(blended_avg, 3), round(blended_best, 3), route_elev


def _subsample_path_points(path_points, target_spacing_m=300):
    """
    Thins a path's points down to roughly target_spacing_m apart —
    used to keep corridor-distance checks fast (comparing every graph
    node against a SPARSE set of ~20-50 path points, not every single
    one of potentially hundreds), without needing real precision: the
    corridor buffer itself (1km+) is far larger than the gap this
    introduces.
    """
    if len(path_points) <= 2:
        return path_points
    kept = [path_points[0]]
    accumulated_m = 0.0
    for i in range(1, len(path_points)):
        lat1, lon1 = path_points[i - 1]
        lat2, lon2 = path_points[i]
        accumulated_m += path_module._haversine_distance_m(lat1, lon1, lat2, lon2)
        if accumulated_m >= target_spacing_m:
            kept.append(path_points[i])
            accumulated_m = 0.0
    if kept[-1] != path_points[-1]:
        kept.append(path_points[-1])
    return kept


def build_corridor_subgraph(graph, node_coords, path_node_ids, buffer_km=1.2):
    """
    NEW — builds a SUBGRAPH containing only nodes within buffer_km of
    ANY point on the given path, plus only the edges connecting two
    such nodes. Searching within this restricted subgraph (see
    search_within_corridor) finds genuine LOCAL alternatives to an
    already-good route — directly addressing a real, named gap:
    nothing previously generated small variations of a good candidate
    (e.g. "same general route, but swap this one segment for the
    parallel backroad one field over") — only radically different,
    far-flung detours (waypoint/feature-forced) or a single global
    blend-parameter sweep.

    Path points are subsampled first (see _subsample_path_points) to
    keep the distance check fast — comparing every graph node against
    a sparse ~20-50 point set, not every point on a long path.

    Returns (corridor_graph, corridor_node_count) — corridor_graph has
    the SAME shape as the main graph ({node_id: [(neighbor, dist,
    time, way_info), ...]}), just restricted to corridor nodes/edges.
    """
    path_points = [node_coords[nid] for nid in path_node_ids if nid in node_coords]
    sparse_points = _subsample_path_points(path_points)

    buffer_m = buffer_km * 1000
    corridor_nodes = set()
    for node_id, (lat, lon) in node_coords.items():
        for plat, plon in sparse_points:
            if path_module._haversine_distance_m(lat, lon, plat, plon) <= buffer_m:
                corridor_nodes.add(node_id)
                break

    corridor_graph = {}
    for node_id in corridor_nodes:
        edges = graph.get(node_id, [])
        corridor_graph[node_id] = [e for e in edges if e[0] in corridor_nodes]

    return corridor_graph, len(corridor_nodes)


def search_within_corridor(graph, node_coords, start_node, end_node, path_to_follow,
                            simple_enjoyment_scores, buffer_km=1.2, penalty_scales=None, verbose=False):
    """
    NEW — runs the SAME enjoyment-blended Dijkstra mechanism already
    used for the blend-scale sweep (04_pathfinding.py's
    dijkstra_shortest_path with penalty_scale_s_per_km), but restricted
    to a corridor around an existing good route, rather than the whole
    graph. Deliberately lean — just 2 penalty scales by default, not
    the full 5-scale sweep — direct feedback: this needs to stay cheap,
    not add real time on top of an already-long run, especially with
    Milestone 2 about to add more cost regardless.

    simple_enjoyment_scores: {way_id: plain_score_number} — the SAME
    simplified shape generate_candidate_routes uses internally (see
    find_three_routes' own "simple_provisional" variable), NOT the
    raw provisional_scores dict-of-dicts that score_all_ways_
    provisional returns. dijkstra_shortest_path does `1 - enjoyment`
    directly on whatever's passed — a dict there is a real, silent-
    until-runtime TypeError, not a dict-vs-number issue worth
    rediscovering twice.

    Returns a list of candidate dicts (same shape as
    generate_candidate_routes' output, source labelled "corridor_*")
    — empty if no path exists within the corridor (a narrow buffer can
    genuinely cut off a needed connecting road; this is a real,
    expected outcome sometimes, not a bug) or nothing was found at all.
    """
    corridor_graph, node_count = build_corridor_subgraph(graph, node_coords, path_to_follow, buffer_km)
    if verbose:
        print(f"    Corridor ({buffer_km}km buffer): {node_count} nodes in subgraph "
              f"(vs {len(node_coords)} in the whole graph).")

    scales = penalty_scales or [1000, 4000]
    candidates = []
    for scale in scales:
        path, t, d, wi = path_module.dijkstra_shortest_path(
            corridor_graph, start_node, end_node,
            enjoyment_scores=simple_enjoyment_scores, penalty_scale_s_per_km=scale,
        )
        if path is None:
            continue
        # Rough PROVISIONAL estimate only (unweighted, using the
        # simple {way_id: score} shape directly) — purely for this
        # function's own verbose logging. The REAL score used for any
        # actual decision is computed fresh by find_three_routes via
        # fully_scored_route() once this candidate's ways are refined
        # with real elevation/land cover — this value is never used
        # for that, deliberately, so it doesn't need the full
        # dict-of-dicts shape average_enjoyment_along_path expects.
        way_ids = set(w["way_id"] for w in wi)
        rough_scores = [simple_enjoyment_scores.get(wid, 0.0) for wid in way_ids]
        rough_avg_enjoyment = round(sum(rough_scores) / len(rough_scores), 3) if rough_scores else 0.0
        candidates.append({
            "path": path, "way_info": wi, "real_time_s": t, "real_distance_m": d,
            "avg_enjoyment": rough_avg_enjoyment, "source": f"corridor_{buffer_km}km_scale_{scale}",
        })

    return candidates


def find_three_routes(graph, start_node, end_node, provisional_scores, ways,
                       region_features=None, node_coords=None,
                       preferences=None, verbose=True, progress_callback=None):
    """
    Top-level function: finds Direct (pure fastest), Compromise, and
    Max Enjoyment routes from the SAME graph, using TWO-PHASE scoring
    and (if region_features/node_coords are provided) deliberate
    feature-targeted candidate generation.

    provisional_scores, ways, region_features: from score_module.
    score_all_ways_provisional(graph, node_coords) — call that FIRST
    (it now returns 3 values), then pass all three here, plus
    node_coords itself, to enable feature targeting. Without
    region_features/node_coords, falls back to the top-N-ways-only
    candidate mechanism (still fully functional, just without source 4).

    progress_callback: optional — called with a short stage
    description string (e.g. "Generating candidates...") at each major
    phase transition. NEW — direct feedback: a long-running web UI
    request had no visibility into progress while it ran. CLI usage
    can ignore this entirely (defaults to None, a no-op); app.py uses
    it to update shared state a background thread polls.

    Returns a dict: {"Direct (fastest)": {...}, "Compromise": {...},
    "Max Enjoyment": {...}}. avg_enjoyment values are 0-1 internally —
    convert with score_module.to_ten() wherever displayed.
    """
    import time as _time
    t_start = _time.time()

    def _report(stage):
        if progress_callback:
            progress_callback(stage)

    prefs = {**JOURNEY_PREFERENCE_DEFAULTS, **(preferences or {})}

    simple_provisional = {wid: s["enjoyment_score"] for wid, s in provisional_scores.items()}

    direct_path, direct_time_s, direct_distance_m, direct_way_info = path_module.dijkstra_shortest_path(
        graph, start_node, end_node,
    )
    if direct_path is None:
        return {}

    compromise_multiplier = path_module.time_budget_multiplier_from_preference(
        direct_time_s,
        extra_time_minutes=prefs["compromise_extra_time_minutes"],
        max_time_multiplier=prefs["compromise_max_time_multiplier"],
    )
    compromise_budget = direct_time_s * compromise_multiplier

    escalation = prefs.get("max_enjoyment_multiplier_escalation")
    if escalation:
        max_enjoyment_multipliers = sorted(escalation)
    else:
        # Escalation explicitly disabled (set to None/[]) -- fall back
        # to the single fixed ceiling, old behaviour.
        single_multiplier = path_module.time_budget_multiplier_from_preference(
            direct_time_s,
            extra_time_minutes=prefs["max_enjoyment_extra_time_minutes"],
            max_time_multiplier=prefs["max_enjoyment_max_time_multiplier"],
        )
        max_enjoyment_multipliers = [single_multiplier]

    max_enjoyment_budgets = [direct_time_s * m for m in max_enjoyment_multipliers]
    widest_budget = max(compromise_budget, max(max_enjoyment_budgets))

    gen_kwargs = {"verbose": verbose}
    if prefs["num_waypoint_candidates"] is not None:
        gen_kwargs["num_waypoint_candidates"] = prefs["num_waypoint_candidates"]
    if prefs["similarity_threshold"] is not None:
        gen_kwargs["similarity_threshold"] = prefs["similarity_threshold"]

    target_features = None
    if region_features is not None and node_coords is not None:
        lats = [lat for lat, lon in node_coords.values()]
        lons = [lon for lat, lon in node_coords.values()]
        graph_bbox = {"min_lat": min(lats), "max_lat": max(lats), "min_lon": min(lons), "max_lon": max(lons)}
        _report("Building feature-targeting candidates (villages, viewpoints, historic sites)...")
        target_features = build_target_features(region_features, prefs, bbox=graph_bbox)
        gen_kwargs["node_coords"] = node_coords
        gen_kwargs["target_features"] = target_features
        if verbose:
            print(f"  Built {len(target_features)} feature-targeting candidates "
                  f"(land cover grid/ScenicOrNot/Conservation Area/historic site) to force routes through directly.")

    if verbose:
        print(f"  PHASE: generating candidates using PROVISIONAL (no-elevation) scores, "
              f"at the widest budget ({round(widest_budget / 60, 1)} min)...")
    _report("Generating candidate routes...")
    candidates = path_module.generate_candidate_routes(
        graph, start_node, end_node, simple_provisional, widest_budget, **gen_kwargs
    )
    if not any(c["source"] == "direct" for c in candidates):
        candidates.append({
            "path": direct_path, "way_info": direct_way_info, "real_time_s": direct_time_s,
            "real_distance_m": direct_distance_m,
            "avg_enjoyment": average_enjoyment_along_path(direct_way_info, provisional_scores),
            "source": "direct",
        })
    if verbose:
        print(f"  Generated {len(candidates)} diverse candidates "
              f"({round(_time.time() - t_start, 1)}s elapsed so far)")

    used_way_ids = set()
    for c in candidates:
        used_way_ids.update(w["way_id"] for w in c["way_info"])

    if verbose:
        print(f"  PHASE: refining elevation for the {len(used_way_ids)} ways actually used "
              f"by these candidates (not the whole {len(provisional_scores)}-way graph)...")
    _report(f"Refining elevation and land cover for {len(used_way_ids)} road segments...")
    refine_kwargs = {"verbose": verbose}
    if prefs["elevation_target_spacing_m"] is not None:
        refine_kwargs["target_spacing_m"] = prefs["elevation_target_spacing_m"]
    refined = score_module.refine_scores_with_elevation(used_way_ids, ways, provisional_scores, **refine_kwargs)

    final_scores = dict(provisional_scores)
    final_scores.update(refined)

    for c in candidates:
        c["avg_enjoyment"], c["best_stretch_enjoyment"], c["route_elevation_score"] = (
            fully_scored_route(c["way_info"], final_scores)
        )
        c["landcover_composition"] = landcover_composition_along_path(c["way_info"], final_scores)

    direct_avg, direct_best, direct_route_elev = fully_scored_route(direct_way_info, final_scores)
    results = {
        "Direct (fastest)": {
            "path": direct_path, "real_time_s": direct_time_s,
            "real_distance_m": direct_distance_m, "way_info": direct_way_info,
            "source": "direct", "within_budget": True,
            "avg_enjoyment": direct_avg,
            "best_stretch_enjoyment": direct_best,
            "route_elevation_score": direct_route_elev,
            "landcover_composition": landcover_composition_along_path(direct_way_info, final_scores),
        },
    }

    # NEW — tiers are selected in order (Compromise first, then Max
    # Enjoyment), and Max Enjoyment is REQUIRED to take strictly more
    # time than Compromise's own pick (min_time_s). Real-route testing
    # found the two tiers silently collapsing onto the IDENTICAL
    # candidate whenever the single best-average route happened to
    # also fit the tighter budget — defeating the point of offering
    # three genuinely different choices. Without this, Max Enjoyment
    # would just be "give me the best route" with no actual connection
    # to its own, more generous time allowance.
    compromise_result = None
    if compromise_budget is not None:
        compromise_result = path_module.select_best_candidate_within_budget(candidates, compromise_budget)
        if compromise_result is not None:
            results["Compromise"] = compromise_result

    if max_enjoyment_budgets:
        min_time_s = compromise_result["real_time_s"] if compromise_result is not None else None

        # NEW — ESCALATING ceiling: try each multiplier in ascending
        # order, stopping at the first one that finds something
        # GENUINELY BETTER than Compromise. Real-route testing found a
        # single fixed ceiling (e.g. 1.8x) reverting to Compromise even
        # when a real improvement existed just beyond that cutoff — this
        # finds the SMALLEST widening that actually pays off, rather
        # than giving up at one arbitrary cutoff or always jumping to
        # the most generous one. No extra Dijkstra cost: the candidate
        # pool was already generated once at the widest value in the
        # list, so each check here is just a cheap budget/score filter.
        max_enjoyment_result = None
        multiplier_used = None
        for multiplier, budget in zip(max_enjoyment_multipliers, max_enjoyment_budgets):
            candidate_result = path_module.select_best_candidate_within_budget(
                candidates, budget, min_time_s=min_time_s,
            )
            # Must also strictly beat Compromise's own score (see
            # below) — real-route testing found a candidate that
            # merely took MORE time without scoring BETTER, which is
            # worse than the original collapse bug this was meant to
            # fix (at least an identical repeat is harmless; a worse
            # "upgrade" actively misleads).
            if (candidate_result is not None and compromise_result is not None
                    and candidate_result["avg_enjoyment"] > compromise_result["avg_enjoyment"]):
                max_enjoyment_result = candidate_result
                multiplier_used = multiplier
                break
            elif candidate_result is not None and compromise_result is None:
                max_enjoyment_result = candidate_result
                multiplier_used = multiplier
                break

        if verbose and multiplier_used is not None and len(max_enjoyment_multipliers) > 1:
            print(f"  Max Enjoyment: found a genuine improvement at {multiplier_used}x Direct's time "
                  f"(tried {[m for m in max_enjoyment_multipliers if m <= multiplier_used]} in order).")

        if max_enjoyment_result is None and compromise_result is not None:
            # No candidate genuinely exceeds Compromise's time AND
            # actually beats its score — rather than inventing a
            # worse "longer" option just to seem different, honestly
            # report that Max Enjoyment found nothing beyond what
            # Compromise already offers, via an explicit flag (not a
            # silent duplicate, and not a misleadingly worse "upgrade").
            max_enjoyment_result = dict(compromise_result)
            max_enjoyment_result["distinct_from_compromise"] = False
        elif max_enjoyment_result is not None:
            max_enjoyment_result["distinct_from_compromise"] = True
        if max_enjoyment_result is not None:
            results["Max Enjoyment"] = max_enjoyment_result

    # NEW — corridor search: runs AFTER Compromise/Max Enjoyment are
    # picked, searching for genuine LOCAL alternatives within a
    # geographic buffer around each, rather than only ever generating
    # radically-different far-flung detours (waypoint/feature-forced)
    # — see search_within_corridor's docstring for the full reasoning.
    # Deliberately kept cheap: only 2 paths explored (Compromise and
    # Max Enjoyment, deduplicated if identical), 2 penalty scales each
    # — direct feedback was clear that this must NOT repeat the
    # earlier grid-targeting cost mistake.
    if prefs.get("corridor_search_enabled") and node_coords is not None:
        _report("Searching for local route variations within a corridor...")
        corridor_buffer_km = prefs.get("corridor_buffer_km", 1.2)
        corridor_scales = prefs.get("corridor_penalty_scales", [1000, 4000])

        seen_path_keys = set()
        paths_to_explore = {}
        for label in ("Compromise", "Max Enjoyment"):
            if label not in results:
                continue
            path_key = tuple(results[label]["path"])
            if path_key not in seen_path_keys:
                seen_path_keys.add(path_key)
                paths_to_explore[label] = results[label]["path"]

        corridor_candidates = []
        for label, path in paths_to_explore.items():
            corridor_candidates.extend(search_within_corridor(
                graph, node_coords, start_node, end_node, path,
                simple_provisional, buffer_km=corridor_buffer_km,
                penalty_scales=corridor_scales, verbose=verbose,
            ))

        if corridor_candidates:
            # Refine any genuinely NEW ways these corridor candidates
            # use that aren't already in final_scores — likely a
            # SMALL set, since the corridor overlaps geographically
            # with the already-explored route (and OS land cover's
            # tile cache means much of this is probably already warm).
            new_way_ids = {
                w["way_id"] for c in corridor_candidates for w in c["way_info"]
                if w["way_id"] not in final_scores
            }
            if new_way_ids:
                if verbose:
                    print(f"  Corridor search found {len(new_way_ids)} new way(s) needing refinement.")
                new_refined = score_module.refine_scores_with_elevation(
                    new_way_ids, ways, provisional_scores, verbose=verbose,
                )
                final_scores.update(new_refined)

            # Score each corridor candidate with the FULL, canonical
            # blend (route-level elevation included, via fully_scored_
            # route — the same function find_three_routes itself uses
            # for every other candidate) and swap in a replacement
            # only on a MEANINGFUL improvement (avoids flip-flopping
            # over negligible, noise-level differences) within a
            # reasonable time tolerance (up to 15% slower is still
            # considered, not just strictly faster-or-equal).
            #
            # CHANGED — this used to ONLY print on success, meaning
            # total silence whenever corridor search ran but didn't
            # win — direct feedback found this genuinely ambiguous
            # ("unclear whether corridor was really used"), since
            # silence looked identical to "didn't run at all." Now
            # reports the outcome for every candidate either way, with
            # the actual reason it was or wasn't accepted.
            MEANINGFUL_IMPROVEMENT_THRESHOLD = 0.02
            any_improvement_applied = False
            if verbose:
                print(f"  Corridor search generated {len(corridor_candidates)} candidate(s) — checking each "
                      f"against the current Compromise/Max Enjoyment picks:")
            for c in corridor_candidates:
                c_avg, c_best, c_route_elev = fully_scored_route(c["way_info"], final_scores)
                for label in ("Compromise", "Max Enjoyment"):
                    if label not in results:
                        continue
                    current = results[label]
                    improvement = c_avg - current["avg_enjoyment"]
                    is_better = improvement > MEANINGFUL_IMPROVEMENT_THRESHOLD
                    is_reasonable_time = c["real_time_s"] <= current["real_time_s"] * 1.15
                    if is_better and is_reasonable_time:
                        any_improvement_applied = True
                        if verbose:
                            print(f"    ACCEPTED for {label}: {c['source']} scores "
                                  f"{score_module.to_ten(c_avg)}/10 vs current "
                                  f"{score_module.to_ten(current['avg_enjoyment'])}/10 (+{score_module.to_ten(improvement)})")
                        results[label] = {
                            "path": c["path"], "way_info": c["way_info"],
                            "real_time_s": c["real_time_s"], "real_distance_m": c["real_distance_m"],
                            "source": c["source"], "within_budget": True,
                            "avg_enjoyment": c_avg, "best_stretch_enjoyment": c_best,
                            "route_elevation_score": c_route_elev,
                            "distinct_from_compromise": label == "Max Enjoyment",
                            "landcover_composition": landcover_composition_along_path(c["way_info"], final_scores),
                        }
                    elif verbose:
                        if not is_better:
                            reason = f"not better ({score_module.to_ten(c_avg)}/10 vs current {score_module.to_ten(current['avg_enjoyment'])}/10)"
                        else:
                            reason = (f"scores better ({score_module.to_ten(c_avg)}/10 vs "
                                      f"{score_module.to_ten(current['avg_enjoyment'])}/10) but too slow relative "
                                      f"to {label} ({round(c['real_time_s']/60, 1)} min vs current "
                                      f"{round(current['real_time_s']/60, 1)} min, >15% over)")
                        print(f"    rejected for {label} ({c['source']}): {reason}")
            if verbose and not any_improvement_applied:
                print(f"  Corridor search ran but found nothing that beat the existing picks "
                      f"(see candidate-by-candidate detail above) — Compromise/Max Enjoyment unchanged.")
        elif verbose:
            print(f"  Corridor search ran but found NO valid path within the corridor at all "
                  f"(the buffer may be too narrow to connect start and end here) — picks unchanged.")

    # NEW — highlights: the SPECIFIC NAMED features each winning route
    # actually passes near, not just its score. Computed only for the
    # 3 final tiers (not all candidates) — this involves a real
    # distance check against every conservation area/ScenicOrNot spot/
    # historic site in the bbox, which is fine for 3 routes but not
    # something to repeat needlessly across a whole candidate pool.
    _report("Finding highlights along each route...")
    for label in list(results.keys()):
        if label.startswith("_"):
            continue
        r = results[label]
        r["highlights"] = find_route_highlights(r["way_info"], ways, region_features or {})

    # NEW (Milestone 2) — EXPERIMENTAL composite scores (Drama, Charm,
    # Uniqueness) — see compute_experimental_scores' docstring for the
    # full grounding/honesty notes. Attached under its own
    # "experimental" key specifically so callers (CLI print, app.py,
    # map_export.py) can choose to label/style these distinctly from
    # the validated core score, per the plan's explicit requirement to
    # mark these as experimental from the start.
    _report("Computing experimental scores (Drama/Charm/Uniqueness)...")
    rf = region_features or {}
    for label in list(results.keys()):
        if label.startswith("_"):
            continue
        r = results[label]
        villages_count = count_qualifying_features(
            r["way_info"], ways, rf.get("conservation_areas", []), HIGHLIGHT_DISTANCE_THRESHOLDS_M["villages"],
        )
        historic_count = count_qualifying_features(
            r["way_info"], ways, rf.get("historic", []), HIGHLIGHT_DISTANCE_THRESHOLDS_M["historic"],
        )
        r["experimental"] = compute_experimental_scores(
            r["way_info"], final_scores, r.get("landcover_composition", {}),
            r.get("route_elevation_score", 0.0), villages_count=villages_count, historic_count=historic_count,
        )

    # NEW (Milestone 2) — food/drink AI recommendations, OPT-IN ONLY
    # (see JOURNEY_PREFERENCE_DEFAULTS' "enable_food_drink_ai" comment
    # for why this defaults to off — the one part of this pipeline
    # with a real per-call cost, everything else being free data
    # sources). Venues are fetched ONCE for the whole bbox (free,
    # OSM-sourced), then filtered down to whatever's actually near
    # EACH specific route before that route's own AI call. Wrapped
    # defensively — unlike core route-finding, a failure here (a
    # missing anthropic package, network issue, API error) should
    # degrade to "no food/drink recommendations," never break the
    # actual route result.
    if prefs.get("enable_food_drink", True) and node_coords is not None:
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
                      f"core route results are unaffected.")

    # NEW — the full candidate pool, attached under a clearly-marked
    # key (not a real tier) so map_export.py (or any future
    # diagnostic) can visualize the GENUINE search coverage, not just
    # the 3 winning tiers — directly useful for questions like "is
    # Max Enjoyment==Compromise a fair result, or is the candidate
    # pool too narrow," which text output alone can't really answer.
    # Existing callers expecting just the 3 named tiers are
    # unaffected — they simply never look at this key.
    results["_diagnostics"] = {"all_candidates": candidates}

    if verbose:
        print(f"  find_three_routes total: {round(_time.time() - t_start, 1)}s\n")

    return results


def run_full_pipeline(start_lat, start_lon, end_lat, end_lon, margin_degrees=None,
                       preferences=None, verbose=True, progress_callback=None):
    """
    NEW — the full, reusable orchestration: fetch roads for the area
    around start/end, build the graph, score it, find the nearest
    start/end nodes, and run find_three_routes.

    Factored out specifically so BOTH this file's own __main__ (CLI
    testing with hardcoded locations) AND app.py (the interactive web
    UI, arbitrary user-clicked locations) share ONE implementation,
    rather than duplicating this same sequence of calls in two places
    — the same "generalize it, don't duplicate it" principle already
    applied elsewhere this session, used proactively here rather than
    after discovering two diverging copies later.

    CHANGED — margin_degrees now defaults to None, meaning "compute a
    sensible margin from the trip's own length" rather than a fixed
    0.03 for every trip regardless of distance. Direct feedback found
    a real test case where the most direct route (confirmed against
    Google Maps) was missing ENTIRELY — the old fixed margin was too
    tight to even include the road network it needed. A first revision
    (0.08 floor / 0.5x multiplier) then turned out a bit too generous
    after testing — current default is max(0.055, 0.25 * straight-line
    distance in degrees), splitting the difference between the
    original and the first revision rather than guessing again from
    scratch. Pass an explicit margin_degrees to override entirely if
    you want a specific value. This may ALSO explain a separate,
    persistent observation (Max Enjoyment always matching Compromise)
    — if the genuinely best detour sits just outside the search area,
    no amount of candidate-generation cleverness within a too-narrow
    bbox would ever find it. Worth re-testing both together, not
    assumed fixed without checking.

    progress_callback: optional — see find_three_routes' docstring;
    passed straight through, plus called directly here for the
    earlier fetch/build/score stages it doesn't otherwise cover.

    Returns (route_results, node_coords, start_node, end_node,
    provisional_scores, bbox) — route_results is find_three_routes'
    own return shape, including the "_diagnostics" key. bbox is the
    same {"min_lat","max_lat","min_lon","max_lon"} dict used for the
    Overpass fetch, handy for rendering on a map (see map_export.py).
    """
    def _report(stage):
        if progress_callback:
            progress_callback(stage)

    if margin_degrees is None:
        straight_line_distance = ((end_lat - start_lat) ** 2 + (end_lon - start_lon) ** 2) ** 0.5
        # CHANGED — split the difference: direct feedback found the
        # previous widening (0.08 floor / 0.5x multiplier) a bit too
        # generous after testing. Halved both the floor and the
        # multiplier — splitting the gap between the original fixed
        # 0.03 and the first revision, not a fresh guess.
        margin_degrees = max(0.055, 0.25 * straight_line_distance)
        if verbose:
            print(f"  Using a computed bbox margin of {round(margin_degrees, 3)} degrees "
                  f"(trip's straight-line distance: {round(straight_line_distance, 3)} degrees).")

    bbox = {
        "min_lat": min(start_lat, end_lat) - margin_degrees,
        "max_lat": max(start_lat, end_lat) + margin_degrees,
        "min_lon": min(start_lon, end_lon) - margin_degrees,
        "max_lon": max(start_lon, end_lon) + margin_degrees,
    }

    _report("Fetching road network data from OpenStreetMap...")
    if verbose:
        print("Fetching real road network data...")
    overpass_data = fetch_module.fetch_all_roads_in_bbox(bbox)

    _report("Building the routable graph...")
    if verbose:
        print("\nBuilding routable graph...")
    graph, node_coords = build_module.build_graph_from_overpass_data(overpass_data)
    if verbose:
        print(f"  Graph has {len(node_coords)} nodes, {sum(len(e) for e in graph.values())} directed edges.\n")

    _report("Scoring every road (villages, viewpoints, historic sites)...")
    if verbose:
        print("PHASE 1: scoring every way in the graph PROVISIONALLY (no whole-graph elevation fetch)...")
    provisional_scores, ways, region_features = score_module.score_all_ways_provisional(graph, node_coords)
    if verbose:
        print(f"  Scored {len(provisional_scores)} unique ways (provisionally).\n")

        all_provisional_values = [s["enjoyment_score"] for s in provisional_scores.values()]
        print(f"Provisional enjoyment score distribution across all {len(all_provisional_values)} ways "
              f"(out of 10):")
        print(f"  Min: {score_module.to_ten(min(all_provisional_values))}")
        print(f"  Max: {score_module.to_ten(max(all_provisional_values))}")
        print(f"  Mean: {score_module.to_ten(sum(all_provisional_values) / len(all_provisional_values))}")
        print()

        top_ways = sorted(provisional_scores.items(), key=lambda kv: -kv[1]["enjoyment_score"])[:10]
        print("Top 10 highest PROVISIONAL-enjoyment ways in the whole graph (out of 10; "
              "no elevation yet — that's refined later, only for ways actually used):")
        for way_id, scores in top_ways:
            print(f"  way_id={way_id}: enjoyment={score_module.to_ten(scores['enjoyment_score'])}, "
                  f"driving={score_module.to_ten(scores['driving_enjoyment'])}, "
                  f"scenery={score_module.to_ten(scores['scenery'])}, "
                  f"interest={score_module.to_ten(scores['interest'])}")
        print()

    start_node, start_dist_m = path_module.find_nearest_node(start_lat, start_lon, node_coords)
    end_node, end_dist_m = path_module.find_nearest_node(end_lat, end_lon, node_coords)

    route_results = find_three_routes(
        graph, start_node, end_node, provisional_scores, ways,
        region_features=region_features, node_coords=node_coords,
        preferences=preferences, verbose=verbose, progress_callback=progress_callback,
    )

    _report("Done.")
    return route_results, node_coords, start_node, end_node, provisional_scores, bbox


if __name__ == "__main__":
    TILLINGTON = {"name": "Tillington", "lat": 50.9896, "lon": -0.6313}
    HASLEMERE = {"name": "Haslemere", "lat": 51.089, "lon": -0.710}

    route_results, node_coords, start_node, end_node, _provisional_scores, run_bbox = run_full_pipeline(
        TILLINGTON["lat"], TILLINGTON["lon"], HASLEMERE["lat"], HASLEMERE["lon"],
    )

    results = {}
    for label, r in route_results.items():
        if label.startswith("_"):
            continue  # skip diagnostic-only keys (e.g. "_diagnostics"), not a real route tier
        avg_enjoyment = r["avg_enjoyment"]
        best_stretch = r.get("best_stretch_enjoyment", 0.0)
        road_sequence = summarize_road_sequence(r["way_info"])
        composition = r.get("landcover_composition", {})

        results[label] = {
            "distance_km": round(r["real_distance_m"] / 1000, 2),
            "time_min": round(r["real_time_s"] / 60, 1),
            "avg_enjoyment_10": score_module.to_ten(avg_enjoyment),
            "best_stretch_10": score_module.to_ten(best_stretch),
            "roads": road_sequence,
            "source": r["source"],
            "within_budget": r["within_budget"],
            "distinct_from_compromise": r.get("distinct_from_compromise"),
            "landcover_composition_pct": {k: round(v * 100, 1) for k, v in composition.items()},
            "highlights": r.get("highlights", {}),
            "experimental": r.get("experimental", {}),
        }

        distinct_note = ""
        if label == "Max Enjoyment" and r.get("distinct_from_compromise") is False:
            distinct_note = " — NOTE: no candidate beyond Compromise's time was better; repeating it honestly rather than inventing a worse 'longer' option"

        print(f"{'=' * 60}")
        print(f"{label} (source={r['source']}, within_budget={r['within_budget']}){distinct_note}")
        print(f"{'=' * 60}")
        print(f"Distance: {results[label]['distance_km']} km")
        print(f"Time: {results[label]['time_min']} minutes")
        print(f"Average enjoyment score along route: {results[label]['avg_enjoyment_10']}/10")
        print(f"Best 30% of route (peak-weighted):    {results[label]['best_stretch_10']}/10")
        comp_pct = results[label]["landcover_composition_pct"]
        comp_str = ", ".join(f"{k.replace('_', ' ').title()} {v}%" for k, v in comp_pct.items() if v > 0)
        print(f"Land cover: {comp_str if comp_str else '(no composition data for this route)'}")
        highlight_lines = format_highlights_text(results[label]["highlights"])
        if highlight_lines:
            print("Highlights (what this route actually passes near):")
            for line in highlight_lines:
                print(f"  - {line}")
        else:
            print("Highlights: none of the named features in this area were close enough to this specific route.")
        exp = results[label]["experimental"]
        if exp:
            print(f"[EXPERIMENTAL — not yet validated against real human perception, see Decisions Log]")
            print(f"  Drama: {score_module.to_ten(exp.get('drama', 0))}/10, "
                  f"Charm: {score_module.to_ten(exp.get('charm', 0))}/10, "
                  f"Uniqueness: {score_module.to_ten(exp.get('uniqueness', 0))}/10")
        print("Roads:")
        for road in road_sequence:
            print(f"  - {road}")
        print()

    print(f"{'=' * 60}")
    print("SUMMARY (scores out of 10 — average / best-30%-stretch)")
    print(f"{'=' * 60}")
    for label in ["Direct (fastest)", "Compromise", "Max Enjoyment"]:
        if label in results:
            r = results[label]
            print(f"{label}: {r['time_min']} min, {r['distance_km']} km, "
                  f"enjoyment={r['avg_enjoyment_10']}/10 (best stretch: {r['best_stretch_10']}/10), "
                  f"source={r['source']}")

    print()
    map_export_module.export_routes_to_html(
        route_results, node_coords, output_path="route_map.html", show_all_candidates=True, bbox=run_bbox,
    )
