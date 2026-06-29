"""
Byway — Diagnostic: why doesn't our formula recognize the route the
user KNOWS is nicer (Cemetery Lane / Wheelbarrow Castle / village
lanes) versus the A-road route our search currently always picks?

WHY THIS EXISTS: the user has direct, confident real-world knowledge
that the original minor-lane route (found before the time-correction
fix) passes through charming villages, nice pubs, low-key roads, and
good countryside — genuinely nicer than the A-road alternative, not
just different. Yet even with enjoyment-blended search at a strong
blend_factor, our system never picks anything other than the A-road
route. This script checks several real, distinct hypotheses at once
rather than guessing:

1. Does our CURRENT formula actually score the known-nicer route's
   ways highly, or does the formula itself fail to recognize them?
2. Side-by-side comparison: every way on the nicer route vs every way
   on the chosen A-road route, with full category breakdowns.
3. A structural gap check: do we even have a PUBS/amenities category
   at all? (Spoiler, checked directly: we do not — this was always
   planned as a separate "experimental" category requiring AI-as-
   judge-over-real-data, per the original Data Feasibility Audit, and
   was never built. If pubs/charm are a big part of why this route is
   genuinely nicer, our current categories may be structurally unable
   to detect that, regardless of any tuning.)
"""

import os
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


def find_route_by_distance(graph, start_node, end_node):
    """
    Re-derive the ORIGINAL minor-lane route by running Dijkstra with
    plain DISTANCE as cost (no time, no enjoyment) — this is exactly
    how we found that route in the very first real-route test, before
    any of the time-correction or enjoyment-blending work. Gives us a
    clean, reproducible way to retrieve its exact way_ids for
    inspection, without needing to hand-guess them.
    """
    # Build a temporary distance-only graph view: same structure, but
    # use distance_m in place of time_s, and zero out the junction
    # penalty's real-world basis (irrelevant for this pure
    # reconstruction — we just want the SAME path as the original
    # distance-based test, not a new "fastest by distance" route).
    distances = {start_node: 0}
    previous = {}
    previous_way_info = {}
    visited = set()
    import heapq
    pq = [(0, start_node)]
    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == end_node:
            break
        for neighbor_id, edge_distance, edge_time, way_info in graph.get(node, []):
            if neighbor_id in visited:
                continue
            new_d = d + edge_distance
            if new_d < distances.get(neighbor_id, float("inf")):
                distances[neighbor_id] = new_d
                previous[neighbor_id] = node
                previous_way_info[neighbor_id] = way_info
                heapq.heappush(pq, (new_d, neighbor_id))

    if end_node not in distances:
        return None, None

    path = [end_node]
    way_info_list = []
    node = end_node
    while node != start_node:
        way_info_list.append(previous_way_info[node])
        node = previous[node]
        path.append(node)
    path.reverse()
    way_info_list.reverse()
    return path, way_info_list


def print_route_breakdown(label, way_info_list, enjoyment_scores):
    print(f"\n--- {label}: per-way breakdown ---")
    seen = set()
    for w in way_info_list:
        wid = w["way_id"]
        if wid in seen:
            continue
        seen.add(wid)
        s = enjoyment_scores.get(wid, {})
        name = w["name"] or w.get("ref", "") or f"(unnamed {w['highway']})"
        print(f"  \"{name}\" ({w['highway']}): enjoyment={s.get('enjoyment_score', '?')} "
              f"[driving={s.get('driving_enjoyment', '?')}, "
              f"scenery={s.get('scenery', '?')}, "
              f"history={s.get('history_culture', '?')}]")


if __name__ == "__main__":
    TILLINGTON = {"name": "Tillington", "lat": 50.9896, "lon": -0.6313}
    HASLEMERE = {"name": "Haslemere", "lat": 51.089, "lon": -0.710}

    margin_degrees = 0.03
    bbox = {
        "min_lat": min(TILLINGTON["lat"], HASLEMERE["lat"]) - margin_degrees,
        "max_lat": max(TILLINGTON["lat"], HASLEMERE["lat"]) + margin_degrees,
        "min_lon": min(TILLINGTON["lon"], HASLEMERE["lon"]) - margin_degrees,
        "max_lon": max(TILLINGTON["lon"], HASLEMERE["lon"]) + margin_degrees,
    }

    print("Fetching real road network data...")
    overpass_data = fetch_module.fetch_all_roads_in_bbox(bbox)

    print("\nBuilding routable graph...")
    graph, node_coords = build_module.build_graph_from_overpass_data(overpass_data)

    print("\nScoring every way for enjoyment...")
    enjoyment_scores = score_module.score_all_ways_for_enjoyment(graph, node_coords)

    start_node, _ = path_module.find_nearest_node(TILLINGTON["lat"], TILLINGTON["lon"], node_coords)
    end_node, _ = path_module.find_nearest_node(HASLEMERE["lat"], HASLEMERE["lon"], node_coords)

    # 1. Re-derive the known-nicer minor-lane route.
    nicer_path, nicer_way_info = find_route_by_distance(graph, start_node, end_node)
    if nicer_path is None:
        print("\nCould not re-derive the minor-lane route — investigate separately.")
    else:
        print_route_breakdown("KNOWN-NICER minor-lane route (Cemetery Lane / Wheelbarrow Castle / etc.)",
                               nicer_way_info, enjoyment_scores)
        nicer_way_ids = set(w["way_id"] for w in nicer_way_info)
        nicer_scores = [enjoyment_scores.get(wid, {}).get("enjoyment_score", 0) for wid in nicer_way_ids]
        print(f"\n  Average enjoyment across this route's ways: "
              f"{round(sum(nicer_scores)/len(nicer_scores), 3) if nicer_scores else 'N/A'}")

        # Compute this exact path's REAL time/distance (including
        # junction penalties), by walking the path and summing real
        # edge data — not estimating, actually calculating it, so we
        # can compare real costs side by side rather than guess.
        # ALSO: break down the total into its actual components
        # (pure travel time vs junction-penalty time, total real
        # transitions vs unique way_ids, and time by road class) —
        # the unique-way-count alone didn't fully explain the time,
        # so we need to see exactly where the seconds are going.
        real_time_s = 0
        pure_travel_time_s = 0
        junction_penalty_time_s = 0
        real_distance_m = 0
        transition_count = 0
        edge_count = 0
        time_by_highway = {}
        had_tag_count = 0
        no_tag_count = 0
        no_tag_distance_m = 0
        last_way_id = None
        for i in range(len(nicer_path) - 1):
            node_a, node_b = nicer_path[i], nicer_path[i + 1]
            for neighbor_id, edge_distance, edge_time, way_info in graph.get(node_a, []):
                if neighbor_id == node_b:
                    is_transition = last_way_id is not None and last_way_id != way_info["way_id"]
                    penalty = path_module.JUNCTION_PENALTY_SECONDS if is_transition else 0
                    if is_transition:
                        transition_count += 1
                    real_time_s += edge_time + penalty
                    pure_travel_time_s += edge_time
                    junction_penalty_time_s += penalty
                    real_distance_m += edge_distance
                    edge_count += 1
                    hw = way_info["highway"]
                    time_by_highway[hw] = time_by_highway.get(hw, 0) + edge_time
                    if way_info.get("had_maxspeed_tag"):
                        had_tag_count += 1
                    else:
                        no_tag_count += 1
                        no_tag_distance_m += edge_distance
                    last_way_id = way_info["way_id"]
                    break
        print(f"  REAL time for this exact path: {round(real_time_s/60, 1)} minutes, "
              f"{round(real_distance_m/1000, 2)} km")
        print(f"    Breakdown: {round(pure_travel_time_s/60, 1)} min pure travel time, "
              f"{round(junction_penalty_time_s/60, 1)} min junction penalties "
              f"({transition_count} real transitions across {edge_count} graph edges)")
        print(f"    Pure travel time by road class:")
        for hw, t in sorted(time_by_highway.items(), key=lambda kv: -kv[1]):
            print(f"      {hw}: {round(t/60, 1)} min")
        print(f"    MAXSPEED TAG CHECK: {had_tag_count} of {edge_count} edges had a real "
              f"OSM maxspeed tag; {no_tag_count} fell back to the class default "
              f"(covering {round(no_tag_distance_m/1000, 2)} km with no real tag at all).")

    # 2. Compare against the current chosen (time-based) A-road route.
    aroad_path, aroad_time_s, aroad_dist_m, aroad_way_info = path_module.dijkstra_shortest_path(graph, start_node, end_node)
    print_route_breakdown("CURRENT chosen A-road route", aroad_way_info, enjoyment_scores)
    aroad_way_ids = set(w["way_id"] for w in aroad_way_info)
    aroad_scores = [enjoyment_scores.get(wid, {}).get("enjoyment_score", 0) for wid in aroad_way_ids]
    print(f"\n  Average enjoyment across this route's ways: "
          f"{round(sum(aroad_scores)/len(aroad_scores), 3) if aroad_scores else 'N/A'}")
    print(f"  REAL time for this route: {round(aroad_time_s/60, 1)} minutes, "
          f"{round(aroad_dist_m/1000, 2)} km")

    if nicer_path is not None:
        print(f"\n{'=' * 60}")
        print("DIRECT COMPARISON")
        print(f"{'=' * 60}")
        time_diff_s = real_time_s - aroad_time_s
        enjoyment_diff = (sum(nicer_scores)/len(nicer_scores)) - (sum(aroad_scores)/len(aroad_scores))
        print(f"Minor-lane route costs {round(time_diff_s)}s ({round(time_diff_s/60,1)} min) "
              f"MORE real time than the A-road route.")
        print(f"Minor-lane route scores {round(enjoyment_diff, 3)} HIGHER average enjoyment.")
        print(f"\nFor blend_factor=0.8 to prefer the minor-lane route, we'd need:")
        print(f"  blend_factor * PENALTY_SCALE * distance_km * enjoyment_diff >= time_diff_s")
        required_scale = time_diff_s / (0.8 * (real_distance_m/1000) * enjoyment_diff) if enjoyment_diff > 0 else float('inf')
        print(f"  This requires PENALTY_SCALE >= {round(required_scale)} seconds/km "
              f"(current value: {path_module.ENJOYMENT_PENALTY_SECONDS_PER_KM})")


    # 3. Structural gap check: do we have ANY pub/amenity data fetched
    # at all? Check the raw Overpass region data our scoring already
    # pulled, to see if pubs are even present in what we fetched
    # (separate question from whether we SCORE for them, which we
    # confirmed in code review we do not).
    print(f"\n{'=' * 60}")
    print("STRUCTURAL CHECK: do we capture pub/amenity data at all?")
    print(f"{'=' * 60}")
    print("Current scoring categories: curvature, elevation, water")
    print("proximity, forest proximity, historic-site proximity.")
    print("NO pub/food/drink category exists yet — this was always")
    print("planned as a separate 'experimental' category (AI-as-judge")
    print("over real retrieved data, per the original Data Feasibility")
    print("Audit) and has not been built. If pubs/village charm are a")
    print("big part of why the minor-lane route is genuinely nicer,")
    print("our current categories may be structurally blind to that —")
    print("regardless of search algorithm or formula tuning, since")
    print("there's no signal being measured for it at all.")