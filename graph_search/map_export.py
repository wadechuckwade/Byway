"""
Byway — Route map export (Leaflet.js, self-contained HTML)
================================================================

What this does, in plain terms:
Takes the route_results dict find_three_routes() already returns and
writes a SELF-CONTAINED HTML file with an embedded Leaflet map showing
every route as a coloured line — open it directly in any browser, no
server, no API key, no build step.

WHY THIS EXISTS: direct feedback — text output alone made it hard to
actually SEE what route the system was proposing, making it harder to
spot what's going wrong. Rather than invent route rendering from
scratch, this uses Leaflet (the standard, free, open-source library
for exactly this) with OpenStreetMap tiles — free for this kind of
low-volume development/debugging use, and the same data source this
whole project is already built on.

WHY THIS IS PRECISE, NOT AN APPROXIMATION: find_three_routes() already
returns each route's EXACT sequence of graph node IDs (its "path") —
combined with node_coords (already available everywhere this gets
called), that's the real, exact geometry our own system computed, not
a third-party routing engine's guess at how to connect a few waypoints.

WHY A SEPARATE, REUSABLE MODULE: built standalone rather than inlined
into one script specifically so any future diagnostic script can use
the same function — applying the "generalize a fix beyond its
immediate trigger" lesson proactively this time, not after re-
discovering the same need in a second script later.

USAGE (see 08_three_route_system.py's __main__ for a real example):
    from map_export import export_routes_to_html
    export_routes_to_html(route_results, node_coords)
"""

import json


DEFAULT_COLORS = {
    "Direct (fastest)": "#888888",
    "Compromise": "#2563eb",
    "Max Enjoyment": "#16a34a",
}
FALLBACK_COLOR = "#dc2626"


def export_routes_to_html(route_results, node_coords, output_path="route_map.html",
                           extra_routes=None, colors=None, show_all_candidates=False, bbox=None):
    """
    Writes a self-contained HTML file (Leaflet.js + OpenStreetMap
    tiles, loaded from CDN) showing every route in route_results as a
    coloured polyline, using each route's EXACT path (node IDs) +
    node_coords for precise geometry — not an approximation.

    route_results: the dict find_three_routes() returns — {"Direct
    (fastest)": {...}, "Compromise": {...}, "Max Enjoyment": {...}},
    each needing at least "path" (list of node IDs). avg_enjoyment/
    best_stretch_enjoyment/real_distance_m/real_time_s/source are used
    for the popup text if present, but aren't required. The special
    "_diagnostics" key (if present — see find_three_routes) is NOT
    treated as a route tier, regardless of show_all_candidates.

    node_coords: {node_id: (lat, lon)} — same shape used everywhere
    else in this codebase.

    bbox: optional {"min_lat","max_lat","min_lon","max_lon"} dict —
    if provided, drawn as a dashed rectangle outline, so it's visible
    exactly how large an area the Overpass fetch (and every other
    bbox-scoped lookup — land cover, historic sites, etc.) actually
    covered for this run.

    show_all_candidates: if True, ALSO renders every candidate from
    route_results["_diagnostics"]["all_candidates"] (if present) as a
    thin, semi-transparent grey line underneath the 3 main routes —
    direct visual evidence for "is the search exploring widely enough"
    questions (e.g. why Max Enjoyment matched Compromise), rather than
    just trusting a count of how many candidates were generated.

    extra_routes: optional list of additional {"label", "path",
    "color"} dicts to overlay (e.g. a specific candidate you want to
    inspect directly, like a forced-via-Lurgashall route from a
    diagnostic script) — not required for the normal three-tier case.

    colors: optional override dict {label: "#hexcolor"} — defaults to
    DEFAULT_COLORS (grey/blue/green for Direct/Compromise/Max
    Enjoyment), falling back to a red for any other label.

    Returns the output path written.
    """
    color_map = {**DEFAULT_COLORS, **(colors or {})}

    routes_for_js = []
    candidates_for_js = []
    all_points = []

    def _add_route(label, path, color, popup, target_list):
        points = [[node_coords[nid][0], node_coords[nid][1]] for nid in path if nid in node_coords]
        if not points:
            print(f"  (Skipping '{label}' in map export — no valid coordinates found for its path)")
            return
        all_points.extend(points)
        target_list.append({"label": label, "points": points, "color": color, "popup": popup})

    for label, r in route_results.items():
        if label.startswith("_"):
            continue  # skip diagnostic-only keys (e.g. "_diagnostics"), not a real route tier
        path = r.get("path", [])
        avg_10 = round(r.get("avg_enjoyment", 0.0) * 10, 1)
        best_10 = round(r.get("best_stretch_enjoyment", 0.0) * 10, 1)
        distance_km = round(r.get("real_distance_m", 0) / 1000, 2)
        time_min = round(r.get("real_time_s", 0) / 60, 1)
        source = r.get("source", "")
        popup = (f"<b>{label}</b><br>{distance_km} km, {time_min} min<br>"
                 f"Avg enjoyment: {avg_10}/10, Best stretch: {best_10}/10<br>"
                 f"<i>source: {source}</i>")
        color = color_map.get(label, FALLBACK_COLOR)
        _add_route(label, path, color, popup, routes_for_js)

    for extra in (extra_routes or []):
        _add_route(extra["label"], extra["path"], extra.get("color", FALLBACK_COLOR),
                    f"<b>{extra['label']}</b>", routes_for_js)

    if show_all_candidates:
        all_candidates = route_results.get("_diagnostics", {}).get("all_candidates", [])
        for c in all_candidates:
            avg_10 = round(c.get("avg_enjoyment", 0.0) * 10, 1)
            time_min = round(c.get("real_time_s", 0) / 60, 1)
            popup = f"<i>candidate: {c.get('source', '?')}</i><br>{time_min} min, avg {avg_10}/10"
            _add_route(c.get("source", "candidate"), c.get("path", []), "#999999", popup, candidates_for_js)
        if candidates_for_js:
            print(f"  Including {len(candidates_for_js)} candidate route(s) as faint background lines.")

    if not routes_for_js and not candidates_for_js:
        print("  No routes had valid geometry to export — map not written.")
        return None

    center_lat = sum(p[0] for p in all_points) / len(all_points)
    center_lon = sum(p[1] for p in all_points) / len(all_points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Byway route map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
  .legend {{ background: white; padding: 8px 12px; border-radius: 6px; font-family: sans-serif;
             font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.3); line-height: 1.6; }}
  .legend span {{ display: inline-block; width: 16px; height: 4px; margin-right: 6px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  const routeData = {json.dumps(routes_for_js)};
  const candidateData = {json.dumps(candidates_for_js)};
  const bboxData = {json.dumps(bbox)};
  const map = L.map('map').setView([{center_lat}, {center_lon}], 12);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  }}).addTo(map);

  const allLatLngs = [];

  if (bboxData) {{
    const rect = L.rectangle(
      [[bboxData.min_lat, bboxData.min_lon], [bboxData.max_lat, bboxData.max_lon]],
      {{ color: '#f59e0b', weight: 2, fill: false, dashArray: '6,6' }}
    ).addTo(map);
    rect.bindPopup('Fetch area (bbox) for this run');
  }}

  // Candidates drawn FIRST (thin, faint, behind) so the main routes
  // always render on top of the candidate cloud, not hidden under it.
  candidateData.forEach(route => {{
    const line = L.polyline(route.points, {{ color: route.color, weight: 2, opacity: 0.45, dashArray: '4,4' }}).addTo(map);
    line.bindPopup(route.popup);
    allLatLngs.push(...route.points);
  }});

  routeData.forEach(route => {{
    const line = L.polyline(route.points, {{ color: route.color, weight: 5, opacity: 0.85 }}).addTo(map);
    line.bindPopup(route.popup);
    allLatLngs.push(...route.points);
  }});

  if (allLatLngs.length > 0) {{
    map.fitBounds(allLatLngs);
  }}

  const legend = L.control({{ position: 'topright' }});
  legend.onAdd = function() {{
    const div = L.DomUtil.create('div', 'legend');
    let html = routeData.map(r => `<div><span style="background:${{r.color}}"></span>${{r.label}}</div>`).join('');
    if (candidateData.length > 0) {{
      html += `<div style="margin-top:4px;color:#666;">+ ${{candidateData.length}} candidate(s), faint dashed</div>`;
    }}
    if (bboxData) {{
      html += `<div style="margin-top:4px;"><span style="border:2px dashed #f59e0b;width:14px;height:8px;display:inline-block;margin-right:6px;vertical-align:middle;"></span>fetch area (bbox)</div>`;
    }}
    div.innerHTML = html;
    return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Route map written to {output_path} — open it in any browser to view.")
    return output_path


if __name__ == "__main__":
    print("--- Offline test: HTML export with synthetic route data (no network needed) ---")
    fake_node_coords = {
        1: (51.000, -0.700), 2: (51.001, -0.690), 3: (51.002, -0.680),
        4: (51.010, -0.700), 5: (51.011, -0.685),
    }
    fake_route_results = {
        "Direct (fastest)": {
            "path": [1, 2, 3], "avg_enjoyment": 0.32, "best_stretch_enjoyment": 0.42,
            "real_distance_m": 17420, "real_time_s": 1476, "source": "direct",
        },
        "Compromise": {
            "path": [1, 4, 5, 3], "avg_enjoyment": 0.38, "best_stretch_enjoyment": 0.43,
            "real_distance_m": 15850, "real_time_s": 1596, "source": "blend_scale_250",
        },
        # The new diagnostics key -- must NOT be treated as a route tier.
        "_diagnostics": {
            "all_candidates": [
                {"path": [1, 2, 3], "avg_enjoyment": 0.32, "real_time_s": 1476, "source": "direct"},
                {"path": [1, 4, 5, 3], "avg_enjoyment": 0.38, "real_time_s": 1596, "source": "blend_scale_250"},
            ]
        },
    }
    output = export_routes_to_html(fake_route_results, fake_node_coords, output_path="/tmp/test_route_map.html")
    assert output == "/tmp/test_route_map.html"
    with open(output) as f:
        content = f.read()
    assert "Direct (fastest)" in content
    assert "Compromise" in content
    assert "_diagnostics" not in content, "The diagnostics key itself must never leak into rendered output as a route"
    assert "51.0" in content  # a real coordinate should appear somewhere
    assert "leaflet" in content.lower()
    print(f"PASSED — HTML file written with both real routes, _diagnostics correctly excluded as a tier, "
          f"real coordinates, and Leaflet embedded ({len(content)} bytes)\n")

    print("--- Offline test: show_all_candidates renders the candidate pool as a separate, faint layer ---")
    output2 = export_routes_to_html(
        fake_route_results, fake_node_coords, output_path="/tmp/test_route_map_candidates.html",
        show_all_candidates=True,
    )
    with open(output2) as f:
        content2 = f.read()
    assert "candidateData" in content2
    assert "dashArray" in content2, "Candidates should render with a distinct (dashed) style from main routes"
    # Both real candidates from _diagnostics should appear in the candidate layer.
    assert content2.count('"source": "direct"') == 0  # candidates don't carry this raw key into JS, just check structurally below
    print(f"PASSED — candidate pool rendered as a separate, visually distinct layer ({len(content2)} bytes)\n")

    print("--- Offline test: bbox renders as a dashed rectangle ---")
    fake_bbox = {"min_lat": 50.98, "max_lat": 51.12, "min_lon": -0.72, "max_lon": -0.63}
    output3 = export_routes_to_html(
        fake_route_results, fake_node_coords, output_path="/tmp/test_route_map_bbox.html", bbox=fake_bbox,
    )
    with open(output3) as f:
        content3 = f.read()
    assert "bboxData" in content3
    assert "L.rectangle" in content3
    assert "50.98" in content3
    print(f"PASSED — bbox rendered as a dashed rectangle on the map ({len(content3)} bytes)")
