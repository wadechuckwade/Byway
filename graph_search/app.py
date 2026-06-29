"""
Byway — Minimal interactive testing UI
===========================================

What this does, in plain terms:
A small local Flask app: click two points on a map, hit "Find
Routes," see the three results (plus the full candidate pool, faint)
rendered live on the same map — no more editing hardcoded coordinates
in a script and re-running from the terminal to test a new location.

WHY THIS EXISTS: direct feedback — testing a new start/end currently
meant editing Python constants and re-running. This reuses 08_three_
route_system.py's run_full_pipeline() directly (the same orchestration
the CLI script uses, factored out specifically so this app wouldn't
need its own, second copy of fetch->build->score->find_three_routes).

HONEST SCOPE: this is a debugging tool, not a step toward the actual
production app — no auth, no persistence, no error recovery beyond a
basic message, single-request-at-a-time (each click-through runs the
full real pipeline: Overpass, elevation, land cover, ScenicOrNot,
Historic England, Conservation Areas — anywhere from a few seconds to
a couple of minutes depending on the area and how much is already
cached). Scoped deliberately small, matching what's actually needed
right now: a way to SEE what a route looks like without re-running
the CLI script with edited constants every time.

RUN:
    pip install flask --break-system-packages   (if not already installed)
    python graph_search/app.py
Then open http://localhost:5000 in a browser.

Network note: needs real internet access for the underlying pipeline
— run in Codespaces, not in Claude's sandboxed tool environment.
"""

import os
import time
import logging
import threading
import importlib.util

from flask import Flask, request, jsonify, render_template_string

# NEW — direct feedback: Werkzeug's default per-request access log
# ("GET /progress HTTP/1.1 200 -") was overrunning the terminal, one
# line every ~1s for the whole duration of a run, since that's exactly
# how often the frontend polls for progress. This only silences
# Werkzeug's OWN automatic HTTP access logging — the pipeline's real
# print statements (fetching, scoring, candidate generation, etc.)
# are untouched and still show normally.
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_this_dir = os.path.dirname(os.path.abspath(__file__))
three_route_module = _import_from_path("three_route_system", os.path.join(_this_dir, "08_three_route_system.py"))

app = Flask(__name__)

# NEW — background job tracking for live progress. Flask's normal
# request/response is synchronous: the old design just blocked the
# whole request until the pipeline finished, with no way to show
# progress while a multi-minute run was in flight. Real feedback
# asked for visibility, so the pipeline now runs in a background
# thread; the frontend polls /progress for status while it runs.
# Deliberately scoped for ONE job at a time (a single-user local
# debugging tool, not a multi-user service) — simpler than real job
# IDs, and matches what's actually needed here.
_job_lock = threading.Lock()
_job_state = {"status": "idle", "stage": "", "result": None, "error": None}

# Rough stage -> percentage mapping for the progress bar. Land cover/
# elevation refinement is the dominant, most VARIABLE-length stage
# (seconds to a couple of minutes depending on the area and what's
# cached) — this can't be a precise time-based percentage, just a
# reasonable sense of "which stage are we in," which is the actually
# useful part (knowing it's on land cover, not stuck somewhere odd).
PROGRESS_STAGE_PERCENT = [
    ("Fetching road network", 10),
    ("Building the routable graph", 20),
    ("Scoring every road", 35),
    ("Building feature-targeting candidates", 45),
    ("Generating candidate routes", 55),
    ("Refining elevation and land cover", 75),
    ("Searching for local route variations", 85),
    ("Finding highlights", 90),
    ("Computing experimental scores", 93),
    ("Fetching food/drink venues", 97),
    ("Done", 100),
]


def _stage_to_percent(stage_text):
    """Maps a progress_callback stage string to a rough percentage, by
    matching its leading words against PROGRESS_STAGE_PERCENT — falls
    back to the LAST matched stage's percentage if nothing matches
    (better than resetting to 0 on an unrecognised stage string)."""
    for prefix, pct in PROGRESS_STAGE_PERCENT:
        if stage_text.startswith(prefix):
            return pct
    return 5


def _run_pipeline_job(start_lat, start_lon, end_lat, end_lon):
    """Runs the real pipeline in a background thread, updating the
    shared _job_state as it progresses — see /find_routes (POST,
    starts this) and /progress (GET, polled by the frontend)."""
    def progress_callback(stage):
        with _job_lock:
            _job_state["stage"] = stage
            _job_state["percent"] = _stage_to_percent(stage)

    try:
        route_results, node_coords, start_node, end_node, _provisional_scores, run_bbox = (
            three_route_module.run_full_pipeline(
                start_lat, start_lon, end_lat, end_lon, verbose=True, progress_callback=progress_callback,
            )
        )

        routes_json = []
        for label, r in route_results.items():
            if label.startswith("_"):
                continue
            routes_json.append(_route_to_json(label, r, node_coords))

        candidates_json = []
        for c in route_results.get("_diagnostics", {}).get("all_candidates", []):
            path = c.get("path", [])
            points = [[node_coords[nid][0], node_coords[nid][1]] for nid in path if nid in node_coords]
            candidates_json.append({
                "source": c.get("source", "?"),
                "points": points,
                "avg_enjoyment_10": round(c.get("avg_enjoyment", 0.0) * 10, 1),
            })

        with _job_lock:
            _job_state["status"] = "done"
            _job_state["stage"] = "Done"
            _job_state["percent"] = 100
            _job_state["result"] = {"routes": routes_json, "candidates": candidates_json, "bbox": run_bbox}
    except Exception as e:
        # Deliberately broad — this is a debugging tool, and ANY
        # failure anywhere in a long real pipeline (Overpass timeout,
        # no road network found, an unexpected empty area, etc.)
        # should surface as a real, readable message in the UI rather
        # than a silently stuck progress bar with no clue what happened.
        with _job_lock:
            _job_state["status"] = "error"
            _job_state["error"] = f"{type(e).__name__}: {e}"


def _route_to_json(label, r, node_coords):
    """Converts one route's result dict into a JSON-friendly shape — same
    point-extraction logic as map_export.py's _add_route, just returned
    as data instead of written into a static HTML file."""
    path = r.get("path", [])
    points = [[node_coords[nid][0], node_coords[nid][1]] for nid in path if nid in node_coords]
    exp = r.get("experimental", {})
    return {
        "label": label,
        "points": points,
        "source": r.get("source", ""),
        "avg_enjoyment_10": round(r.get("avg_enjoyment", 0.0) * 10, 1),
        "best_stretch_10": round(r.get("best_stretch_enjoyment", 0.0) * 10, 1),
        "distance_km": round(r.get("real_distance_m", 0) / 1000, 2),
        "time_min": round(r.get("real_time_s", 0) / 60, 1),
        "distinct_from_compromise": r.get("distinct_from_compromise"),
        "highlights": three_route_module.format_highlights_text(r.get("highlights", [])),
        "landcover_composition_pct": {
            k: round(v * 100, 1) for k, v in r.get("landcover_composition", {}).items()
        },
        # NEW (Milestone 2) — EXPERIMENTAL, not yet validated against
        # real human perception — see compute_experimental_scores'
        # docstring. Labelled distinctly in the frontend (see
        # PAGE_HTML's renderResults), not presented with the same
        # confidence as the core, validated enjoyment score.
        "experimental": {
            "drama_10": round(exp.get("drama", 0.0) * 10, 1),
            "charm_10": round(exp.get("charm", 0.0) * 10, 1),
            "uniqueness_10": round(exp.get("uniqueness", 0.0) * 10, 1),
        },
    }


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/find_routes", methods=["POST"])
def find_routes():
    data = request.get_json(force=True)
    try:
        start_lat = float(data["start_lat"])
        start_lon = float(data["start_lon"])
        end_lat = float(data["end_lat"])
        end_lon = float(data["end_lon"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid start/end coordinates: {e}"}), 400

    with _job_lock:
        if _job_state["status"] == "running":
            return jsonify({"error": "A search is already running — wait for it to finish first."}), 409
        _job_state["status"] = "running"
        _job_state["stage"] = "Starting..."
        _job_state["percent"] = 0
        _job_state["result"] = None
        _job_state["error"] = None

    thread = threading.Thread(target=_run_pipeline_job, args=(start_lat, start_lon, end_lat, end_lon))
    thread.start()
    return jsonify({"started": True})


@app.route("/progress")
def progress():
    """Polled by the frontend every ~1s while a search is running —
    returns the current stage/percentage, or the final result once
    status is 'done' (or the error message if status is 'error')."""
    with _job_lock:
        return jsonify(dict(_job_state))


PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Byway — Route Tester</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { height: 100%; margin: 0; padding: 0; font-family: -apple-system, sans-serif; }
  #container { display: flex; height: 100%; }
  #map { flex: 1; }
  #sidebar { width: 340px; padding: 16px; overflow-y: auto; background: #f7f7f7; border-left: 1px solid #ddd; box-sizing: border-box; }
  h2 { margin-top: 0; font-size: 18px; }
  .instructions { font-size: 13px; color: #555; margin-bottom: 12px; line-height: 1.4; }
  button { width: 100%; padding: 10px; font-size: 14px; cursor: pointer; margin-bottom: 8px; border-radius: 4px; border: 1px solid #ccc; background: white; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  #findBtn { background: #2563eb; color: white; border: none; }
  #findBtn:disabled { background: #93a8d8; }
  .route-card { background: white; border-radius: 6px; padding: 10px 12px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.15); font-size: 13px; line-height: 1.5; }
  .route-card h3 { margin: 0 0 6px 0; font-size: 14px; }
  .status { font-size: 13px; color: #888; margin: 10px 0; line-height: 1.4; }
  .error { color: #c0392b; }
  .legend-color { display: inline-block; width: 14px; height: 4px; margin-right: 6px; vertical-align: middle; }
  .note { font-size: 11px; color: #999; margin-top: 2px; }
</style>
</head>
<body>
<div id="container">
  <div id="map"></div>
  <div id="sidebar">
    <h2>Byway route tester</h2>
    <div class="instructions">
      Click the map once for a <b>start</b> point, then again for an <b>end</b> point, then hit Find Routes.
      Each run does a real fetch + score + search — usually a few seconds to a couple of minutes
      depending on the area and what's already cached.
    </div>
    <div id="status" class="status">Click the map to set a start point.</div>
    <div id="progressWrap" style="display:none; margin-bottom:10px;">
      <div style="background:#e2e2e2; border-radius:4px; height:8px; overflow:hidden;">
        <div id="progressBar" style="background:#2563eb; height:100%; width:0%; transition:width 0.4s;"></div>
      </div>
    </div>
    <button id="findBtn" disabled>Find Routes</button>
    <button id="resetBtn">Reset</button>
    <div id="results"></div>
  </div>
</div>
<script>
  const map = L.map('map').setView([51.05, -0.65], 11);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
  }).addTo(map);

  let startMarker = null, endMarker = null;
  let routeLayers = [];

  const statusDiv = document.getElementById('status');
  const findBtn = document.getElementById('findBtn');
  const resultsDiv = document.getElementById('results');

  const colors = {'Direct (fastest)': '#888888', 'Compromise': '#2563eb', 'Max Enjoyment': '#16a34a'};

  function clearRoutes() {
    routeLayers.forEach(l => map.removeLayer(l));
    routeLayers = [];
    resultsDiv.innerHTML = '';
  }

  map.on('click', function(e) {
    if (!startMarker) {
      startMarker = L.marker(e.latlng, {title: 'Start'}).addTo(map);
      statusDiv.textContent = 'Start set (' + e.latlng.lat.toFixed(4) + ', ' + e.latlng.lng.toFixed(4) + '). Click again for an end point.';
    } else if (!endMarker) {
      endMarker = L.marker(e.latlng, {title: 'End'}).addTo(map);
      statusDiv.textContent = 'Start and end set. Click "Find Routes."';
      findBtn.disabled = false;
    }
    // Further clicks after both are set do nothing -- use Reset first.
  });

  document.getElementById('resetBtn').addEventListener('click', function() {
    if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
    if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
    clearRoutes();
    findBtn.disabled = true;
    statusDiv.textContent = 'Click the map to set a start point.';
  });

  const progressWrap = document.getElementById('progressWrap');
  const progressBar = document.getElementById('progressBar');
  let pollTimer = null;

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    progressWrap.style.display = 'none';
  }

  findBtn.addEventListener('click', function() {
    if (!startMarker || !endMarker) return;
    findBtn.disabled = true;
    clearRoutes();
    progressWrap.style.display = 'block';
    progressBar.style.width = '0%';
    statusDiv.textContent = 'Starting...';

    const payload = {
      start_lat: startMarker.getLatLng().lat, start_lon: startMarker.getLatLng().lng,
      end_lat: endMarker.getLatLng().lat, end_lon: endMarker.getLatLng().lng,
    };

    fetch('/find_routes', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
    })
    .then(r => r.json().then(data => ({status: r.status, data: data})))
    .then(({status, data}) => {
      if (status >= 400 || data.error) {
        findBtn.disabled = false;
        stopPolling();
        statusDiv.innerHTML = '<span class="error">Error: ' + (data.error || 'unknown error') + '</span>';
        return;
      }
      // Started successfully -- now poll /progress for live updates,
      // since this real pipeline can take anywhere from a few seconds
      // to a couple of minutes and the request itself returned
      // immediately rather than blocking for that whole time.
      pollTimer = setInterval(pollProgress, 1000);
    })
    .catch(err => {
      findBtn.disabled = false;
      stopPolling();
      statusDiv.innerHTML = '<span class="error">Request failed: ' + err + '</span>';
    });
  });

  function pollProgress() {
    fetch('/progress')
      .then(r => r.json())
      .then(state => {
        if (state.status === 'running') {
          statusDiv.textContent = state.stage || 'Working...';
          progressBar.style.width = (state.percent || 5) + '%';
        } else if (state.status === 'done') {
          stopPolling();
          findBtn.disabled = false;
          const data = state.result;
          statusDiv.textContent = data.routes.length + ' route(s) found, ' + data.candidates.length + ' candidate(s) considered.';
          renderResults(data);
        } else if (state.status === 'error') {
          stopPolling();
          findBtn.disabled = false;
          statusDiv.innerHTML = '<span class="error">Error: ' + state.error + '</span>';
        }
      })
      .catch(err => {
        stopPolling();
        findBtn.disabled = false;
        statusDiv.innerHTML = '<span class="error">Lost connection while polling for progress: ' + err + '</span>';
      });
  }

  function renderResults(data) {
    const allLatLngs = [];

    if (data.bbox) {
      const rect = L.rectangle(
        [[data.bbox.min_lat, data.bbox.min_lon], [data.bbox.max_lat, data.bbox.max_lon]],
        {color: '#f59e0b', weight: 2, fill: false, dashArray: '6,6'}
      ).addTo(map);
      rect.bindPopup('Fetch area (bbox) for this run');
      routeLayers.push(rect);
    }

    data.candidates.forEach(c => {
      if (c.points.length === 0) return;
      const line = L.polyline(c.points, {color: '#999999', weight: 2, opacity: 0.4, dashArray: '4,4'}).addTo(map);
      line.bindPopup('<i>candidate: ' + c.source + '</i><br>avg ' + c.avg_enjoyment_10 + '/10');
      routeLayers.push(line);
      allLatLngs.push(...c.points);
    });

    data.routes.forEach(r => {
      if (r.points.length === 0) return;
      const color = colors[r.label] || '#dc2626';
      const line = L.polyline(r.points, {color: color, weight: 5, opacity: 0.85}).addTo(map);
      line.bindPopup('<b>' + r.label + '</b><br>' + r.distance_km + ' km, ' + r.time_min + ' min<br>Avg: ' + r.avg_enjoyment_10 + '/10, Best stretch: ' + r.best_stretch_10 + '/10<br><i>' + r.source + '</i>');
      routeLayers.push(line);
      allLatLngs.push(...r.points);

      const composition = Object.entries(r.landcover_composition_pct || {})
        .filter(([k, v]) => v > 0)
        .map(([k, v]) => k.replace('_', ' ') + ' ' + v + '%')
        .join(', ');

      const highlightsList = (r.highlights || []);
      const highlightsHtml = highlightsList.length > 0
        ? '<ul style="margin:4px 0 0 0;padding-left:18px;">' + highlightsList.map(h => '<li>' + h + '</li>').join('') + '</ul>'
        : '<div class="note">No named features close enough to this specific route.</div>';

      const exp = r.experimental || {};
      const experimentalHtml =
        '<div style="margin-top:8px;padding:6px 8px;background:#fef3c7;border-radius:4px;border:1px solid #fde68a;">' +
        '<div style="font-size:11px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:0.03em;">EXPERIMENTAL — not yet validated</div>' +
        '<div style="font-size:12px;color:#78350f;margin-top:2px;">Drama ' + exp.drama_10 + '/10 &middot; Charm ' + exp.charm_10 + '/10 &middot; Uniqueness ' + exp.uniqueness_10 + '/10</div>' +
        '</div>';

      let note = '';
      if (r.distinct_from_compromise === false) {
        note = '<div class="note">No candidate beyond Compromise\\'s time was better -- repeating it honestly.</div>';
      }

      const card = document.createElement('div');
      card.className = 'route-card';
      card.innerHTML = '<h3><span class="legend-color" style="background:' + color + '"></span>' + r.label + '</h3>' +
        r.distance_km + ' km, ' + r.time_min + ' min<br>' +
        'Avg enjoyment: ' + r.avg_enjoyment_10 + '/10<br>' +
        'Best 30% stretch: ' + r.best_stretch_10 + '/10<br>' +
        (composition ? ('Land cover: ' + composition + '<br>') : '') +
        '<i>source: ' + r.source + '</i>' + note +
        '<div style="margin-top:6px;font-weight:600;">Highlights:</div>' + highlightsHtml +
        experimentalHtml;
      resultsDiv.appendChild(card);
    });

    if (allLatLngs.length > 0) {
      map.fitBounds(allLatLngs);
    }
  }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Starting Byway route tester...")
    print("Open http://localhost:5000 in a browser. Ctrl+C to stop.")
    app.run(debug=True, port=5000, threaded=True)
