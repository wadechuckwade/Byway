"""
Byway — Shared ArcGIS REST query pagination utility
========================================================

What this does, in plain terms:
Paginates through an ArcGIS FeatureServer query, accumulating ALL
matching features rather than silently stopping at the server's own
per-request limit.

WHY THIS EXISTS — A GENERALIZED FIX, NOT A NARROW ONE: found via
direct feedback after a related performance fix (OS land cover's
per-point vs whole-bbox query tuning) was originally applied to just
one file. The SAME underlying risk — an arbitrary per-request cap
silently excluding the most relevant results, with no guarantee the
server's own return order favours significance — existed in BOTH
historic_england.py's NHLE query (capped at 500 records, no ordering)
and villages.py's Conservation Areas query (capped at 200 records). A
dense historic city (York, Bath, Oxford) could plausibly have more
than 500 listed buildings; a large National Park could have more than
200 Conservation Areas. Whichever subset the server happens to return
first isn't guaranteed to be the most significant — exactly the same
"don't trust an unordered cap" lesson the building-density fix already
taught, just not yet generalized beyond that one file.

Rather than patch each file's cap independently (the exact class of
narrow fix this was written in response to), this is a SHARED utility
both now use — one correct implementation, not two independently-
maintained copies of the same pagination logic that could drift apart
or get fixed in one place and not the other.

STANDING PRACTICE NOTE FOR FUTURE WORK ON THIS PROJECT: when a fix
addresses a genuine class of problem (not just one specific symptom),
check whether the SAME class of problem exists elsewhere in the
codebase and fix it there too, in the same pass — rather than waiting
to be asked again for each separate instance. This file exists
specifically because that didn't happen the first time.
"""

import time
import requests


def paginated_arcgis_query(url, params, page_size=500, max_pages=20, delay_seconds=0.3,
                            user_agent="BywayApp-DevelopmentPrototype/0.1", timeout=20, debug=False):
    """
    Repeatedly queries an ArcGIS REST FeatureServer query endpoint,
    accumulating ALL features across pages rather than stopping at
    the first page's worth — which could silently exclude relevant
    results in a dense area, with no guarantee the server's own
    return order favours significance.

    params: the base query params dict (where, outFields, geometry,
    geometryType, spatialRel, outSR, f, etc.) — resultRecordCount and
    resultOffset are added/overridden here per page, so don't set
    those in the params you pass in.

    Stops when a page returns fewer features than page_size, AND the
    server's own "exceededTransferLimit" flag is absent/false, OR
    when a page returns zero features, OR after max_pages (a hard
    safety cap against a pathologically unbounded query — flagged
    clearly if hit, never silently truncated without comment).

    Returns the combined list of raw "features" (same shape as a
    single ArcGIS query response's "features" list) across all pages.
    """
    all_features = []
    offset = 0

    for page in range(max_pages):
        page_params = dict(params)
        page_params["resultRecordCount"] = page_size
        page_params["resultOffset"] = offset

        time.sleep(delay_seconds)
        try:
            response = requests.get(url, params=page_params, headers={"User-Agent": user_agent}, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"    (Paginated ArcGIS query failed on page {page} ({e}) — "
                  f"returning {len(all_features)} feature(s) gathered so far)")
            return all_features

        if "error" in data:
            print(f"    (Paginated ArcGIS query returned an error on page {page}: {data['error']} — "
                  f"returning {len(all_features)} feature(s) gathered so far)")
            return all_features

        page_features = data.get("features", [])
        all_features.extend(page_features)

        if debug:
            print(f"    [DEBUG] Page {page}: {len(page_features)} feature(s), "
                  f"exceededTransferLimit={data.get('exceededTransferLimit', False)}")

        exceeded = data.get("exceededTransferLimit", False)
        if len(page_features) == 0 or (not exceeded and len(page_features) < page_size):
            break

        offset += page_size
    else:
        print(f"    (Paginated ArcGIS query hit max_pages={max_pages} for this area — there may be MORE "
              f"features beyond what was gathered. A real, flagged limit, not a silent truncation — "
              f"raise max_pages if this area genuinely needs it.)")

    return all_features


if __name__ == "__main__":
    print("--- Offline test: pagination stops correctly across multiple pages (mocked, no network) ---")
    import unittest.mock as mock

    # Simulate a server with 3 pages of 2 features each (page_size=2),
    # then a final short page signalling the end.
    fake_pages = [
        {"features": [{"id": 1}, {"id": 2}], "exceededTransferLimit": True},
        {"features": [{"id": 3}, {"id": 4}], "exceededTransferLimit": True},
        {"features": [{"id": 5}], "exceededTransferLimit": False},
    ]
    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        response = mock.Mock()
        response.raise_for_status = lambda: None
        response.json = lambda: fake_pages[idx] if idx < len(fake_pages) else {"features": []}
        return response

    with mock.patch("requests.get", side_effect=fake_get):
        results = paginated_arcgis_query(
            "https://fake.example.com/query", {"where": "1=1", "f": "json"},
            page_size=2, delay_seconds=0, debug=True,
        )

    print(f"Total features gathered: {len(results)}")
    assert len(results) == 5, f"Expected all 5 features across 3 pages, got {len(results)}"
    assert call_count["n"] == 3, f"Expected exactly 3 page requests (stopping at the short final page), got {call_count['n']}"
    print("PASSED — pagination correctly gathers ALL features across multiple pages and stops "
          "at the right point, rather than silently capping at one page's worth\n")

    print("--- Offline test: max_pages safety cap is honoured and flagged, not silent ---")
    call_count["n"] = 0
    infinite_pages = [{"features": [{"id": i}, {"id": i + 1}], "exceededTransferLimit": True} for i in range(100)]

    def fake_get_infinite(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        response = mock.Mock()
        response.raise_for_status = lambda: None
        response.json = lambda: infinite_pages[idx]
        return response

    with mock.patch("requests.get", side_effect=fake_get_infinite):
        results = paginated_arcgis_query(
            "https://fake.example.com/query", {"where": "1=1", "f": "json"},
            page_size=2, max_pages=5, delay_seconds=0,
        )
    assert call_count["n"] == 5, f"Expected exactly max_pages=5 requests, got {call_count['n']}"
    print(f"Stopped at max_pages=5 as expected, with {len(results)} features gathered (flagged, not silent)")
    print("PASSED — the safety cap stops a pathological query rather than looping forever, "
          "and prints a clear flag rather than silently truncating without comment\n")

    print("All arcgis_utils tests passed.")
