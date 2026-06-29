"""
Byway — Google Places rating enrichment (Milestone 2, opt-in, capped)
============================================================================

What this does, in plain terms:
For a SMALL set of food/drink venues already chosen as candidates to
actually surface to the user (NOT the whole OSM corridor fetch), looks
up a real Google rating + review count via the Places API (New) Text
Search endpoint, and attaches it to the venue. This is the
comprehensive "is this place actually any good" baseline the free
sources (Michelin, CAMRA) can't provide at UK-wide density — see the
Decisions Log entry on Food/Drink Quality Research for why the free
options were each either too narrow (Michelin), paywalled (Good Food
Guide, Good Beer Guide), or a proxy mismatch (CAMRA Historic Pub
Interiors).

WHY ONLY A SMALL CANDIDATE SET, NEVER THE WHOLE CORRIDOR: the fields
this needs (`rating`, `userRatingCount`) trigger the Enterprise SKU,
which gets only 1,000 free events/month (see scoring/google_places_
circuit_breaker.py) — calling this on every OSM venue in a wide
corridor fetch would burn through that fast. This mirrors the
project's own established two-phase pattern (cheap whole-graph
provisional scoring, expensive elevation/land-cover refinement only
for ways actually used) — cheap OSM fetch for everything, this
expensive lookup only for the handful of venues about to be shown.

WHY TEXT SEARCH, NOT NEARBY SEARCH: Nearby Search returns multiple
plausible venues near a point with no guarantee of matching a SPECIFIC
OSM venue. Text Search (New), queried with the venue's own name plus a
locality hint and biased toward its known coordinates, is the closer
analogue of the old "Find Place from Text" pattern -- asking "does
THIS specific place have a Google entry," not "what's generally
nearby."

THE GEOGRAPHIC CONFIDENCE CHECK -- WHY IT EXISTS: a text-name search
can return a same-named or similarly-named venue that ISN'T the one we
meant (a chain pub with branches in several towns, a common pub name
like "The Crown" appearing many times). Rather than trust whatever
Google's top result is, the returned location is checked against the
OSM venue's own coordinates -- a result more than MAX_MATCH_DISTANCE_M
away is treated as "no confident match" and discarded, same
conservative-bar philosophy already used for the Michelin proximity
matcher (scoring/michelin.py) and the AI guardrail's index validation
(scoring/food_drink_ai.py) -- a missed real match costs nothing
(simply no rating shown); a wrong match would misrepresent a real
venue, which matters more to avoid.

DEFAULTS TO OFF, EXACTLY LIKE enable_food_drink_ai: this is the one
part of the food/drink pipeline with a real per-call cost (even though
that cost defaults to $0 thanks to the circuit breaker's free-tier
cap) -- enrich_venues_with_google_ratings() does nothing at all unless
explicitly enabled, no API key lookup, no budget check, no network
call, mirroring food_drink_ai.py's "no food_drink key even appears on
results" discipline when disabled.

Network/dependency note: needs a real GOOGLE_PLACES_API_KEY
environment variable and real internet access. Like food_drink_ai.py,
this has NEVER been run against the real Google API from the
environment that wrote it -- the offline tests below cover the
matching/validation/circuit-breaker logic against mocked responses;
the live call itself is the one piece needing a real first run with a
real key, same discipline as every other new external integration
this project has built.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.curvature import haversine_distance_m
from scoring.google_places_circuit_breaker import check_budget_available, record_usage, get_month_call_count

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
FIELD_MASK = "places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount"
# Requesting `rating`/`userRatingCount` is what triggers the Enterprise
# SKU (see google_places_circuit_breaker.py) -- `location` is included
# too, specifically so the geographic confidence check below has
# something real to check against, not an extra cost tier of its own
# (location is a Basic-tier field; it doesn't change which SKU this
# request is billed at, since rating/userRatingCount already determine
# that).

MAX_MATCH_DISTANCE_M = 400
# How far Google's returned location can be from the OSM venue's own
# coordinates and still be trusted as "the same place." Wider than
# the Michelin matcher's 200m -- Google's own geocoding for a name-only
# text query (no coordinates in the query itself, only used as a bias)
# has more room to drift than two independently-geocoded structured
# datasets being compared directly.

DEFAULT_MAX_VENUES = 10
# A second, defensive cap independent of whatever the caller passes --
# even if a caller forgets to pre-narrow their list, this never fires
# more than DEFAULT_MAX_VENUES real requests in one call, keeping a
# single mistake from meaningfully eating into the monthly budget.


def _text_search_top_match(name, lat, lon, locality_hint=None, api_key=None, timeout=15):
    """
    Makes ONE real Places API (New) Text Search request for the given
    venue name (optionally with a locality hint, e.g. a village/town
    name, to help disambiguate common pub names), biased toward the
    venue's known coordinates. Returns the raw top result dict (a
    "Place" object, per Google's schema) or None if there's no result
    or the request fails for any reason -- never raises into the
    caller; a failed lookup just means no rating attached, the same
    fail-safe philosophy as every other external call in this
    pipeline (e.g. food_drink_ai.py's broad try/except around the
    Anthropic call).
    """
    import requests  # imported here, not at module level, so importing this
                      # module never requires `requests` to be installed if
                      # the feature is simply left disabled (enable=False)

    query_text = f"{name}, {locality_hint}" if locality_hint else name
    body = {
        "textQuery": query_text,
        "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": 200.0}},
        "maxResultCount": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }
    try:
        response = requests.post(TEXT_SEARCH_URL, json=body, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"    (Google Places request failed for '{name}' ({type(e).__name__}: {e}) — "
              f"no rating attached, pipeline continues.)")
        return None

    places = data.get("places", [])
    return places[0] if places else None


def get_google_rating_for_venue(venue, locality_hint=None, api_key=None, verbose=True):
    """
    Looks up ONE venue's real Google rating, subject to the circuit
    breaker. venue: a dict with at least "name", "lat", "lon" (the
    shape returned by scoring.food_drink.fetch_food_drink_venues_in_
    bbox). api_key: defaults to the GOOGLE_PLACES_API_KEY environment
    variable if not given explicitly.

    Returns {"google_rating": float, "google_rating_count": int,
    "match_confidence_m": float, "matched_name": str} on a confident
    match, or None if: the circuit breaker refuses the call (budget),
    no API key is configured, the request fails, no result came back,
    or the result's location is too far from the venue's own
    coordinates to trust (see MAX_MATCH_DISTANCE_M / module
    docstring). Fails safe in every case -- never partially trusts an
    unconfirmed match.

    NOTE: usage is recorded for every REAL API call attempted,
    regardless of whether a confident match results -- the request
    itself is what's billed, not the outcome.
    """
    api_key = api_key or os.environ.get("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        if verbose:
            print("    GOOGLE_PLACES_API_KEY not set — skipping Google rating lookup for this venue.")
        return None

    allowed, reason = check_budget_available()
    if not allowed:
        if verbose:
            print(f"    Google Places circuit breaker: {reason}")
        return None

    place = _text_search_top_match(venue["name"], venue["lat"], venue["lon"], locality_hint, api_key)
    record_usage()  # the request was made (or attempted) -- record it regardless of the outcome below

    if not place:
        if verbose:
            print(f"    No Google Places result for '{venue['name']}'.")
        return None

    location = place.get("location", {})
    result_lat, result_lon = location.get("latitude"), location.get("longitude")
    if result_lat is None or result_lon is None:
        if verbose:
            print(f"    Google result for '{venue['name']}' had no location to confirm against — discarding.")
        return None

    distance_m = haversine_distance_m((venue["lon"], venue["lat"]), (result_lon, result_lat))
    if distance_m > MAX_MATCH_DISTANCE_M:
        if verbose:
            print(f"    Google's top result for '{venue['name']}' is {round(distance_m)}m away — "
                  f"beyond {MAX_MATCH_DISTANCE_M}m, treating as no confident match, not trusting it.")
        return None

    rating = place.get("rating")
    rating_count = place.get("userRatingCount")
    if rating is None:
        if verbose:
            print(f"    Google has a confident location match for '{venue['name']}' but no rating on file.")
        return None

    if verbose:
        print(f"    '{venue['name']}' matched Google rating {rating}/5 ({rating_count} reviews), "
              f"{round(distance_m)}m from the OSM coordinates.")

    return {
        "google_rating": rating,
        "google_rating_count": rating_count,
        "match_confidence_m": round(distance_m, 1),
        "matched_name": place.get("displayName", {}).get("text"),
    }


def enrich_venues_with_google_ratings(venues, locality_hint=None, enable=False, max_venues=DEFAULT_MAX_VENUES, verbose=True):
    """
    Attaches a "google" key to each venue in a SMALL, already-narrowed
    candidate list, IF enable=True. With the default enable=False,
    this is a complete no-op: no API key lookup, no budget check, no
    network call, venues returned exactly as given -- mirroring food_
    drink_ai.py's "disabled means genuinely nothing happens" pattern.

    venues: list of dicts with at least "name", "lat", "lon" -- the
    already-narrowed set actually about to be shown to the user, NOT
    the whole OSM corridor fetch (see module docstring for why this
    distinction matters for the monthly call budget).
    locality_hint: optional short place name (e.g. the nearer of the
    route's two endpoints, or a village the venue is in) to help
    disambiguate common venue names in the text query.
    max_venues: a defensive second cap (see module-level constant) --
    only the first max_venues entries of `venues` are ever looked up,
    even if a longer list is passed in by mistake.

    Mutates and returns the SAME venues list. Wrapped per-venue in a
    broad try/except -- one venue's lookup failing for any reason
    never breaks the rest of the batch, or the caller's pipeline.
    """
    if not enable:
        return venues

    for venue in venues[:max_venues]:
        try:
            result = get_google_rating_for_venue(venue, locality_hint=locality_hint, verbose=verbose)
            if result is not None:
                venue["google"] = result
        except Exception as e:
            if verbose:
                print(f"    (Google Places lookup raised an unexpected error for "
                      f"'{venue.get('name', '?')}' ({type(e).__name__}: {e}) — skipping this venue, "
                      f"pipeline continues.)")

    return venues


if __name__ == "__main__":
    import unittest.mock as mock
    import contextlib

    @contextlib.contextmanager
    def _env_unset(key):
        had_key = key in os.environ
        old_value = os.environ.pop(key, None)
        try:
            yield
        finally:
            if had_key:
                os.environ[key] = old_value

    fake_venue_nearby = {"name": "The Crown", "lat": 51.090, "lon": -0.700, "amenity_type": "pub"}

    def _fake_place(lat, lon, rating=4.3, rating_count=210, display_name="The Crown"):
        return {
            "displayName": {"text": display_name},
            "location": {"latitude": lat, "longitude": lon},
            "rating": rating,
            "userRatingCount": rating_count,
        }

    print("--- Offline test: enable=False is a COMPLETE no-op (the actual cost guarantee) ---")
    with _env_unset("GOOGLE_PLACES_API_KEY"):
        with mock.patch("requests.post") as mocked_post:
            result_venues = enrich_venues_with_google_ratings([dict(fake_venue_nearby)], enable=False)
            assert mocked_post.call_count == 0, "enable=False must make ZERO network calls, no exceptions"
    assert "google" not in result_venues[0]
    print("PASSED — disabled means genuinely zero network calls, exactly like food_drink_ai's disabled state\n")

    print("--- Offline test: no API key configured -> graceful skip, no crash ---")
    with _env_unset("GOOGLE_PLACES_API_KEY"):
        out = get_google_rating_for_venue(dict(fake_venue_nearby))
    assert out is None
    print("PASSED — missing API key fails safe (returns None), doesn't raise\n")

    print("--- Offline test: a confident geographic match is accepted and parsed correctly ---")
    os.environ["GOOGLE_PLACES_API_KEY"] = "fake-test-key-not-real"
    with mock.patch("requests.post") as mocked_post:
        mocked_post.return_value.raise_for_status = lambda: None
        mocked_post.return_value.json = lambda: {"places": [_fake_place(51.0901, -0.7001)]}
        result = get_google_rating_for_venue(dict(fake_venue_nearby), verbose=True)
    assert result is not None
    assert result["google_rating"] == 4.3
    assert result["google_rating_count"] == 210
    assert result["match_confidence_m"] < 50
    print(f"Matched: {result}")
    print("PASSED — a geographically confident match is correctly parsed and trusted\n")

    print("--- Offline test: THE ACTUAL GUARDRAIL — a same-named but far-away result is REJECTED ---")
    with mock.patch("requests.post") as mocked_post:
        mocked_post.return_value.raise_for_status = lambda: None
        # Google's top result claims to be "The Crown" but its real
        # location is nowhere near the OSM venue we asked about --
        # e.g. a different pub of the same common name in another town.
        mocked_post.return_value.json = lambda: {"places": [_fake_place(53.000, -2.500)]}
        result_far = get_google_rating_for_venue(dict(fake_venue_nearby), verbose=True)
    assert result_far is None, "A same-named result far from the venue's real coordinates must be REJECTED, not trusted"
    print("PASSED — geographic confidence check rejects a same-named-but-wrong-place match, "
          "exactly the failure mode this check exists to catch\n")

    print("--- Offline test: circuit breaker integration -- budget exhaustion blocks the call before any request ---")
    from scoring.google_places_circuit_breaker import _save_usage_log, _this_month_key, MONTHLY_EVENT_BUDGET
    _save_usage_log({_this_month_key(): {"call_count": MONTHLY_EVENT_BUDGET}})
    with mock.patch("requests.post") as mocked_post:
        result_blocked = get_google_rating_for_venue(dict(fake_venue_nearby), verbose=True)
        assert mocked_post.call_count == 0, "A budget-exhausted month must block BEFORE making any real request"
    assert result_blocked is None
    _save_usage_log({})  # reset the real (test-process) log so this doesn't pollute anything else
    print("PASSED — circuit breaker blocks the call entirely once the monthly free allowance is used up, "
          "with zero network calls made, not just a refusal after the fact\n")

    print("--- Offline test: max_venues caps how many real lookups happen, even on a longer list ---")
    many_venues = [dict(fake_venue_nearby, name=f"Venue {i}") for i in range(20)]
    with mock.patch("requests.post") as mocked_post:
        mocked_post.return_value.raise_for_status = lambda: None
        mocked_post.return_value.json = lambda: {"places": []}
        enrich_venues_with_google_ratings(many_venues, enable=True, max_venues=3, verbose=False)
    assert mocked_post.call_count == 3, f"Expected exactly 3 real requests (max_venues cap), got {mocked_post.call_count}"
    print(f"Real requests made for a 20-venue list with max_venues=3: {mocked_post.call_count}")
    print("PASSED — the defensive cap is honoured regardless of how many venues are passed in\n")

    del os.environ["GOOGLE_PLACES_API_KEY"]
    print("All Google Places module offline tests passed.")
