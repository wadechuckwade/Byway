"""
Byway — AI-powered food & drink ranking/narration (Milestone 2)
======================================================================

What this does, in plain terms:
Given a list of REAL venues (from scoring.food_drink.
fetch_food_drink_venues_in_bbox) near a route, asks Claude (Haiku 4.5
— the cheapest appropriate model for this task; see ai_circuit_
breaker.py for the pricing this is built around) to pick the most
appealing few and write a short narrative blurb for each.

THE HARD GUARDRAIL AGAINST HALLUCINATED VENUES — ENFORCED IN CODE, NOT
JUST ASKED FOR IN THE PROMPT: the model is required to respond with
INDICES into the exact venue list it was given, never free-text
names. Every returned index is validated against the real list's
bounds before being trusted at all — an out-of-range, non-integer, or
malformed index means that ONE selection is dropped, and a completely
unparseable response means the WHOLE response is rejected. This makes
it STRUCTURALLY impossible for the model to surface a venue that
wasn't in the original OSM-sourced list — not just unlikely because
the prompt asked nicely. Every field in the final output except
"blurb" is copied DIRECTLY from the original venue dict, never from
the model's own text.

HONEST STATUS: this module was written carefully, with the validation
logic itself directly tested against mocked API responses (covering
valid selections, out-of-range indices, and unparseable JSON) — but
has NEVER been run against the real Anthropic API, since no API key
or network access is available in the environment that wrote it. Run
this directly, with a real ANTHROPIC_API_KEY configured, as the
actual first test before trusting it further — same discipline as
every other new external integration this project has built.

Network/dependency note: needs `pip install anthropic --break-system-
packages` and a real ANTHROPIC_API_KEY environment variable.
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.ai_circuit_breaker import check_budget_available, record_usage, get_today_spend_usd

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_VENUES_TO_RECOMMEND = 3
MAX_VENUES_TO_OFFER_MODEL = 25
# Cap on how many venues get listed in the prompt at all — a genuinely
# venue-dense area (a town centre) could have far more than needed for
# this task, and a longer prompt costs more (and the circuit breaker's
# cost ESTIMATE assumes a bounded prompt size — see check_budget_
# available's docstring). Top 25 by no particular ranking (the model's
# whole job is to rank among them) keeps this bounded and predictable.


def _build_prompt(venues, route_context=""):
    venue_list_text = "\n".join(
        f"{i}: {v['name']} ({v['amenity_type']}{', ' + v['cuisine'] if v.get('cuisine') else ''})"
        for i, v in enumerate(venues)
    )
    context_line = f" Route context: {route_context}" if route_context else ""
    return f"""You are recommending food and drink stops along a scenic driving route.{context_line}

Here is the COMPLETE list of real venues near this route, each with an index number:
{venue_list_text}

Pick the {MAX_VENUES_TO_RECOMMEND} most appealing venues from this EXACT list for a scenic drive stop. You MUST reference them only by their index number above — do not invent, rename, or describe any venue that is not in this list.

Respond with ONLY a JSON object in this exact shape, no other text, no markdown formatting:
{{"selections": [{{"index": <int>, "blurb": "<one short, appealing sentence about this stop>"}}, ...]}}"""


def _parse_and_validate_response(raw_text, venues, verbose=True):
    """
    THE ACTUAL HALLUCINATION GUARDRAIL — see module docstring. Parses
    the model's raw text as JSON and validates every returned index
    against the real venues list's bounds. Returns a list of
    {"name", "lat", "lon", "amenity_type", "blurb"} dicts — every
    field except "blurb" copied DIRECTLY from the matched original
    venue, never from the model's own text.

    A completely unparseable response (not valid JSON, missing the
    "selections" key) rejects the WHOLE response — fails safe, never
    partially trusts a response that didn't even follow the basic
    format asked for. An individual out-of-range/malformed index
    within an otherwise-valid response just drops THAT selection,
    logged clearly, not silently.
    """
    try:
        parsed = json.loads(raw_text)
        selections = parsed["selections"]
        if not isinstance(selections, list):
            raise TypeError("'selections' must be a list")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        if verbose:
            print(f"  AI response could not be parsed as the expected JSON shape ({e}) — "
                  f"rejecting the ENTIRE response (fail safe, not a partial salvage).")
        return []

    results = []
    for sel in selections:
        if not isinstance(sel, dict):
            continue
        idx = sel.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(venues):
            if verbose:
                print(f"  AI returned an invalid index ({idx!r}) — dropping this ONE selection "
                      f"(a real guardrail enforced in code, not a soft warning).")
            continue
        venue = venues[idx]
        results.append({
            "name": venue["name"],
            "lat": venue["lat"],
            "lon": venue["lon"],
            "amenity_type": venue["amenity_type"],
            "blurb": sel.get("blurb", "") if isinstance(sel.get("blurb"), str) else "",
        })

    return results


def get_food_drink_recommendations(venues, route_context="", verbose=True):
    """
    Ranks and narrates the most appealing food/drink stops from a REAL
    venue list, using Claude Haiku 4.5 — constrained so it can never
    surface anything not already in `venues` (see module docstring
    and _parse_and_validate_response for the actual enforcement).

    venues: from scoring.food_drink.fetch_food_drink_venues_in_bbox —
    the real, OSM-sourced ground truth this is constrained to.
    route_context: optional short text describing the route (e.g.
    "a scenic drive through the South Downs") — included in the
    prompt for better-fitting blurbs, has NO effect on the guardrail.

    Returns a list of recommendation dicts (see
    _parse_and_validate_response) — empty if there are no venues to
    choose from, the circuit breaker refuses the call (budget),
    the API call itself fails, or the response can't be validated at
    all. Fails SAFE in every case — never partially trusts something
    suspect just to return SOMETHING.
    """
    if not venues:
        if verbose:
            print("  No venues to recommend from — skipping the AI call entirely.")
        return []

    venues_for_prompt = venues[:MAX_VENUES_TO_OFFER_MODEL]

    allowed, reason = check_budget_available()
    if not allowed:
        if verbose:
            print(f"  AI circuit breaker: {reason}")
        return []

    try:
        import anthropic
    except ImportError:
        if verbose:
            print("  anthropic package not installed — pip install anthropic --break-system-packages")
        return []

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    prompt = _build_prompt(venues_for_prompt, route_context)

    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        if verbose:
            print(f"  AI request failed ({type(e).__name__}: {e}) — returning no recommendations.")
        return []

    cost_usd = record_usage(response.usage.input_tokens, response.usage.output_tokens)
    if verbose:
        print(f"  AI call cost: ${cost_usd:.4f} (today's running total: ${get_today_spend_usd():.4f})")

    raw_text = "".join(block.text for block in response.content if hasattr(block, "text"))
    return _parse_and_validate_response(raw_text, venues_for_prompt, verbose=verbose)


if __name__ == "__main__":
    class _FakeUsage:
        def __init__(self, input_tokens, output_tokens):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class _FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class _FakeResponse:
        def __init__(self, text, input_tokens=2000, output_tokens=200):
            self.content = [_FakeTextBlock(text)]
            self.usage = _FakeUsage(input_tokens, output_tokens)

    fake_venues = [
        {"name": "The Crown", "lat": 51.0, "lon": -0.7, "amenity_type": "pub", "cuisine": "british"},
        {"name": "Bella Italia", "lat": 51.001, "lon": -0.701, "amenity_type": "restaurant", "cuisine": "italian"},
        {"name": "Riverside Cafe", "lat": 51.002, "lon": -0.702, "amenity_type": "cafe", "cuisine": None},
    ]

    print("--- Offline test: valid response correctly maps indices back to REAL venues ---")
    valid_json = json.dumps({"selections": [
        {"index": 0, "blurb": "A classic countryside pub, great for a halfway stop."},
        {"index": 2, "blurb": "A quiet riverside spot for coffee."},
    ]})
    results = _parse_and_validate_response(valid_json, fake_venues)
    print(f"Got {len(results)} validated recommendation(s):")
    for r in results:
        print(f"  {r['name']} — {r['blurb']}")
    assert len(results) == 2
    assert results[0]["name"] == "The Crown"
    assert results[0]["lat"] == 51.0, "Coordinates must come from the REAL venue, never the model's text"
    assert results[1]["name"] == "Riverside Cafe"
    print("PASSED — valid indices correctly map back to the real, original venue data\n")

    print("--- Offline test: THE ACTUAL GUARDRAIL — out-of-range index is dropped, not trusted ---")
    hallucinated_json = json.dumps({"selections": [
        {"index": 0, "blurb": "Real venue, fine."},
        {"index": 99, "blurb": "A charming little place called The Hallucinated Tavern!"},
        {"index": -1, "blurb": "Negative index, also invalid."},
    ]})
    results2 = _parse_and_validate_response(hallucinated_json, fake_venues)
    print(f"Got {len(results2)} validated recommendation(s) (expected 1 — only the real, in-range one)")
    assert len(results2) == 1
    assert results2[0]["name"] == "The Crown"
    assert "Hallucinated" not in str(results2), "An out-of-range index must NEVER produce output, structurally"
    print("PASSED — out-of-range indices are dropped, never trusted, regardless of how plausible the blurb sounds\n")

    print("--- Offline test: completely unparseable response rejects EVERYTHING (fail safe) ---")
    garbage = "I think you should visit The Crown, it's lovely! Also maybe Bella Italia."
    results3 = _parse_and_validate_response(garbage, fake_venues)
    assert results3 == [], "A response that isn't even the right JSON shape must be rejected entirely"
    print("PASSED — free-text response (no structured indices at all) is rejected wholesale, not parsed by guessing\n")

    print("--- Offline test: empty venue list skips the AI call entirely (no wasted spend) ---")
    empty_results = get_food_drink_recommendations([], verbose=True)
    assert empty_results == []
    print("PASSED\n")

    print("All offline guardrail tests passed — this is genuinely tested logic, not just written-and-hoped.\n")

    print("--- Live test: attempting a REAL API call, only if a real key is actually configured ---")
    real_key = os.environ.get("ANTHROPIC_API_KEY", "")
    looks_like_a_real_key = real_key.startswith("sk-ant-")
    if not looks_like_a_real_key:
        print(f"No real-looking ANTHROPIC_API_KEY found in the environment "
              f"({'empty' if not real_key else 'set, but doesn' + chr(39) + 't look like a real Anthropic key'}).")
        print("Set a genuine key (export ANTHROPIC_API_KEY=sk-ant-...) and re-run this exact script")
        print("to actually exercise the live path for the first time — this is the real first test,")
        print("not the offline tests above, which never touch the network at all.")
    else:
        print("Real-looking API key found — attempting one small, real call now (this WILL incur a tiny real cost)...")
        live_results = get_food_drink_recommendations(
            fake_venues, route_context="a short test drive through the countryside", verbose=True,
        )
        print(f"\nLive call returned {len(live_results)} validated recommendation(s):")
        for r in live_results:
            print(f"  {r['name']} — {r['blurb']}")
        if live_results:
            print("\nCONFIRMED — for the first time — that the live API call, the index-based guardrail, "
                  "and the real circuit breaker all work correctly together against the actual Anthropic API.")
        else:
            print("\nGot zero recommendations back from a real call — check the verbose output above for why "
                  "(could be a budget refusal, a real API error, or a response that didn't parse as expected). "
                  "This is real, informative evidence either way, not a guess.")
