"""
Byway — AI usage circuit breaker (Milestone 2)
====================================================

What this does, in plain terms:
Tracks REAL cumulative cost (from each API response's actual token
usage, not an estimate) across runs, persisted to a local file, and
HARD-STOPS further AI calls once a configured daily budget is
exceeded — the explicit safety mechanism the Milestone 2 plan
requires: "nothing in this milestone can generate an unexpected bill."

WHY TRACK REAL USAGE, NOT JUST CALL COUNT: a call-count cap (e.g. "max
50 calls/day") doesn't account for variable prompt/response length —
a cap based on ACTUAL reported token usage (Claude API responses
include real input_tokens/output_tokens in their "usage" field) is
the accurate, defensible way to bound real spend, not a proxy for it.

PRICING (verified live via web search this session, not pulled from
memory — see Decisions Log): Claude Haiku 4.5 at $1.00 input / $5.00
output per million tokens, confirmed via Anthropic's own published
pricing. Haiku 4.5 is deliberately the ONLY model this is built
around — a ranking/narration task over a short list of real venues
doesn't need Sonnet or Opus's added capability, and Haiku is
meaningfully cheaper (the whole reason it's the cost-control tier).
If a different model is ever used instead, these per-token constants
need updating to match — they are NOT looked up live each call.

HONEST LIMITATION: this only catches usage that goes THROUGH
record_usage() — i.e. only this project's own AI calls, made via
scoring/food_drink_ai.py. It cannot see or limit any other usage on
the same API key from elsewhere (a different app, a teammate, manual
testing in a separate script). It's a real, working safety net for
THIS pipeline specifically, not a substitute for Anthropic Console's
own organization-wide spend limits — those are worth setting too, as
a second, independent layer.
"""

import os
import json
import time

USAGE_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".ai_usage_log.json")
DAILY_BUDGET_USD = 0.50  # deliberately conservative — a real, hard ceiling, not a soft target

HAIKU_45_INPUT_COST_PER_MTOK = 1.00
HAIKU_45_OUTPUT_COST_PER_MTOK = 5.00


def _load_usage_log(path=None):
    path = path or USAGE_LOG_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_usage_log(log, path=None):
    path = path or USAGE_LOG_PATH
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def _today_key():
    return time.strftime("%Y-%m-%d")


def get_today_spend_usd(path=None):
    """Returns today's REAL cumulative spend (USD), from actual logged usage."""
    log = _load_usage_log(path)
    return log.get(_today_key(), {}).get("total_usd", 0.0)


def check_budget_available(estimated_max_cost_usd=0.02, path=None):
    """
    Call BEFORE making an AI request — returns (allowed: bool,
    reason: str). A real, hard check: refuses if today's ALREADY-
    LOGGED spend, plus a conservative estimate of this NEXT call's
    cost, would exceed DAILY_BUDGET_USD. The estimate is deliberately
    a pessimistic guess (a call's REAL cost is only known AFTER it
    returns, from its usage field) — on purpose, so a borderline call
    is refused upfront rather than discovered to have pushed spend
    over budget only after the fact.

    Default estimated_max_cost_usd=0.02 assumes a generously-sized
    prompt for this task (~3000 input tokens covering even a long
    venue list, ~500 output tokens for a few short blurbs) — real
    food/drink ranking calls should cost meaningfully less than this
    in practice; the estimate stays deliberately on the high side.
    """
    today_spend = get_today_spend_usd(path)
    if today_spend + estimated_max_cost_usd > DAILY_BUDGET_USD:
        return False, (f"Daily AI budget (${DAILY_BUDGET_USD:.2f}) would be exceeded — "
                        f"already spent ${today_spend:.4f} today, this call could add "
                        f"up to ~${estimated_max_cost_usd:.4f}. Refusing the call.")
    return True, ""


def record_usage(input_tokens, output_tokens, path=None):
    """
    Call AFTER a real API response returns, with its ACTUAL usage
    (response.usage.input_tokens / response.usage.output_tokens) —
    updates today's logged real spend based on what the call actually
    cost, not an estimate. Returns the real cost (USD) of this call.
    """
    cost_usd = (
        (input_tokens / 1_000_000) * HAIKU_45_INPUT_COST_PER_MTOK
        + (output_tokens / 1_000_000) * HAIKU_45_OUTPUT_COST_PER_MTOK
    )
    log = _load_usage_log(path)
    today = _today_key()
    if today not in log:
        log[today] = {"total_usd": 0.0, "call_count": 0}
    log[today]["total_usd"] += cost_usd
    log[today]["call_count"] += 1
    _save_usage_log(log, path)
    return cost_usd


if __name__ == "__main__":
    import tempfile

    print("--- Offline test: circuit breaker tracks real cumulative spend correctly ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_log_path = os.path.join(tmpdir, "test_usage_log.json")

        allowed, reason = check_budget_available(path=test_log_path)
        assert allowed, "Should allow the first call with an empty log"
        print(f"First call allowed: {allowed}")

        # Simulate 5 real calls, ~3000 input + 400 output tokens each.
        for i in range(5):
            cost = record_usage(3000, 400, path=test_log_path)
        spend_after_5 = get_today_spend_usd(test_log_path)
        print(f"Spend after 5 calls (~3000in/400out tokens each): ${spend_after_5:.4f}")
        expected_per_call = (3000 / 1_000_000) * 1.00 + (400 / 1_000_000) * 5.00
        assert abs(spend_after_5 - expected_per_call * 5) < 0.0001
        print("PASSED — real spend tracked accurately across multiple calls\n")

        print("--- Offline test: circuit breaker actually BLOCKS once budget is exceeded ---")
        # Force the log to look like today's already spent right up
        # to the edge of the budget.
        log = {_today_key(): {"total_usd": DAILY_BUDGET_USD - 0.005, "call_count": 100}}
        _save_usage_log(log, test_log_path)
        allowed, reason = check_budget_available(estimated_max_cost_usd=0.02, path=test_log_path)
        print(f"Allowed when nearly at budget: {allowed}")
        print(f"Reason given: {reason}")
        assert not allowed, "Should REFUSE once the next call's estimated cost would exceed the daily budget"
        assert "budget" in reason.lower()
        print("PASSED — circuit breaker actually refuses calls once budget is exceeded, "
              "with a real, specific reason given\n")

        print("--- Offline test: spend resets on a new day (different date key) ---")
        log = {"2020-01-01": {"total_usd": DAILY_BUDGET_USD, "call_count": 999}}
        _save_usage_log(log, test_log_path)
        todays_spend = get_today_spend_usd(test_log_path)
        assert todays_spend == 0.0, "A different day's logged spend should not count against today's budget"
        allowed, reason = check_budget_available(path=test_log_path)
        assert allowed
        print("PASSED — budget correctly resets for a new day, old days' spend doesn't carry over")

    print("\nAll circuit breaker tests passed.")
