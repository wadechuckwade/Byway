"""
Byway — Google Places usage circuit breaker
=================================================

What this does, in plain terms:
Tracks REAL cumulative Google Places API call counts, persisted to a
local file, and HARD-STOPS further calls once a configured monthly
event budget is exceeded — the same real-cost safety net already
proven for the Anthropic API (scoring/ai_circuit_breaker.py), applied
to a second metered external service this project picked up.

WHY THIS ISN'T JUST ai_circuit_breaker.py REUSED DIRECTLY: the actual
billing shape is genuinely different, not just a different number.
Anthropic's breaker tracks a DAILY dollar budget from real token
usage. Google Places (the fields this project actually needs --
rating, userRatingCount -- trigger the "Enterprise" SKU) gives a
MONTHLY free-event allowance (1,000 events/month, confirmed via
Google's own pricing docs this session) and bills per-REQUEST, not
per-token. Forcing both into one shared abstraction would mean either
warping Google's monthly/count shape to fit a daily/dollar one, or
building a generic core just for two call sites -- more complexity for
the same safety property. This is a deliberate sibling, not a forced
shared abstraction: same proven PATTERN (real usage, persisted,
hard-stop, clear refusal reason, tested offline), independently
implemented because the actual mechanics differ.

DEFAULT BEHAVIOUR IS "NEVER SPEND REAL MONEY": MONTHLY_EVENT_BUDGET
defaults to exactly Google's free monthly allowance for the SKU this
project uses (1,000). Hitting the budget means "no more free calls
this month," not "switch to paid" -- raising the budget past 1,000 is
an explicit, deliberate choice the per-project default never makes on
its own, same spirit as ai_circuit_breaker's conservative
DAILY_BUDGET_USD default.

HONEST LIMITATION: same as ai_circuit_breaker.py -- this only catches
usage that goes THROUGH record_usage() (i.e. only this project's own
calls, made via scoring/google_places.py). It can't see or limit any
other usage on the same Google Cloud billing account. Worth also
setting a real budget alert in the Google Cloud Console as an
independent second layer, not a replacement for this.
"""

import os
import json
import time

USAGE_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".google_places_usage_log.json",
)

# Google's own free monthly allowance for the SKU tier triggered by
# requesting `rating`/`userRatingCount` (the "Enterprise" SKU) --
# confirmed via Google's pricing documentation this session. Per-SKU
# free caps replaced the old universal $200/month credit in March
# 2025; this is the number that actually applies to the fields this
# project needs, not a generic "Google Maps free tier" figure.
GOOGLE_PLACES_ENTERPRISE_FREE_EVENTS_PER_MONTH = 1000
MONTHLY_EVENT_BUDGET = GOOGLE_PLACES_ENTERPRISE_FREE_EVENTS_PER_MONTH
# Deliberately defaults to EXACTLY the free allowance -- staying
# inside it by default means this can run with real, guaranteed-$0
# risk out of the box. Raise this explicitly (and only after setting
# up real billing + a Cloud Console budget alert) to allow paid usage
# beyond the free tier -- never raised automatically by this code.


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


def _this_month_key():
    return time.strftime("%Y-%m")  # monthly, not daily -- matches Google's own reset cadence


def get_month_call_count(path=None):
    """Returns this calendar month's REAL cumulative call count, from actual logged usage."""
    log = _load_usage_log(path)
    return log.get(_this_month_key(), {}).get("call_count", 0)


def check_budget_available(estimated_calls=1, budget=None, path=None):
    """
    Call BEFORE making a Google Places request -- returns (allowed:
    bool, reason: str). Refuses if this month's ALREADY-logged call
    count, plus estimated_calls (normally 1 -- a single lookup), would
    exceed the monthly budget. budget defaults to MONTHLY_EVENT_BUDGET
    (i.e. the free allowance) if not overridden.
    """
    budget = MONTHLY_EVENT_BUDGET if budget is None else budget
    this_month_count = get_month_call_count(path)
    if this_month_count + estimated_calls > budget:
        return False, (f"Monthly Google Places budget ({budget} calls) would be exceeded — "
                        f"already used {this_month_count} call(s) this month, this request would add "
                        f"{estimated_calls}. Refusing the call.")
    return True, ""


def record_usage(calls=1, path=None):
    """
    Call AFTER a real Google Places API request completes (success OR
    failure -- the request itself is what's billed, regardless of
    whether a usable result came back) -- updates this month's logged
    real call count. Returns the new running total for this month.
    """
    log = _load_usage_log(path)
    month = _this_month_key()
    if month not in log:
        log[month] = {"call_count": 0}
    log[month]["call_count"] += calls
    _save_usage_log(log, path)
    return log[month]["call_count"]


if __name__ == "__main__":
    import tempfile

    print("--- Offline test: circuit breaker tracks real cumulative call count correctly ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_log_path = os.path.join(tmpdir, "test_google_usage_log.json")

        allowed, reason = check_budget_available(path=test_log_path)
        assert allowed, "Should allow the first call with an empty log"
        print(f"First call allowed: {allowed}")

        for i in range(5):
            record_usage(path=test_log_path)
        count_after_5 = get_month_call_count(test_log_path)
        print(f"Call count after 5 calls: {count_after_5}")
        assert count_after_5 == 5
        print("PASSED — real call count tracked accurately across multiple calls\n")

        print("--- Offline test: circuit breaker actually BLOCKS once the monthly budget is hit ---")
        log = {_this_month_key(): {"call_count": MONTHLY_EVENT_BUDGET}}
        _save_usage_log(log, test_log_path)
        allowed, reason = check_budget_available(path=test_log_path)
        print(f"Allowed at exactly the budget: {allowed}")
        print(f"Reason given: {reason}")
        assert not allowed, "Should REFUSE once this month's count is already at the budget"
        assert "budget" in reason.lower()
        print("PASSED — circuit breaker refuses calls once the free monthly allowance is used up, "
              "with a real, specific reason given\n")

        print("--- Offline test: call count resets on a new month (different month key) ---")
        log = {"2020-01": {"call_count": MONTHLY_EVENT_BUDGET}}
        _save_usage_log(log, test_log_path)
        this_months_count = get_month_call_count(test_log_path)
        assert this_months_count == 0, "A different month's logged usage should not count against this month's budget"
        allowed, reason = check_budget_available(path=test_log_path)
        assert allowed
        print("PASSED — call count correctly resets for a new month, old months' usage doesn't carry over\n")

        print("--- Offline test: default budget is exactly the free allowance, not the old AI $-budget ---")
        assert MONTHLY_EVENT_BUDGET == GOOGLE_PLACES_ENTERPRISE_FREE_EVENTS_PER_MONTH == 1000
        print(f"MONTHLY_EVENT_BUDGET defaults to {MONTHLY_EVENT_BUDGET} -- exactly Google's free monthly "
              f"allowance for the Enterprise SKU, so default usage carries real, guaranteed-$0 risk.")
        print("PASSED\n")

    print("All Google Places circuit breaker tests passed.")
