"""
Byway — Food & Drink venue fetching (Milestone 2)
======================================================

What this does, in plain terms:
Fetches REAL candidate food/drink venues (pubs, restaurants, cafes,
bars) from OpenStreetMap for a route's bbox — the raw, ground-truth
list that the AI ranking/narration step (scoring/food_drink_ai.py) is
constrained to choose from. That module enforces — in code, not just
by asking nicely in a prompt — that it can NEVER surface a venue that
isn't in this exact list.

WHY OSM, NOT A NEW DATA SOURCE: this project's whole pattern has been
"prefer free, real, already-integrated data sources over guessing or
inventing." OSM already has a rich `amenity=pub/restaurant/cafe/bar`
tagging scheme, the SAME data source the rest of this pipeline (roads,
historic sites' OSM tags) already depends on. Reuses scoring.
proximity's existing, already rate-limit/retry-handled
`_query_overpass` helper directly, rather than building a second,
independent Overpass client with its own retry logic to maintain.

Only venues with a real OSM `name` tag are kept — an AI recommendation
needs a genuine name to reference ("The Three Horseshoes," not
"unnamed pub at this location") — same filtering pattern already used
for historic sites.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring.proximity import _query_overpass


FOOD_DRINK_AMENITY_TAGS = ["pub", "restaurant", "cafe", "bar"]


def fetch_food_drink_venues_in_bbox(bbox):
    """
    Query Overpass for named pub/restaurant/cafe/bar venues within the
    given route bounding box (same {min_lat, max_lat, min_lon,
    max_lon} shape used elsewhere in this codebase).

    Returns a list of {"name", "lat", "lon", "amenity_type", "cuisine"}
    dicts — "cuisine" is whatever OSM's own `cuisine` tag says (e.g.
    "fish_and_chips", "italian"), or None if untagged; not required,
    just useful context for the AI ranking step's narrative if present.

    Returns an empty list (not an error) if nothing matches in this
    area — a real, expected outcome in genuinely rural stretches, not
    a failure case.
    """
    b = bbox
    clauses = "\n".join(
        f'  node["amenity"="{tag}"]({b["min_lat"]},{b["min_lon"]},{b["max_lat"]},{b["max_lon"]});'
        for tag in FOOD_DRINK_AMENITY_TAGS
    )
    query = f"""
    [out:json][timeout:55];
    (
{clauses}
    );
    out center;
    """
    print("  Fetching food/drink venue data for the whole route area (one-time)...")
    data = _query_overpass(query)

    venues = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        elif el.get("type") == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            continue

        venues.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "amenity_type": tags.get("amenity", "unknown"),
            "cuisine": tags.get("cuisine"),
        })

    print(f"  Found {len(venues)} NAMED food/drink venue(s) in the route area.\n")
    return venues


if __name__ == "__main__":
    print("--- Live test: fetching food/drink venues (needs real internet access) ---")
    # A reasonably venue-dense small bbox -- central Guildford, used
    # earlier this session as a known-dense test area for buildings,
    # so a real, comparable sanity check here too.
    guildford_bbox = {"min_lat": 51.230, "max_lat": 51.245, "min_lon": -0.580, "max_lon": -0.565}
    venues = fetch_food_drink_venues_in_bbox(guildford_bbox)
    print(f"Found {len(venues)} venue(s):")
    for v in venues[:10]:
        cuisine_note = f" ({v['cuisine']})" if v.get("cuisine") else ""
        print(f"  {v['name']} — {v['amenity_type']}{cuisine_note} at ({v['lat']:.4f}, {v['lon']:.4f})")
    if venues:
        print("\nPASSED — got real venue data back")
    else:
        print("\nNo results — could mean no internet access here (expected in Claude's sandbox), "
              "or a real query problem worth checking when run somewhere with real network access.")
