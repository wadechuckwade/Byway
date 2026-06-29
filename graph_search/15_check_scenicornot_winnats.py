"""
Byway — Direct check: does ScenicOrNot have a real, nearby rating for
Winnats Pass -- the exact road that exposed today's core scoring gap?

WHY THIS EXISTS: today's whole debugging arc traced back to one root
cause -- no current signal captures dramatic terrain (a road through a
narrow limestone gorge, towering crags either side). ScenicOrNot is a
real, crowd-sourced scenicness dataset (217,000 geotagged photos,
~95% of Great Britain's 1km grid squares, 1.5M+ ratings on a 1-10
scale) -- directly in this project's own previously-cited research
lineage (Seresinhe/Preis/Moat, building on Quercia). Licensed under
the Open Database Licence (ODbL) -- same license family as OSM itself,
confirmed directly from the dataset's current maintainers (Data
Science Lab, Warwick Business School), so no commercial-use blocker.

This is the single most decisive, cheap check available before
committing to integrate this as a permanent pipeline component: if
Winnats Pass has a real high rating nearby, that's direct confirmation
this fixes the actual gap we found today, not just a plausible theory.

Run from anywhere with internet access (downloads ~20-30MB once, then
caches locally).
"""

import os
import csv
import math
import urllib.request

VOTES_URL = "http://scenicornot.datasciencelab.co.uk/votes.tsv"
LOCAL_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenicornot_votes.tsv")

# Winnats Pass -- converged estimate from multiple independent sources
# (Mindat, Wikipedia, OS Maps, walkingbritain.co.uk), all clustering
# within ~300m of each other.
WINNATS_PASS = {"lat": 53.3408, "lon": -1.8000}
SEARCH_RADIUS_M = 1500  # generous -- a 1km grid square plus margin
                         # for our coordinate estimate being slightly off


def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


if __name__ == "__main__":
    if not os.path.exists(LOCAL_CACHE_PATH):
        print(f"Downloading ScenicOrNot votes dataset (one-time, ~20-30MB) from {VOTES_URL} ...")
        urllib.request.urlretrieve(VOTES_URL, LOCAL_CACHE_PATH)
        print("Downloaded and cached locally for future runs.\n")
    else:
        print(f"Using already-downloaded local copy at {LOCAL_CACHE_PATH}\n")

    nearby = []
    with open(LOCAL_CACHE_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 7:
                continue
            try:
                lat, lon, avg_rating = float(row[1]), float(row[2]), float(row[3])
            except ValueError:
                continue
            dist = haversine_distance_m(WINNATS_PASS["lat"], WINNATS_PASS["lon"], lat, lon)
            if dist <= SEARCH_RADIUS_M:
                nearby.append({
                    "id": row[0], "lat": lat, "lon": lon, "avg_rating": avg_rating,
                    "distance_m": round(dist, 1), "votes": row[5], "geograph_url": row[6],
                })

    nearby.sort(key=lambda r: r["distance_m"])

    print(f"Found {len(nearby)} ScenicOrNot rating(s) within {SEARCH_RADIUS_M}m of Winnats Pass "
          f"({WINNATS_PASS['lat']}, {WINNATS_PASS['lon']}):\n")
    for r in nearby:
        print(f"  {r['distance_m']}m away: avg_rating={r['avg_rating']}/10 "
              f"({r['votes']} votes) -- {r['geograph_url']}")

    if nearby:
        best = max(nearby, key=lambda r: r["avg_rating"])
        print(f"\nBest nearby rating: {best['avg_rating']}/10, {best['distance_m']}m away.")
        print("For reference: ScenicOrNot's overall distribution typically centers around "
              "4-5/10, with genuinely standout scenic locations scoring 7+.")
    else:
        print("\nNo rating found within range -- worth widening SEARCH_RADIUS_M and re-checking, "
              "since our coordinate estimate for Winnats Pass itself carries some uncertainty.")
