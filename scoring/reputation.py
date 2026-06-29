import time
import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BywayApp-DevelopmentPrototype/0.1 (research prototype)"
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 4

GENERIC_NAME_SKIP_LIST = {
    "unnamed road", "main road", "main street", "high street", "the street",
    "station road", "church street", "mill street", "the hill", "back street",
    "park road", "victoria road", "victoria street", "church green",
}

def _wikipedia_request(params):
    full_params = {"format": "json", **params}
    for attempt in range(MAX_RETRIES):
        time.sleep(REQUEST_DELAY_SECONDS)
        try:
            response = requests.get(WIKIPEDIA_API_URL, params=full_params,
                                     headers={"User-Agent": USER_AGENT}, timeout=15)
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 * (attempt + 1)
                print(f"    (Wikipedia request failed ({e}), waiting {wait_time}s and retrying...)")
                time.sleep(wait_time)
                continue
            else:
                raise
        if response.status_code == 429:
            wait_time = REQUEST_DELAY_SECONDS * (attempt + 3)
            print(f"    (Wikipedia rate limit hit, waiting {wait_time:.1f}s and retrying...)")
            time.sleep(wait_time)
            continue
        response.raise_for_status()
        return response.json()
    raise requests.exceptions.RequestException(f"Wikipedia API still failing after {MAX_RETRIES} attempts.")

def search_wikipedia(road_name, place_context):
    query = f"{road_name} {place_context}"
    data = _wikipedia_request({"action": "query", "list": "search", "srsearch": query, "srlimit": 1})
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None

def fetch_article_extract(page_title, max_chars=1200):
    data = _wikipedia_request({"action": "query", "prop": "extracts", "exintro": True,
                                "explaintext": True, "titles": page_title})
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None
        extract = page_data.get("extract", "").strip()
        if extract:
            return extract[:max_chars]
    return None

def _looks_relevant(road_name, place_context, article_title, extract):
    if not extract:
        return False
    road_name_lower = road_name.lower().strip()
    place_lower = place_context.lower().strip()
    combined_text = (article_title.lower() + " " + extract.lower())
    road_name_found = road_name_lower in combined_text
    place_found = place_lower in combined_text
    generic_fragments = {"way", "road", "lane", "street", "hill", "close", "avenue", "drive"}
    road_name_is_weak = road_name_lower in generic_fragments or len(road_name_lower) <= 4
    if road_name_is_weak:
        return road_name_found and place_found
    return road_name_found

def get_road_reputation(road_name, place_context):
    no_signal = {"found": False, "source_title": None, "source_url": None, "extract": None}
    if road_name.strip().lower() in GENERIC_NAME_SKIP_LIST:
        return no_signal
    try:
        matched_title = search_wikipedia(road_name, place_context)
    except requests.exceptions.RequestException as e:
        print(f"    (Could not search Wikipedia for '{road_name}': {e})")
        return no_signal
    if not matched_title:
        return no_signal
    try:
        extract = fetch_article_extract(matched_title)
    except requests.exceptions.RequestException as e:
        print(f"    (Could not fetch Wikipedia extract for '{matched_title}': {e})")
        return no_signal
    if not _looks_relevant(road_name, place_context, matched_title, extract):
        return no_signal
    source_url = "https://en.wikipedia.org/wiki/" + matched_title.replace(" ", "_")
    return {"found": True, "source_title": matched_title, "source_url": source_url, "extract": extract}

if __name__ == "__main__":
    result = get_road_reputation("Main Road", "Castleton")
    assert result["found"] is False
    print("Self-test passed")
