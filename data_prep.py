"""
Yalies Data Preparation - Optimized Version

Fetches student data from the Yalies API and prepares it for embedding.
Optimizations:
- Environment variable for API key
- Parallel page fetching
- Caching with force override
- Modular functions for web app reuse
- Retry logic and error handling
- Progress bar
"""

import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

import requests

#-------------------------------------------------------------------------#
# Configuration
#-------------------------------------------------------------------------#

API_URL = "https://api.yalies.io/v2/people"
OUTPUT_FILE = "yalies.json"
PAGE_SIZE = 100
MAX_PAGES = 100
MAX_WORKERS = 10
MAX_RETRIES = 3

#-------------------------------------------------------------------------#
# API Key Management
#-------------------------------------------------------------------------#

def get_api_key():
    """Get API key from environment variable or .env file."""
    api_key = os.environ.get("YALIES_API_KEY")
    
    if not api_key:
        # Fallback to hardcoded key (for backwards compatibility)
        # TODO: Remove this fallback in production
        api_key = "d8szIfAZepJlvKUB8wWfyfLRN9uEi7QoVGk5RHMhMwmPXEx1ctyWcA"
        print("⚠️  Using hardcoded API key. Set YALIES_API_KEY environment variable for production.")
    
    return api_key


#-------------------------------------------------------------------------#
# API Functions
#-------------------------------------------------------------------------#

def fetch_page(page_num, api_key, page_size=PAGE_SIZE):
    """
    Fetch a single page of results from the Yalies API.
    
    Args:
        page_num: Page number to fetch (1-indexed)
        api_key: Yalies API key
        page_size: Number of results per page
    
    Returns:
        List of people dicts, or None if failed
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "query": "",
        "page_size": page_size,
        "page": page_num
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=body, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data if data else []
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
            else:
                return None
    
    return None


def fetch_all_pages(n_pages=MAX_PAGES, max_workers=MAX_WORKERS):
    """
    Fetch all pages in parallel using ThreadPoolExecutor.
    
    Args:
        n_pages: Maximum number of pages to fetch
        max_workers: Number of parallel workers
    
    Returns:
        Tuple of (all_people, failed_pages)
    """
    api_key = get_api_key()
    all_people = []
    failed_pages = []
    
    print(f"\nFetching up to {n_pages} pages ({PAGE_SIZE} people each)...")
    print(f"Using {max_workers} parallel workers\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all page fetches
        future_to_page = {
            executor.submit(fetch_page, page, api_key): page 
            for page in range(1, n_pages + 1)
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_page), total=n_pages, desc="Fetching pages"):
            page_num = future_to_page[future]
            try:
                result = future.result()
                if result is None:
                    failed_pages.append(page_num)
                elif len(result) == 0:
                    # Empty page means we've reached the end
                    pass
                else:
                    all_people.extend(result)
            except Exception as e:
                failed_pages.append(page_num)
    
    return all_people, failed_pages


def filter_with_photos(people):
    """
    Filter to only include people who have photos.
    
    Args:
        people: List of people dicts
    
    Returns:
        Filtered list with only people who have images
    """
    return [p for p in people if p.get('image') is not None]


#-------------------------------------------------------------------------#
# Main Functions
#-------------------------------------------------------------------------#

def scrape_yalies(n_pages=MAX_PAGES, force=False):
    """
    Main function to scrape and save Yalies data.
    
    Args:
        n_pages: Maximum number of pages to fetch
        force: If True, ignore cache and re-fetch
    
    Returns:
        List of people with photos
    """
    # Check cache
    if os.path.exists(OUTPUT_FILE) and not force:
        print(f"✓ Cache exists: {OUTPUT_FILE}")
        print("  Use --force to re-fetch data\n")
        
        with open(OUTPUT_FILE, 'r') as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} people from cache")
        return cached
    
    # Fetch all pages
    all_people, failed_pages = fetch_all_pages(n_pages)
    
    # Report failures
    if failed_pages:
        print(f"\n⚠️  Failed to fetch {len(failed_pages)} pages: {sorted(failed_pages)[:10]}...")
    
    # Filter to people with photos
    print(f"\nTotal people fetched: {len(all_people)}")
    people_with_photos = filter_with_photos(all_people)
    print(f"People with photos: {len(people_with_photos)}")
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(people_with_photos, f, indent=2)
    
    print(f"\n✅ Saved {len(people_with_photos)} people to {OUTPUT_FILE}")
    
    return people_with_photos


def get_cached_data():
    """
    Get cached data if available (for web app use).
    
    Returns:
        List of people or None if no cache
    """
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            return json.load(f)
    return None


#-------------------------------------------------------------------------#
# Entry Point
#-------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Yalies data from API")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-fetch, ignore cache")
    parser.add_argument("--pages", "-p", type=int, default=MAX_PAGES,
                        help=f"Maximum pages to fetch (default: {MAX_PAGES})")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS,
                        help=f"Parallel workers (default: {MAX_WORKERS})")
    
    args = parser.parse_args()
    
    # Update global config
    MAX_WORKERS = args.workers
    
    scrape_yalies(n_pages=args.pages, force=args.force)
