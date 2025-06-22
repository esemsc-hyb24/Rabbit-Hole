import os
import shutil
import sqlite3
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime, timedelta
from pathlib import Path

# === CONFIGURATION ===
FIREFOX_PROFILE_DIR = Path.home() / ".mozilla" / "firefox"
COPY_DB_PATH = "places_copy.sqlite"
CSV_OUTPUT_FILE = "firefox_history_export.csv"
MAX_ENTRIES = 100  # Limit for performance

# === FUNCTIONS ===

def find_places_db(profile_path):
    db_path = profile_path
    if db_path.exists():
        return db_path
    raise FileNotFoundError(f"'places.sqlite' not found at {db_path}")

def copy_db(source_path):
    shutil.copy(source_path, COPY_DB_PATH)

def firefox_timestamp_to_datetime(microseconds):
    return datetime(1970, 1, 1) + timedelta(microseconds=microseconds)

def extract_domain(url):
    return urlparse(url).netloc

def scrape_page_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=5, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"Error scraping: {e}"

def read_history(db_path, max_entries):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = """
        SELECT moz_places.url, moz_places.title, moz_historyvisits.visit_date
        FROM moz_places
        JOIN moz_historyvisits ON moz_places.id = moz_historyvisits.place_id
        ORDER BY visit_date DESC
        LIMIT ?
    """
    cursor.execute(query, (max_entries,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for url, title, visit_date in rows:
        visit_time = firefox_timestamp_to_datetime(visit_date)
        domain = extract_domain(url)
        text = scrape_page_text(url)
        history.append({
            "url": url,
            "title": title,
            "domain": domain,
            "visit_time": visit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "page_text": text[:1000],  # Limit text size for CSV
        })
    return history

def write_to_csv(history, filename):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["url", "title", "domain", "visit_time", "page_text"])
        writer.writeheader()
        writer.writerows(history)

# === MAIN SCRIPT ===

def main():
    print("📁 Locating Firefox history...")
    original_db = find_places_db(FIREFOX_PROFILE_DIR)

    print("📋 Copying history DB...")
    copy_db(original_db)

    print("🔍 Reading and scraping history entries...")
    history_data = read_history(COPY_DB_PATH, MAX_ENTRIES)

    print(f"📤 Writing to CSV: {CSV_OUTPUT_FILE}")
    write_to_csv(history_data, CSV_OUTPUT_FILE)

    print("✅ Done! CSV exported.")

if __name__ == "__main__":
    main()
