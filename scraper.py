# scraper.py
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Selenium (for dynamic sites)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ── Helpers ─────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)           # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # remove non-ASCII
    return text.strip()

def is_dynamic(url: str) -> bool:
    dynamic_domains = ["amazon", "flipkart", "yelp", "tripadvisor",
                       "zomato", "trustpilot", "booking", "makemytrip"]
    domain = get_domain(url)
    return any(d in domain for d in dynamic_domains)

# ── Static Scraper (requests + BeautifulSoup) ────────────────────────────

def scrape_static(url: str) -> list[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return extract_reviews_from_soup(soup, url)
    except Exception as e:
        print(f"⚠️ Static fetch failed ({e}), trying Selenium...")
        return scrape_dynamic(url)   # ← auto fallback to Selenium
# ── Dynamic Scraper (Selenium) ────────────────────────────────────────────

def get_driver():
    options = Options()
    options.add_argument("--headless")           # no browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver

def scrape_dynamic(url: str) -> list[str]:
    driver = get_driver()
    reviews = []
    try:
        driver.get(url)

        # Check if Amazon is showing robot/captcha page
        time.sleep(4)
        page_text = driver.page_source.lower()

        if "robot" in page_text or "captcha" in page_text or "enter the characters" in page_text:
            raise RuntimeError(
                "Amazon detected automated access and showed a CAPTCHA. "
                "Try the product page URL (amazon.in/dp/XXXXX) instead of the reviews page."
            )

        if "sign in" in page_text and "review" not in page_text:
            raise RuntimeError(
                "Amazon is asking for login to view this page. "
                "Use the product page URL instead: amazon.in/dp/PRODUCTID"
            )

        # Scroll to load lazy content
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.2)

        # Scroll back up and down again to trigger all lazy loads
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        reviews = extract_reviews_from_soup(soup, url)

        # Save debug file to inspect if still empty
        if not reviews:
            with open("debug_page.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("⚠️  No reviews found. Page saved to debug_page.html")

    finally:
        driver.quit()
    return reviews

# ── Site-Specific Extractors ──────────────────────────────────────────────

def extract_reviews_from_soup(soup: BeautifulSoup, url: str) -> list[str]:
    domain = get_domain(url)

    if "amazon" in domain:
        return extract_amazon(soup)
    elif "flipkart" in domain:
        return extract_flipkart(soup)
    elif "yelp" in domain:
        return extract_yelp(soup)
    elif "tripadvisor" in domain:
        return extract_tripadvisor(soup)
    else:
        return extract_generic(soup)

def extract_amazon(soup: BeautifulSoup) -> list[str]:
    reviews = []

    # Selector 1 — standard review body
    for tag in soup.select("span[data-hook='review-body'] span"):
        t = clean_text(tag.get_text())
        if len(t) > 20:
            reviews.append(t)

    # Selector 2 — review text content class
    if not reviews:
        for tag in soup.select(".review-text-content span"):
            t = clean_text(tag.get_text())
            if len(t) > 20:
                reviews.append(t)

    # Selector 3 — product page reviews (dp/ URL)
    if not reviews:
        for tag in soup.select("div[data-hook='review'] span[data-hook='review-body']"):
            t = clean_text(tag.get_text())
            if len(t) > 20:
                reviews.append(t)

    # Selector 4 — cr-review-list (another Amazon layout)
    if not reviews:
        for tag in soup.select("div.a-expander-content p"):
            t = clean_text(tag.get_text())
            if len(t) > 20:
                reviews.append(t)

    # Selector 5 — generic fallback for any Amazon layout
    if not reviews:
        for tag in soup.select("[class*='review-text'], [class*='reviewText']"):
            t = clean_text(tag.get_text())
            if len(t) > 20:
                reviews.append(t)

    return reviews

def extract_flipkart(soup: BeautifulSoup) -> list[str]:
    reviews = []
    selectors = [
        "div.ZmyHeo",          # review text block
        "div.row.feRevW",
        "p.z9E0IG",
        "div._6K-7Co",
    ]
    for sel in selectors:
        for tag in soup.select(sel):
            t = clean_text(tag.get_text())
            if len(t) > 20:
                reviews.append(t)
        if reviews:
            break
    return reviews

def extract_yelp(soup: BeautifulSoup) -> list[str]:
    reviews = []
    for tag in soup.select("p.comment__373c0__Nsutg, span.raw__373c0__3rcx7"):
        t = clean_text(tag.get_text())
        if len(t) > 20:
            reviews.append(t)
    # Fallback
    if not reviews:
        for tag in soup.select("[class*='comment']"):
            t = clean_text(tag.get_text())
            if len(t) > 30:
                reviews.append(t)
    return reviews

def extract_tripadvisor(soup: BeautifulSoup) -> list[str]:
    reviews = []
    for tag in soup.select("div.biGQs._P.pZUbB.KxBGd span"):
        t = clean_text(tag.get_text())
        if len(t) > 20:
            reviews.append(t)
    if not reviews:
        for tag in soup.select("q.XllAv span, div.review-container"):
            t = clean_text(tag.get_text())
            if len(t) > 20:
                reviews.append(t)
    return reviews

def extract_generic(soup: BeautifulSoup) -> list[str]:
    """
    Fallback: find the largest clusters of <p> tags —
    works on most blog/review sites.
    """
    candidates = []
    for tag in soup.find_all("p"):
        t = clean_text(tag.get_text())
        if len(t) > 40:
            candidates.append(t)
    return candidates[:30]  # cap at 30

# ── Main Entry Point ──────────────────────────────────────────────────────

def scrape_reviews(url: str) -> list[str]:
    if not url.startswith("http"):
        raise ValueError("URL must start with http:// or https://")

    # ── Auto-fix Amazon URLs ──────────────────────────────
    if "amazon" in get_domain(url):
        # Convert product-reviews URL to dp URL
        if "/product-reviews/" in url:
            product_id = url.split("/product-reviews/")[1].split("/")[0].split("?")[0]
            url = f"https://www.amazon.in/dp/{product_id}"
            print(f"🔄 Auto-converted URL to: {url}")

        # Convert gp/reviews URL to dp URL
        elif "/gp/reviews/" in url:
            product_id = url.split("/gp/reviews/")[1].split("/")[0].split("?")[0]
            url = f"https://www.amazon.in/dp/{product_id}"
            print(f"🔄 Auto-converted URL to: {url}")

    if is_dynamic(url):
        reviews = scrape_dynamic(url)
    else:
        reviews = scrape_static(url)

    # Deduplicate while preserving order
    seen, unique = set(), []
    for r in reviews:
        if r not in seen and len(r) > 15:
            seen.add(r)
            unique.append(r)

    if not unique:
        raise RuntimeError(
            "No reviews found on this page. The site may block scrapers, "
            "or the page structure may have changed."
        )
    return unique