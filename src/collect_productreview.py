"""
Collect Commonwealth Bank reviews from ProductReview.com.au.

Run this on your own machine.

    python src/collect_productreview.py --inspect        # DO THIS FIRST
    python src/collect_productreview.py --pages 40

Writes data/raw/productreview.csv.

Why this source is worth the trouble
------------------------------------
The app stores tell you about the app. This site has roughly 4,200 reviews of
the *bank* (branches, fees, call centres, disputes, lending), which is the
service-level material a CX unit actually acts on. It also skews heavily
negative, around 1.6 out of 5, against the app stores' much higher scores.
That gap is itself a finding: the two populations are complaining about
different things, and pooling them without saying so would hide it.

Why the dates are a problem
---------------------------
This site publishes *relative* timestamps only: "21h", "3w", "2mo", "4y".
There is no absolute date in the markup. So:

  * recent reviews can be placed to within a day or a week
  * anything older than a year can only be placed to within a year

This module records the granularity per review in ``date_precision`` and the
worst-case error in ``date_uncertainty_days``. Honour them downstream. Do not
bucket a review dated "3y" into a month; the tracking analysis should use this
source at quarterly resolution at best, and for the older tail not at all.
The app store data, which carries exact timestamps, is what the monthly time
series should be built from.

Why you must run --inspect first
--------------------------------
The scraper below was written without being able to reach the site, so its
CSS selectors are a best guess at a layout that changes without notice.
``--inspect`` fetches one page, reports which extraction strategy finds
reviews, and dumps a sample of the markup. Run it, check the counts look
sane, and adjust ``REVIEW_SELECTORS`` if they do not. A scraper that silently
returns 0 rows, or 40 copies of page 1, is the classic way to lose a weekend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from config import PRODUCTREVIEW_SLUG, RAW, SCHEMA  # noqa: E402

BASE = "https://www.productreview.com.au/listings/{slug}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}

# Candidate containers, tried in order. Adjust after --inspect.
REVIEW_SELECTORS = [
    "[itemtype*='Review']",
    "div[data-testid*='review']",
    "article",
    "li[class*='review']",
    "div[class*='review-card']",
]

RELATIVE = re.compile(r"\b(\d+)\s*(h|hr|hrs|hour|d|day|w|wk|week|mo|month|y|yr|year)s?\b", re.I)

# unit -> (days per unit, precision label, worst-case error in days)
UNITS = {
    "h": (1 / 24, "day", 1),
    "d": (1, "day", 1),
    "w": (7, "week", 7),
    "mo": (30.44, "month", 31),
    "y": (365.25, "year", 366),
}


def _unit_key(raw: str) -> str:
    raw = raw.lower()
    if raw.startswith("mo"):
        return "mo"
    if raw.startswith(("y", "yr")):
        return "y"
    if raw.startswith(("w", "wk")):
        return "w"
    if raw.startswith(("d", "day")):
        return "d"
    return "h"


def parse_relative(text: str, now: datetime) -> tuple[datetime | None, str, int]:
    """Turn '3w' into an approximate date, its granularity and its error bar."""
    m = RELATIVE.search(text or "")
    if not m:
        return None, "unknown", 9999
    n = int(m.group(1))
    key = _unit_key(m.group(2))
    per_day, precision, err = UNITS[key]
    # The error bar is set by the unit, not the count: "3y" means somewhere
    # between three and four years ago, the same one-year window as "1y".
    return now - timedelta(days=n * per_day), precision, err


def _from_json_ld(soup) -> list[dict]:
    """Preferred path: schema.org Review markup, if the site publishes it."""
    found: list[dict] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            blob = json.loads(tag.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [blob]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                if node.get("@type") == "Review":
                    rating = node.get("reviewRating") or {}
                    found.append(
                        {
                            "text": node.get("reviewBody") or "",
                            "title": node.get("name") or "",
                            "rating": rating.get("ratingValue"),
                            "raw_date": node.get("datePublished") or "",
                        }
                    )
                stack.extend(node.values())
    return found


def _from_html(soup) -> tuple[list[dict], str]:
    """Fallback: whichever selector finds blocks that look like reviews."""
    from bs4 import BeautifulSoup  # noqa: F401  (import guard only)

    for sel in REVIEW_SELECTORS:
        blocks = soup.select(sel)
        keep = []
        for b in blocks:
            text = b.get_text(" ", strip=True)
            # A review has prose and a relative timestamp; nav chrome has neither.
            if len(text) > 80 and RELATIVE.search(text):
                stars = b.select_one("[aria-label*='star'], [title*='star']")
                label = (stars.get("aria-label") or stars.get("title")) if stars else ""
                rating = None
                if label:
                    m = re.search(r"([0-5](?:\.\d)?)", label)
                    rating = float(m.group(1)) if m else None
                keep.append({"text": text, "title": "", "rating": rating, "raw_date": text})
        if len(keep) >= 3:
            return keep, sel
    return [], ""


def scrape_page(url: str, session) -> tuple[list[dict], str]:
    from bs4 import BeautifulSoup

    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    ld = _from_json_ld(soup)
    if len(ld) >= 3:
        return ld, "json-ld"
    return _from_html(soup)


def inspect(slug: str) -> int:
    import requests
    from bs4 import BeautifulSoup

    url = BASE.format(slug=slug)
    with requests.Session() as s:
        resp = s.get(url, headers=HEADERS, timeout=30)
        print(f"GET {url} -> HTTP {resp.status_code}, {len(resp.text):,} bytes")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        ld = _from_json_ld(soup)
        print(f"\njson-ld Review nodes found: {len(ld)}")
        if ld:
            print("  sample:", json.dumps(ld[0], indent=2)[:500])

        print("\nselector sweep:")
        for sel in REVIEW_SELECTORS:
            blocks = soup.select(sel)
            plausible = [
                b
                for b in blocks
                if len(b.get_text(" ", strip=True)) > 80
                and RELATIVE.search(b.get_text(" ", strip=True))
            ]
            print(f"  {sel:32s} {len(blocks):>4} matched, {len(plausible):>4} plausible")

        rows, how = _from_html(soup)
        print(f"\nchosen strategy: {how or 'NONE'} ({len(rows)} rows)")
        if rows:
            print("  first row text:", rows[0]["text"][:300])
            print("  parsed date:", parse_relative(rows[0]["raw_date"], datetime.now()))
        else:
            print(
                "  No strategy worked. Paste the output of this command back to "
                "Claude, or open the page's HTML and find the container that "
                "wraps one review, then add its selector to REVIEW_SELECTORS."
            )
    return 0 if rows or ld else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default=PRODUCTREVIEW_SLUG)
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--pause", type=float, default=1.5, help="be polite")
    ap.add_argument("--inspect", action="store_true", help="diagnose one page and exit")
    args = ap.parse_args()

    if args.inspect:
        return inspect(args.slug)

    import requests

    now = datetime.now()
    rows: list[dict] = []
    seen_first_texts: set[str] = set()

    with requests.Session() as session:
        for page in range(1, args.pages + 1):
            url = BASE.format(slug=args.slug) + (f"?page={page}" if page > 1 else "")
            try:
                batch, how = scrape_page(url, session)
            except Exception as exc:  # noqa: BLE001
                print(f"  page {page}: {exc}", file=sys.stderr)
                break

            if not batch:
                print(f"  page {page}: no reviews found, stopping")
                break

            # Guard against a site that serves page 1 for every page number.
            fingerprint = batch[0]["text"][:120]
            if fingerprint in seen_first_texts:
                print(f"  page {page} repeats an earlier page, stopping")
                break
            seen_first_texts.add(fingerprint)

            for r in batch:
                date, precision, err = parse_relative(r["raw_date"], now)
                rows.append(
                    {
                        "review_id": f"pr-{abs(hash(r['text'][:200])):x}",
                        "source": "productreview",
                        "date": date,
                        "date_precision": precision,
                        "date_uncertainty_days": err,
                        "rating": r["rating"],
                        "title": r["title"],
                        "text": r["text"],
                        "version": None,
                    }
                )
            print(f"  page {page:>3} via {how:8s} +{len(batch):>3}  total {len(rows):>5}")
            time.sleep(args.pause)

    if not rows:
        print("Nothing collected. Run with --inspect to see why.")
        return 1

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="review_id")
        .dropna(subset=["date"])
        .sort_values("date")
    )
    path = RAW / "productreview.csv"
    df.to_csv(path, index=False)

    print(f"\nWrote {len(df):,} reviews to {path}")
    print(f"Date span: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print("\nDate precision breakdown (this limits your bucket size):")
    print(df["date_precision"].value_counts().to_string())
    coarse = (df["date_precision"].isin(["month", "year"])).mean()
    print(
        f"\n{coarse:.0%} of these reviews can only be dated to the month or year. "
        "Build the monthly series from the app store data and use this source "
        "for the service-level topic vocabulary and quarterly comparison."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
