"""
Collect CommBank reviews from the Apple App Store.

Run this on your own machine; it needs outbound access to apple.com.

    python src/collect_appstore.py
    python src/collect_appstore.py --max 3000
    python src/collect_appstore.py --countries au,nz

Writes data/raw/app_store.csv, merging and de-duplicating on re-runs.

Apple's review feed is considerably stingier than Google's: it serves a few
thousand of the most recent reviews per storefront and then stops, whatever
you ask for. Two consequences for the analysis.

First, expect less history here than from Google Play. Check the date span the
script prints before deciding your tracking window.

Second, if you pull extra storefronts to raise the volume, keep the country
column and check it before pooling. Reviews from different storefronts are
different customer populations under different regulations, and pooling them
silently is the kind of thing a marker asks about. For a Commonwealth Bank
study, au alone is the defensible choice.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from config import COUNTRY, IOS_APP_ID, IOS_APP_NAME, RAW, SCHEMA  # noqa: E402


def fetch_country(
    country: str, app_name: str, app_id: int, how_many: int, pause: float
) -> pd.DataFrame:
    from app_store_scraper import AppStore

    store = AppStore(country=country, app_name=app_name, app_id=app_id)
    store.review(how_many=how_many, sleep=pause)

    rows = store.reviews or []
    print(f"  {country}: {len(rows):,} reviews")
    if not rows:
        return pd.DataFrame(columns=[*SCHEMA, "country"])

    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            # Apple's feed has no stable public review id, so build one from
            # the fields that together identify a review.
            "review_id": (
                country
                + "-"
                + df["date"].astype(str)
                + "-"
                + df["userName"].astype(str)
            ).str.replace(r"\s+", "_", regex=True),
            "source": "app_store",
            "date": pd.to_datetime(df["date"]),
            "date_precision": "exact",
            "rating": df["rating"],
            "title": df.get("title", pd.Series(dtype=object)).fillna(""),
            "text": df["review"].fillna(""),
            "version": None,
            "country": country,
        }
    )
    time.sleep(pause)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app-name", default=IOS_APP_NAME)
    ap.add_argument("--app-id", type=int, default=IOS_APP_ID)
    ap.add_argument("--countries", default=COUNTRY, help="comma separated")
    ap.add_argument("--max", type=int, default=5000, help="target per storefront")
    ap.add_argument("--pause", type=float, default=0.5)
    args = ap.parse_args()

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    print(f"Collecting App Store reviews for id={args.app_id} from {countries}")

    frames = [
        fetch_country(c, args.app_name, args.app_id, args.max, args.pause)
        for c in countries
    ]
    fresh = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if fresh.empty:
        print("Nothing collected. Check your network access to apple.com.")
        return 1

    path = RAW / "app_store.csv"
    if path.exists():
        before = len(pd.read_csv(path))
        fresh = (
            pd.concat([pd.read_csv(path, parse_dates=["date"]), fresh])
            .drop_duplicates(subset="review_id", keep="last")
            .sort_values("date")
        )
        print(f"Merged with {before} existing rows")

    fresh.to_csv(path, index=False)
    print(f"\nWrote {len(fresh):,} reviews to {path}")
    print(f"Date span: {fresh['date'].min():%Y-%m-%d} to {fresh['date'].max():%Y-%m-%d}")
    print(f"Mean rating: {fresh['rating'].mean():.2f}")
    print("\nReviews per year:")
    print(fresh.groupby(fresh["date"].dt.year).size().to_string())

    thin = fresh.groupby(fresh["date"].dt.to_period("M")).size()
    sparse = thin[thin < 20]
    if len(sparse):
        print(
            f"\nNote: {len(sparse)} month(s) have under 20 reviews. "
            "Consider quarterly buckets, or trim the window to where volume holds up."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
