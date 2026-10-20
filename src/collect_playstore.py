"""
Collect CommBank reviews from Google Play.

Run this on your own machine; it needs outbound access to play.google.com.

    python src/collect_playstore.py                  # everything available
    python src/collect_playstore.py --max 5000       # cap it while testing
    python src/collect_playstore.py --since 2019-01-01

Writes data/raw/google_play.csv. Re-running merges with whatever is already
there and de-duplicates on review_id, so it is safe to stop it and resume.

Google Play serves reviews newest-first through a continuation token. There is
no documented page limit, but the endpoint rate-limits, so this sleeps between
pages and stops cleanly if the token dries up.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from config import ANDROID_PACKAGE, COUNTRY, LANG, RAW, SCHEMA  # noqa: E402

BATCH = 200  # Google Play's practical maximum per request.


def fetch(
    package: str,
    max_reviews: int | None,
    since: datetime | None,
    pause: float,
) -> pd.DataFrame:
    from google_play_scraper import Sort, reviews

    collected: list[dict] = []
    token = None
    page = 0

    while True:
        try:
            batch, token = reviews(
                package,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=BATCH,
                continuation_token=token,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  stopped after {len(collected)} reviews: {exc}", file=sys.stderr)
            break

        if not batch:
            break

        page += 1
        collected.extend(batch)
        oldest = batch[-1]["at"]
        print(f"  page {page:>4}  total {len(collected):>7}  reached {oldest:%Y-%m-%d}")

        if since is not None and oldest < since:
            print(f"  reached the --since cutoff ({since:%Y-%m-%d})")
            break
        if max_reviews is not None and len(collected) >= max_reviews:
            print(f"  reached the --max cap ({max_reviews})")
            break
        if token is None:
            print("  Google Play returned no continuation token; that is the end")
            break

        time.sleep(pause)

    if not collected:
        return pd.DataFrame(columns=SCHEMA)

    df = pd.DataFrame(collected)
    out = pd.DataFrame(
        {
            "review_id": df["reviewId"],
            "source": "google_play",
            "date": pd.to_datetime(df["at"]),
            "date_precision": "exact",
            "rating": df["score"],
            "title": "",
            "text": df["content"].fillna(""),
            "version": df.get("reviewCreatedVersion", pd.Series(dtype=object)),
        }
    )
    if since is not None:
        out = out[out["date"] >= since]
    if max_reviews is not None:
        out = out.head(max_reviews)
    return out[SCHEMA]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default=ANDROID_PACKAGE)
    ap.add_argument("--max", type=int, default=None, help="stop after N reviews")
    ap.add_argument("--since", default=None, help="stop once older than YYYY-MM-DD")
    ap.add_argument("--pause", type=float, default=0.4, help="seconds between pages")
    args = ap.parse_args()

    since = pd.to_datetime(args.since) if args.since else None
    print(f"Collecting Google Play reviews for {args.package} ({COUNTRY})")

    fresh = fetch(args.package, args.max, since, args.pause)
    if fresh.empty:
        print("Nothing collected. Check your network access to play.google.com.")
        return 1

    path = RAW / "google_play.csv"
    if path.exists():
        before = len(pd.read_csv(path))
        fresh = (
            pd.concat([pd.read_csv(path, parse_dates=["date"]), fresh])
            .drop_duplicates(subset="review_id", keep="last")
            .sort_values("date")
        )
        print(f"Merged with {before} existing rows")

    fresh.to_csv(path, index=False)
    span = f"{fresh['date'].min():%Y-%m-%d} to {fresh['date'].max():%Y-%m-%d}"
    print(f"\nWrote {len(fresh):,} reviews to {path}")
    print(f"Date span: {span}")
    print(f"Mean rating: {fresh['rating'].mean():.2f}")
    print("\nReviews per year:")
    print(fresh.groupby(fresh["date"].dt.year).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
