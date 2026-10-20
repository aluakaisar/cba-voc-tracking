"""Shared identifiers and paths for the CBA digital VoC collection."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"

for _d in (RAW, PROCESSED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Commonwealth Bank of Australia, CommBank app.
ANDROID_PACKAGE = "com.commbank.netbank"
IOS_APP_ID = 310251202
IOS_APP_NAME = "commbank"
PRODUCTREVIEW_SLUG = "commonwealth-bank"

COUNTRY = "au"
LANG = "en"

# Canonical column set every collector writes, so the three sources stack.
#
#   review_id   source-unique id, used to drop duplicates on re-runs
#   source      "google_play" | "app_store" | "productreview"
#   date        timestamp; exact for the app stores, reconstructed from a
#               relative timestamp for productreview
#   date_precision  "exact" | "day" | "week" | "month" | "year" | "unknown".
#               Never bucket a review finer than its own precision: a review
#               the source dated only as "3y" cannot go in a monthly bin.
#   rating      1-5 stars, or NaN where the source has none
#   title       may be empty
#   text        the review body, the only field the topic model reads
#   version     app version where available
SCHEMA = [
    "review_id",
    "source",
    "date",
    "date_precision",
    "rating",
    "title",
    "text",
    "version",
]

# Only the productreview collector emits this; the app store collectors have
# exact timestamps and leave it out.
OPTIONAL_COLUMNS = ["date_uncertainty_days", "country"]
