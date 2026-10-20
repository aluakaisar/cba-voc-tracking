"""Run the full analysis: topics, tracking, sentiment, figures.

    python src/run_analysis.py

Reads data/raw/google_play.csv (regenerate it with collect_playstore.py) and
writes tables to outputs/ and charts to outputs/figures/.

Periods are quarters, not months. After dropping reviews under five words the
recent years have well under a hundred reviews a month, and a topic share
estimated from that few reviews moves more from noise than from customers.
Quarters keep roughly 200 to 600 reviews per period.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import model
import tracking
from config import FIGURES, RAW, ROOT

OUT = ROOT / "outputs"


def main() -> None:
    reviews = model.load_reviews(RAW / "google_play.csv")

    # Keep whole quarters only: the first and last are partial.
    reviews["period"] = reviews["date"].dt.to_period("Q")
    full = reviews["period"].between(
        reviews["period"].min() + 1, reviews["period"].max() - 1
    )
    reviews = reviews[full].reset_index(drop=True)

    weights, nmf, vec = model.fit_topics(reviews["text"])
    terms = model.top_terms(nmf, vec)
    labels = model.label_topics(terms)
    tp, keep = model.to_prevalence(weights, labels)
    reviews = reviews[keep].reset_index(drop=True)
    periods = reviews["period"].astype(str)

    terms_df = pd.DataFrame({"label": labels, "top_terms": [", ".join(t) for t in terms]})
    terms_df.to_csv(OUT / "topic_terms.csv", index=False)
    print(terms_df.to_string(index=False))

    result = tracking.track(tp, periods)
    summary = result["summary"]
    imtp = result["imtp"]

    reviews["sentiment"] = model.sentiment(reviews["text"])
    sent_overall = model.topic_sentiment(tp, reviews["sentiment"].to_numpy())
    sent_period = model.topic_sentiment(tp, reviews["sentiment"].to_numpy(), periods)
    rating_overall = tp.mul(reviews["rating"].to_numpy(), axis=0).sum() / tp.sum()

    summary["mean_sentiment"] = sent_overall
    summary["mean_rating"] = rating_overall
    summary.round(4).to_csv(OUT / "topic_summary.csv")
    imtp.round(4).to_csv(OUT / "imtp_quarterly.csv")
    sent_period.round(4).to_csv(OUT / "sentiment_quarterly.csv")
    pd.DataFrame([a.to_dict() for a in result["anomalies"]]).to_csv(
        OUT / "anomalies.csv", index=False
    )
    meta = {
        "reviews_modelled": int(len(reviews)),
        "first_period": periods.min(),
        "last_period": periods.max(),
        "n_periods": int(imtp.shape[0]),
        "reviews_per_period_min": int(result["counts"].min()),
        "reviews_per_period_max": int(result["counts"].max()),
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print()
    print(summary.round(3).to_string())
    print()
    for a in result["anomalies"]:
        print(a)
    print(meta)

    plot_shares(imtp, summary)
    plot_trend_vs_sentiment(summary)
    plot_share_and_mood(imtp, sent_period, "Tap & pay (Android NFC)", "tap_and_pay.png")
    plot_share_and_mood(imtp, sent_period, "Customer service & staff", "customer_service.png")


def plot_shares(imtp: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Small multiples: one panel per topic, same y-axis, so shares compare."""
    order = summary.sort_values("MTP", ascending=False).index
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    x = range(len(imtp))
    for ax, topic in zip(axes.flat, order):
        ax.plot(x, imtp[topic], color="#2a6f97", lw=1.6)
        ax.set_title(topic, fontsize=9)
        ax.grid(alpha=0.25)
    ticks = [i for i, p in enumerate(imtp.index) if p.endswith("Q1")]
    for ax in axes[-1]:
        ax.set_xticks(ticks[::2])
        ax.set_xticklabels([imtp.index[i][:4] for i in ticks[::2]], fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of conversation")
    fig.suptitle("What CommBank app reviewers talk about, by quarter", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "topic_shares.png", dpi=150)
    plt.close(fig)


def plot_trend_vs_sentiment(summary: pd.DataFrame) -> None:
    """Where each topic sits on 'growing or shrinking' against 'liked or not'."""
    fig, ax = plt.subplots(figsize=(8, 6))
    s = summary
    ax.scatter(s["imtp_slope"] * 4, s["mean_sentiment"], s=s["MTP"] * 3000,
               color="#2a6f97", alpha=0.6)
    # Two bubbles sit almost on top of each other; nudge their labels apart.
    nudge = {"Money transfers": (-8, -16), "Cards, cardless cash & notifications": (10, 6)}
    for topic, row in s.iterrows():
        dx, dy = nudge.get(topic, (6, 5))
        ax.annotate(topic, (row["imtp_slope"] * 4, row["mean_sentiment"]),
                    fontsize=8, xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("change in share of conversation per year (Theil-Sen slope)")
    ax.set_ylabel("mean sentiment (VADER, -1 to +1)")
    ax.set_title("Growing or shrinking, and how people feel about it\n(bubble size = overall share)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "trend_vs_sentiment.png", dpi=150)
    plt.close(fig)


def plot_share_and_mood(imtp: pd.DataFrame, sent: pd.DataFrame, topic: str, fname: str) -> None:
    """The case for the sentiment layer: how much a topic is discussed, and how
    people feel when they discuss it, on separate panels sharing one time axis.

    Quarterly sentiment on a single topic is noisy, so the mood line is a
    four-quarter rolling mean with the raw values shown faintly behind it.
    """
    x = list(range(len(imtp)))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    a1.plot(x, imtp[topic], color="#2a6f97", lw=1.8)
    a1.set_ylabel("share of conversation")
    a1.set_title(f"{topic}: how much it is discussed, and how people feel about it")
    a2.plot(x, sent[topic], color="#c1121f", lw=0.8, alpha=0.3)
    a2.plot(x, sent[topic].rolling(4, min_periods=2).mean(), color="#c1121f", lw=1.8)
    a2.axhline(0, color="grey", lw=0.8)
    a2.set_ylabel("sentiment (4-quarter mean)")
    for a in (a1, a2):
        a.grid(alpha=0.25)
    ticks = [i for i, p in enumerate(imtp.index) if p.endswith("Q1")]
    a2.set_xticks(ticks)
    a2.set_xticklabels([imtp.index[i][:4] for i in ticks], fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
