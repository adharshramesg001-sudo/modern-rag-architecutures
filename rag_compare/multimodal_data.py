"""Builds the real chart image and table used by Multimodal RAG from the
quarterly revenue numbers in corpus.py — the chart is rendered with
matplotlib from actual data, not a static asset."""

from __future__ import annotations

import io
from typing import Tuple

import pandas as pd

from rag_compare.corpus import REVENUE_QUARTERS, REVENUE_VALUES_M


def revenue_table() -> pd.DataFrame:
    return pd.DataFrame({"Quarter": REVENUE_QUARTERS, "Revenue ($M)": REVENUE_VALUES_M})


def revenue_trend_description() -> str:
    first, last = REVENUE_VALUES_M[0], REVENUE_VALUES_M[-1]
    pct_change = (last - first) / first * 100
    quarter_deltas = [
        REVENUE_VALUES_M[i] - REVENUE_VALUES_M[i - 1] for i in range(1, len(REVENUE_VALUES_M))
    ]
    all_up = all(d > 0 for d in quarter_deltas)
    trend_word = "grew every quarter" if all_up else "fluctuated"
    return (
        f"Revenue {trend_word} across 2025, from ${first:.1f}M in {REVENUE_QUARTERS[0]} to "
        f"${last:.1f}M in {REVENUE_QUARTERS[-1]} — a cumulative increase of {pct_change:.0f}%."
    )


def revenue_chart_png() -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(REVENUE_QUARTERS, REVENUE_VALUES_M, color="#0f766e")
    ax.bar_label(bars, fmt="$%.1fM")
    ax.set_ylabel("Revenue ($M)")
    ax.set_title("Quarterly Revenue — 2025")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
