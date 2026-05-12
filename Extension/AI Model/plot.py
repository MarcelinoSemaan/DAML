import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from collections import Counter

# ── Constants ─────────────────────────────────────────────────────────────────

TIER_ORDER  = ["Clean (0%)", "Very low (1–10%)", "Low (11–30%)",
               "Moderate (31–60%)", "High (61–80%)", "Critical (81–100%)"]
TIER_COLORS = ["#1D9E75", "#97C459", "#EF9F27", "#D85A30", "#E24B4A", "#791F1F"]

def parse_detection_pct(ratio_str):
    if not ratio_str or not isinstance(ratio_str, str):
        return None
    try:
        detected, total = ratio_str.split("/")
        total = int(total)
        return (int(detected) / total * 100) if total > 0 else None
    except Exception:
        return None

def risk_tier(pct):
    if pct is None:  return None
    if pct == 0:     return "Clean (0%)"
    if pct <= 10:    return "Very low (1–10%)"
    if pct <= 30:    return "Low (11–30%)"
    if pct <= 60:    return "Moderate (31–60%)"
    if pct <= 80:    return "High (61–80%)"
    return                  "Critical (81–100%)"

def tier_color_for_pct(pct):
    """Return TIER_COLORS hex for a given detection percentage bucket."""
    if pct == 0:       return TIER_COLORS[0]
    if pct < 10:       return TIER_COLORS[1]
    if pct < 30:       return TIER_COLORS[2]
    if pct < 60:       return TIER_COLORS[3]
    if pct < 80:       return TIER_COLORS[4]
    return                    TIER_COLORS[5]

# ── 1. Loader — file reading untouched, aggregation counters added ─────────────

def load_data(data_dir):
    data_dir    = Path(data_dir)
    files       = sorted(data_dir.glob("*.jsonl"))
    total_files = len(files)
    file_count  = line_count = bad_lines = 0

    label_counter    = Counter()
    filetype_counter = Counter()
    family_counter   = Counter()
    tier_counter     = Counter()
    hist_counter     = Counter()   # detection_pct binned into 5 % buckets

    for p in files:
        file_count += 1
        print(f"Processing file {file_count}/{total_files}: {p.name}")
        with p.open("r", encoding="utf-8") as f:
            for i, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    line_count += 1

                    label = rec.get("label")
                    if label == 1:
                        label_counter["Malware"] += 1
                    elif label == 0:
                        label_counter["Benign"] += 1

                    ft = rec.get("file_type")
                    if ft:
                        filetype_counter[ft] += 1

                    fam = rec.get("family")
                    if fam:
                        family_counter[fam] += 1

                    pct  = parse_detection_pct(rec.get("detection_ratio"))
                    tier = risk_tier(pct)
                    if tier:
                        tier_counter[tier] += 1
                    if pct is not None:
                        bucket = int(pct // 5) * 5   # 0,5,10,...,95
                        hist_counter[bucket] += 1

                except json.JSONDecodeError:
                    bad_lines += 1

        print(f"  -> lines read: {i}, valid: {line_count}, bad: {bad_lines}")

    print(f"\nFinished. Files: {total_files} | Records: {line_count:,} | Bad lines: {bad_lines}")

    return dict(
        label_counts = pd.Series(label_counter),
        top_types    = pd.Series(filetype_counter).nlargest(10),
        top_families = pd.Series(family_counter).nlargest(10).iloc[::-1],
        tier_counts  = pd.Series(tier_counter).reindex(TIER_ORDER, fill_value=0),
        hist_counts  = pd.Series(hist_counter).sort_index(),   # index = bucket start (0..95)
    )

# ── 2. Plot ───────────────────────────────────────────────────────────────────

def fmt_k(x, _):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"

def _bar_labels(ax, bars, orient="h"):
    """Add value labels outside bars with consistent alignment."""
    for bar in bars:
        if orient == "h":
            w = bar.get_width()
            ax.text(
                w, bar.get_y() + bar.get_height() / 2,
                fmt_k(w, None),
                va="center", ha="left", fontsize=8.5,
                color="#444", clip_on=False,
            )
        else:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h,
                    fmt_k(h, None),
                    va="bottom", ha="center", fontsize=8.5,
                    color="#444", clip_on=False,
                )

def plot_dashboard(agg: dict, save_path: str = "malware_analysis.png"):
    label_counts = agg["label_counts"]
    top_types    = agg["top_types"]
    top_families = agg["top_families"]
    tier_counts  = agg["tier_counts"]
    hist_counts  = agg["hist_counts"]

    fig = plt.figure(figsize=(18, 16), facecolor="white")
    fig.suptitle("Malware Dataset Analysis", fontsize=16, fontweight="bold", y=0.99)

    # layout: 3 rows — top pair, bottom pair, full-width histogram
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1, 1, 0.85],
        hspace=0.52, wspace=0.38,
    )

    # ── Donut ─────────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    _, _, autotexts = ax0.pie(
        label_counts, labels=label_counts.index, autopct="%1.1f%%",
        colors=["#E24B4A", "#1D9E75"], startangle=90,
        wedgeprops=dict(width=0.55), textprops=dict(fontsize=11),
        labeldistance=1.12, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")
    ax0.set_title("Malware vs benign distribution", fontsize=12, pad=10)

    # ── Top file types ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    max_ft = top_types.values.max()
    bars = ax1.barh(top_types.index, top_types.values, color="#378ADD", height=0.6)
    ax1.set_xlim(0, max_ft * 1.18)          # room for labels
    ax1.set_xlabel("File count", fontsize=10)
    ax1.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax1.invert_yaxis()
    ax1.set_title("Top 10 file types", fontsize=12)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="both", labelsize=10)
    _bar_labels(ax1, bars, orient="h")

    # ── Top malware families ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    max_fam = top_families.values.max()
    bars2 = ax2.barh(top_families.index, top_families.values, color="#7F77DD", height=0.6)
    ax2.set_xlim(0, max_fam * 1.18)
    ax2.set_xlabel("Sample count", fontsize=10)
    ax2.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax2.set_title("Top 10 malware families", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="both", labelsize=10)
    _bar_labels(ax2, bars2, orient="h")

    # ── Risk tier summary ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    x_pos = range(len(TIER_ORDER))
    bars3 = ax3.bar(x_pos, tier_counts.values, color=TIER_COLORS, width=0.6)
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels(
        ["Clean\n(0%)", "Very low\n(1–10%)", "Low\n(11–30%)",
         "Moderate\n(31–60%)", "High\n(61–80%)", "Critical\n(81–100%)"],
        fontsize=9,
    )
    ax3.set_ylim(0, tier_counts.values.max() * 1.15)
    ax3.set_ylabel("File count", fontsize=10)
    ax3.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax3.set_title("Detection risk tier distribution\n(% of scanners flagging file)", fontsize=12)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(axis="y", labelsize=10)
    _bar_labels(ax3, bars3, orient="v")

    # tier colour legend for histogram
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_COLORS[i], label=TIER_ORDER[i]) for i in range(len(TIER_ORDER))
    ]

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")

# ── 3. Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "/home/marcelino/Desktop/Data/Dataset/Combined-DB"
    agg = load_data(DATA_DIR)
    plot_dashboard(agg)