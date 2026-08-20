"""
evaluate_by_crime_type.py

Meets supervisor feedback item 1: run KDE + LSTM separately for each of the four
street crime types, generating a directly comparable crime‑type‑specific PAI
results table.

Usage:
    python -m src.evaluate_by_crime_type

Outputs:
    outputs/results/pai_by_crime_type.csv
    outputs/figures/pai_by_crime_type_comparison.png
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import STREET_CRIME_TYPES, PROC_DIR, RESULT_DIR, FIG_DIR
from src.kde_baseline import run_kde
from src.lstm_model    import run_lstm


def run_all_crime_types() -> pd.DataFrame:
    panel = pd.read_parquet(PROC_DIR / "panel.parquet")
    panel["month"] = pd.to_datetime(panel["month"])

    all_results = []

    # ── All crimes (baseline, already existing) ───────────────────────────────────
    print("=" * 55)
    print("[All Crime Types] KDE + LSTM")
    kde_all  = run_kde(panel, count_col="crime_count")
    lstm_all = run_lstm(panel, target_col="crime_density",
                        count_col="crime_count")
    kde_all["crime_type"]  = "all_crime"
    lstm_all["crime_type"] = "all_crime"
    all_results += [kde_all, lstm_all]

    # ── Run per crime type ──────────────────────────────────────────────
    for ct_key, ct_cfg in STREET_CRIME_TYPES.items():
        n_col       = f"n_{ct_key}"
        density_col = f"density_{ct_key}"

        if n_col not in panel.columns or panel[n_col].sum() == 0:
            print(f"\nSkipping [{ct_key}]: data unavailable (column missing or all zero)")
            continue

        print("\n" + "=" * 55)
        print(f"[{ct_cfg['display_name']}] KDE + LSTM")

        kde_ct = run_kde(panel, count_col=n_col)
        kde_ct["crime_type"] = ct_key
        kde_ct["model"] = "KDE"

        lstm_ct = run_lstm(panel, target_col=density_col,
                           count_col=n_col)
        lstm_ct["crime_type"] = ct_key
        lstm_ct["model"] = "LSTM"

        all_results += [kde_ct, lstm_ct]

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_parquet(RESULT_DIR / "pai_by_crime_type.parquet",
                        index=False)

    summary = (combined.groupby(["crime_type", "model"])["pai"]
               .agg(["mean", "std", "min", "max"]).round(3))
    summary.to_csv(RESULT_DIR / "pai_by_crime_type.csv")
    print("\n" + "=" * 55)
    print("PAI Summary by Crime Type")
    print(summary.to_string())

    _plot_comparison(summary.reset_index())
    return combined


def _plot_comparison(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')

    types  = summary["crime_type"].unique()
    x      = range(len(types))
    width  = 0.35

    kde_means  = [summary[(summary.crime_type == t) &
                          (summary.model == "KDE")]["mean"].values[0]
                 if len(summary[(summary.crime_type == t) &
                                (summary.model == "KDE")]) else 0
                 for t in types]
    lstm_means = [summary[(summary.crime_type == t) &
                          (summary.model == "LSTM")]["mean"].values[0]
                 if len(summary[(summary.crime_type == t) &
                                (summary.model == "LSTM")]) else 0
                 for t in types]

    ax.bar([i - width/2 for i in x], kde_means, width,
           label="KDE", color="#95A5A6")
    ax.bar([i + width/2 for i in x], lstm_means, width,
           label="LSTM", color="#1D9E75")

    ax.axhline(1.0, color="#D85A30", linestyle="--", linewidth=0.8,
              label="PAI = 1 (random)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(types, rotation=20, ha='right')
    ax.set_ylabel("Mean PAI (test period)")
    ax.set_title("Predictive Accuracy Index by Crime Type — KDE vs LSTM",
                fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "pai_by_crime_type_comparison.png",
               dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print("Saved: pai_by_crime_type_comparison.png")


if __name__ == "__main__":
    run_all_crime_types()