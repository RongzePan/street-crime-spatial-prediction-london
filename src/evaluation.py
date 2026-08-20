"""
Aggregate PAI results from KDE and LSTM, generating comparison plots and
tables suitable for the dissertation.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from config import RESULT_DIR, FIG_DIR


def compare_and_plot(kde_results: pd.DataFrame,
                     lstm_results: pd.DataFrame):
    combined = pd.concat([kde_results, lstm_results], ignore_index=True)
    combined["month"] = pd.to_datetime(combined["month"])
    combined.to_parquet(RESULT_DIR / "pai_results.parquet", index=False)

    # ── Summary statistics table ──────────────────────────────────────
    summary = (combined.groupby("model")["pai"]
               .agg(["mean", "std", "min", "max"])
               .round(3))
    summary.columns = ["Mean PAI", "Std PAI", "Min PAI", "Max PAI"]
    summary.to_csv(RESULT_DIR / "pai_summary.csv")
    print("\n── PAI Summary ──")
    print(summary.to_string())

    # ── Monthly PAI time‑series comparison ─────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    for model, grp in combined.groupby("model"):
        grp = grp.sort_values("month")
        color = "#1D9E75" if model == "LSTM" else "#888780"
        ax.plot(grp["month"], grp["pai"], label=model,
                color=color, linewidth=2, marker="o", markersize=3)

    ax.axhline(1.0, color="#D85A30", linestyle="--",
               linewidth=0.8, label="PAI = 1 (random)")
    ax.set_title("Predictive Accuracy Index (PAI) — KDE vs LSTM\n"
                 f"Test period: 2025-07 to 2026-04 (top 10% hotspot)")
    ax.set_xlabel("Month")
    ax.set_ylabel("PAI")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pai_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pai_comparison.png")

    # ── Box plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [combined[combined["model"] == m]["pai"].values
            for m in ["KDE", "LSTM"]]
    bp = ax.boxplot(data, label=["KDE baseline", "LSTM"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#E1F5EE", color="#0F6E56"),
                    medianprops=dict(color="#D85A30", linewidth=2))
    ax.axhline(1.0, color="#888780", linestyle="--", linewidth=0.8)
    ax.set_title("PAI distribution — KDE vs LSTM")
    ax.set_ylabel("PAI")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pai_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pai_boxplot.png")

    return summary