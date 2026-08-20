"""
KDE baseline model:
For each month in the test set, fit a KDE to the crime points from the
previous 12 months, predict hotspot areas, and evaluate predictive
accuracy using PAI.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import gaussian_kde
from shapely.geometry import box
from config import (PROC_DIR, RESULT_DIR, CRS_BNG,
                    KDE_BANDWIDTH, KDE_GRID_SIZE, HOTSPOT_PCT,
                    TRAIN_END, VAL_END)

# ═══════════════════════════════════════════════════════════════════
# KDE core: evaluate density at LSOA centroids
# ═══════════════════════════════════════════════════════════════════

def _kde_at_centroids(train_arr: np.ndarray,
                      eval_arr:  np.ndarray) -> np.ndarray:
    """
    Compute KDE density values at each point in eval_arr (Scott adaptive bandwidth).

    train_arr: (N, 2)  weighted training points [BNG_E, BNG_N]
    eval_arr:  (M, 2)  evaluation points (LSOA centroids)
    Returns:   (M,)    KDE density values
    """
    if len(train_arr) < 5:
        return np.zeros(len(eval_arr))

    # Column‑wise standardisation (ensures E and N are equally weighted;
    # though both BNG axes are metric, this is harmless).
    col_std = train_arr.std(axis=0)
    col_std[col_std == 0] = 1.0

    kde = gaussian_kde(
        (train_arr / col_std).T,
        bw_method="scott",   # adaptive bandwidth; replaces the previous manual setting
    )
    return kde((eval_arr / col_std).T)

# ═══════════════════════════════════════════════════════════════════
# Main function
# ═══════════════════════════════════════════════════════════════════

def run_kde(panel: pd.DataFrame,
            lsoa:  gpd.GeoDataFrame,
            count_col: str = "crime_count") -> pd.DataFrame:
    """
    KDE baseline model: for each test month, fit a KDE to the crime
    distribution from the previous 12 months, predict hotspots, and
    compute PAI (Predictive Accuracy Index).

    Fix note:
    Original crash point (kde_baseline.py:80):
        if geom.contains(gpd.points_from_xy([px], [py]).values[0]):
    Now we no longer call gpd.points_from_xy() at all; instead we
    directly read the BNG_E/BNG_N numeric coordinates from the LSOA data.

    Parameters
    ----------
    panel : pd.DataFrame
        Pre‑processed panel dataset
    lsoa : gpd.GeoDataFrame
        LSOA boundary data
    count_col : str
        Crime count column name, e.g. 'crime_count', 'n_theft_from_person', etc.

    Returns
    -------
    pd.DataFrame
        DataFrame containing PAI evaluation results

    """

    # Check that count_col exists in the panel
    if count_col not in panel.columns:
        raise ValueError(f"count_col '{count_col}' not found in panel data.")

    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"])

    # ── Pre‑processing: build LSOA centroid coordinate table ──────────────
    # Logs confirm BGC_V5 fields include BNG_E / BNG_N, so use them directly.
    lsoa_cols = list(lsoa.columns)

    if "BNG_E" in lsoa_cols and "BNG_N" in lsoa_cols:
        centroid_df = (
            lsoa[["LSOA21CD", "BNG_E", "BNG_N", "area_m2"]]
            .rename(columns={
                "LSOA21CD": "lsoa_code",
                "BNG_E":    "cx",
                "BNG_N":    "cy",
            })
            .copy()
            .reset_index(drop=True)
        )
        print("  [KDE] Using BNG_E/BNG_N as LSOA centroid coordinates")
    else:
        # Fallback: compute centroids from geometry (if BNG_E/BNG_N are unexpectedly absent)
        print("  [KDE] BNG_E/BNG_N not found; computing centroids from geometry...")
        centroid_df = (
            lsoa[["LSOA21CD", "area_m2", "geometry"]]
            .copy()
            .assign(
                cx=lambda d: d.geometry.centroid.x,
                cy=lambda d: d.geometry.centroid.y,
            )
            [["LSOA21CD", "cx", "cy", "area_m2"]]
            .rename(columns={"LSOA21CD": "lsoa_code"})
            .reset_index(drop=True)
        )

    total_area = centroid_df["area_m2"].sum()
    eval_arr   = centroid_df[["cx", "cy"]].values  # (M, 2)

    # ── Test months ─────────────────────────────────────────────
    test_months = (
        panel[panel["month"] > pd.to_datetime(VAL_END)]
        ["month"].sort_values().unique()
    )
    print(f"  [KDE] {len(test_months)} test months, "
          f"HOTSPOT_PCT={HOTSPOT_PCT:.0%}")

    results = []

    for month in test_months:
        ts        = pd.Timestamp(month)
        month_str = ts.strftime("%Y-%m")

        # ── Training window: previous 12 months ─────────────────────────
        window_start = ts - pd.DateOffset(months=12)
        train  = panel[(panel["month"] >= window_start) &
                       (panel["month"] <  ts)]
        actual = panel[panel["month"] == ts][
            ["lsoa_code", count_col]
        ].copy()

        actual_total = int(actual[count_col].sum())
        if actual_total == 0:
            print(f"  {month_str}: No actual crimes ({count_col}), skipping")
            continue

        # ── Build weighted training points (centroid × crime_count) ────────
        # Aggregate crime_count per LSOA over the training window, then merge
        # with centroid coordinates. Replicate centroid points proportionally
        # to their count to form the KDE input. A cap of 50 repetitions is
        # applied to prevent memory overflow (relative weights are preserved,
        # so the KDE shape is unaffected).
        train_agg = (
            train.groupby("lsoa_code")[count_col]
            .sum()
            .reset_index()
            .rename(columns={count_col: "cnt"})
        )
        train_agg = train_agg[train_agg["cnt"] > 0]
        train_agg = train_agg.merge(
            centroid_df[["lsoa_code", "cx", "cy"]],
            on="lsoa_code", how="inner",
        )

        if len(train_agg) < 5:
            print(f"  {month_str}: Fewer than 5 valid training LSOAs, skipping")
            continue

        max_count = train_agg["cnt"].max()
        scale     = max(1, int(max_count // 50))   # scaling factor
        repeats   = (
            (train_agg["cnt"] / scale)
            .clip(lower=1)
            .astype(int)
        )

        xs        = np.repeat(train_agg["cx"].values, repeats)
        ys        = np.repeat(train_agg["cy"].values, repeats)
        train_arr = np.column_stack([xs, ys])

        # ── KDE scoring ─────────────────────────────────────────
        scores = _kde_at_centroids(train_arr, eval_arr)

        # ── Hotspot identification (area accumulation method) ────────────
        # Sort LSOAs by descending KDE density, and define the top area
        # (≤ HOTSPOT_PCT × total_area) as predicted hotspot, which is
        # consistent with the standard PAI definition in criminology literature.
        scored            = centroid_df.copy()
        scored["score"]   = scores
        scored            = scored.sort_values("score", ascending=False)
        scored["cum_area_pct"] = (
            scored["area_m2"].cumsum() / total_area
        )
        scored["predicted_hotspot"] = (
            scored["cum_area_pct"] <= HOTSPOT_PCT
        )

        # Ensure at least one hotspot (edge‑case protection)
        if not scored["predicted_hotspot"].any():
            scored.iloc[0, scored.columns.get_loc("predicted_hotspot")] = True

        hotspot_set  = set(
            scored.loc[scored["predicted_hotspot"], "lsoa_code"]
        )
        hotspot_area = (
            scored.loc[scored["predicted_hotspot"], "area_m2"].sum()
        )

        # ── PAI computation ─────────────────────────────────────────
        captured = int(
            actual[actual["lsoa_code"].isin(hotspot_set)]
            [count_col].sum()
        )
        area_pct     = hotspot_area / total_area if total_area > 0 else 0.0
        capture_rate = captured / actual_total   if actual_total > 0 else 0.0
        pai          = capture_rate / area_pct   if area_pct     > 0 else 0.0

        results.append({
            "month":           ts,
            "model":           "KDE",
            "pai":             pai,
            "capture_rate":    capture_rate,
            "area_pct":        area_pct,
            "crimes_captured": captured,
            "crimes_total":    actual_total,
        })
        print(f"  KDE {month_str}: "
              f"PAI={pai:.3f}  "
              f"capture={capture_rate:.1%}  "
              f"area={area_pct:.1%}")

    df = pd.DataFrame(results)
    if not df.empty:

        # Generate different result filenames according to the count column
        suffix = count_col.replace("n_", "").replace("crime_count", "all")
        fname = f"kde_results_{suffix}.parquet" if suffix != "all" else "kde_results.parquet"

        df.to_parquet(RESULT_DIR / "kde_results.parquet", index=False)
        print(f"\n[KDE] Complete — mean PAI = {df['pai'].mean():.3f}  "
              f"(min={df['pai'].min():.3f}, max={df['pai'].max():.3f})")
    else:
        print("[KDE] No valid results; please check test month range and data integrity.")

    return df