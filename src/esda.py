"""
Exploratory Spatial Data Analysis:
1. Global temporal trend plots
2. Monthly Moran's I (spatial autocorrelation)
3. Hotspot distribution maps

esda.py  —— Professional cartography + street crime type‑focused analysis

Key modifications:
1. Added professional cartographic helper functions: scale bar, north arrow, map credits
2. plot_hotspot_map(): supports specific crime types and professional map elements
3. plot_crime_type_comparison(): 2×2 comparison of the four crime types
4. plot_city_comparison(): optional London vs Manchester comparison
5. compute_morans_series(): supports per‑crime‑type calculation
"""

import calendar
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
import esda
from esda.moran import Moran_Local
import libpysal.weights as lps
from config import (PROC_DIR, FIG_DIR, STREET_CRIME_TYPES, CITIES, ACTIVE_CITY)
import textwrap
import logging
from shapely.geometry import box as shapely_box

logger = logging.getLogger(__name__)

METRIC_COL   = "crime_density"
METRIC_LABEL = "Crime Density (crimes per km²)"


# ══════════════════════════════════════════════════════════════════
# Professional cartographic helper functions
# ══════════════════════════════════════════════════════════════════

def _add_scale_bar(ax,
                   length_km: int   = 10,
                   loc:       str   = 'lower left',
                   fontsize:  int   = 7) -> None:
    """
    Add a standard map scale bar (alternating black‑and‑white segments).
    Uses BNG data coordinates (metres) and must be called after the
    main plot has been drawn.

    Parameters
    ----------
    ax        : matplotlib Axes
    length_km : scale bar length (kilometres)
    loc       : position string, e.g. 'lower left' / 'lower right'
    fontsize  : label font size
    """
    length_m = length_km * 1000
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr  = x1 - x0, y1 - y0

    pad_x  = xr * 0.05
    pad_y  = yr * 0.05
    bar_h  = yr * 0.009          # bar height ≈ 0.9% of vertical range

    bx = x0 + pad_x if 'left' in loc else x1 - pad_x - length_m
    by = y0 + pad_y if 'lower' in loc else y1 - pad_y - bar_h * 6

    # White background
    bg = FancyBboxPatch(
        (bx - length_m * 0.06, by - bar_h * 0.6),
        length_m * 1.12, bar_h * 4.8,
        boxstyle  = "round,pad=0",
        facecolor = 'white', edgecolor = 'none',
        alpha     = 0.85, zorder = 9,
        transform = ax.transData, clip_on = False,
    )
    ax.add_patch(bg)

    # Alternating black‑and‑white segments
    half_m = length_m / 2
    for xi, fc in [(bx, 'black'), (bx + half_m, 'white')]:
        seg = Rectangle(
            (xi, by), half_m, bar_h,
            facecolor = fc, edgecolor = 'black',
            linewidth = 0.7, zorder = 10,
            transform = ax.transData, clip_on = False,
        )
        ax.add_patch(seg)

    # End tick marks
    for tx in [bx, bx + half_m, bx + length_m]:
        ax.plot([tx, tx], [by, by + bar_h * 1.9], 'k-',
                linewidth = 1.0, zorder = 11,
                transform = ax.transData, clip_on = False)

    # Labels (0, midpoint, end)
    half_label = (f'{length_km // 2}'
                  if length_km % 2 == 0 else f'{length_km / 2:.1f}')
    for label, tx in [
        ('0',               bx),
        (half_label,        bx + half_m),
        (f'{length_km} km', bx + length_m),
    ]:
        ax.text(tx, by + bar_h * 2.4, label,
                ha = 'center', va = 'bottom',
                fontsize = fontsize, color = 'black',
                zorder = 12,
                transform = ax.transData, clip_on = False)


def _add_north_arrow(ax,
                     x:    float = 0.955,
                     y:    float = 0.09,
                     size: float = 0.065) -> None:
    """
    Add a north arrow (using axis‑relative coordinates, suitable for any
    data extent).

    Parameters
    ----------
    ax   : matplotlib Axes
    x    : horizontal centre (axis proportion, 0–1)
    y    : arrow base (axis proportion, 0–1)
    size : arrow length (axis proportion units)
    """
    # Filled triangular arrow
    ax.annotate(
        '',
        xy         = (x, y + size),
        xytext     = (x, y),
        xycoords   = 'axes fraction',
        textcoords = 'axes fraction',
        arrowprops = dict(
            arrowstyle     = '-|>',
            color          = 'black',
            lw             = 2.0,
            mutation_scale = 16,
        ),
        annotation_clip = False,
        zorder = 15,
    )
    # 'N' label
    ax.text(
        x, y + size + 0.028, 'N',
        transform  = ax.transAxes,
        ha         = 'center', va = 'bottom',
        fontsize   = 9, fontweight = 'bold', color = 'black',
        zorder     = 15,
        # annotation_clip = False,
    )


def _add_map_credits(ax,
                     data_period: str = 'Jun 2023 – May 2026',
                     extra_note:  str = '') -> None:
    """
    Add standard data source attribution at the lower‑left corner of the map.

    Parameters
    ----------
    ax          : matplotlib Axes
    data_period : crime data time window
    extra_note  : additional note (e.g., crime type caveat)
    """
    lines = [
        f'Crime data: data.police.uk  |  {data_period}  |  Open Government Licence v3.0',
        'Boundaries: ONS Open Geography Portal — LSOA Dec 2021 BGC V5  |  OGL v3.0',
        'Cartography: UCL Dept. of Security & Crime Science  |  July 2026'
        + (f'  |  {extra_note}' if extra_note else ''),
    ]
    ax.text(
        0.01, 0.01, '\n'.join(lines),
        transform   = ax.transAxes,
        fontsize    = 5.2, color = '#444444',
        va = 'bottom', ha = 'left', linespacing = 1.4, zorder = 15,
        bbox = dict(
            boxstyle  = 'round,pad=0.25',
            facecolor = 'white', alpha = 0.82, linewidth = 0,
        ),
    )


# ══════════════════════════════════════════════════════════════════
# Temporal trend plots
# ══════════════════════════════════════════════════════════════════

def plot_temporal_trends(panel: pd.DataFrame,
                         city_name: str = "Greater London") -> None:
    """Monthly temporal trends (total crime count + crime density)."""
    monthly = (panel.groupby("month")[["crime_count", METRIC_COL]]
               .mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f"Monthly Crime Trends  |  {city_name}  |  Jun 2023 – May 2026",
        fontsize=12, fontweight='bold', y=1.02,
    )

    axes[0].plot(monthly["month"], monthly["crime_count"],
                 color="#1D9E75", linewidth=1.5)
    axes[0].set_title("Average Monthly Crime Count per LSOA",
                       fontsize=10, fontweight='bold', loc='left')
    axes[0].set_ylabel("Crime count", fontsize=9)
    axes[0].fill_between(monthly["month"], monthly["crime_count"],
                         alpha=0.12, color="#1D9E75")
    axes[1].plot(monthly["month"], monthly[METRIC_COL],
                 color="#D85A30", linewidth=1.5)
    axes[1].set_title(f"Average Monthly {METRIC_LABEL} per LSOA",
                       fontsize=10, fontweight='bold', loc='left')
    axes[1].set_ylabel(METRIC_LABEL, fontsize=9)
    axes[0].fill_between(monthly["month"], monthly["crime_count"],
                         alpha=0.12, color="#1D9E75")
    axes[1].plot(monthly["month"], monthly[METRIC_COL],
                 color="#D85A30", linewidth=1.8)
    axes[1].set_title(f"Average Monthly {METRIC_LABEL} per LSOA",
                       fontsize=10, fontweight='bold', loc='left')
    axes[1].set_ylabel(METRIC_LABEL, fontsize=9)
    axes[1].fill_between(monthly["month"], monthly[METRIC_COL],
                         alpha=0.12, color="#D85A30")

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator(interval=3)
    for ax in axes:
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_loc)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_xlabel("Month", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Figure‑level source credit
    fig.text(0.01, -0.02,
             "Data: data.police.uk (OGL v3.0)  |  "
             "UCL Dept. of Security & Crime Science, 2026",
             fontsize=6, color='#888888')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "temporal_trends.png", dpi=150,
                bbox_inches="tight", facecolor='white')
    plt.close()
    print("Saved: temporal_trends.png")


# ══════════════════════════════════════════════════════════════════
# Global Moran's I
# ══════════════════════════════════════════════════════════════════

def compute_morans_series(panel: pd.DataFrame,
                          lsoa:  gpd.GeoDataFrame,
                          crime_type: str = None) -> pd.DataFrame:
    """
    Compute global Moran's I for each month.

    Parameters
    ----------
    panel      : pre‑processed panel dataset
    lsoa       : LSOA boundary GeoDataFrame
    crime_type : if None, uses crime_density; otherwise uses density_{crime_type}
    """
    col = f"density_{crime_type}" if crime_type else METRIC_COL
    col_label = (STREET_CRIME_TYPES[crime_type]["short_name"]
                 if crime_type else "All Crime Density")

    lsoa_sorted = lsoa.set_index("LSOA21CD").sort_index()

    # ── Suppress repeated island warnings from libpysal, print only once ──
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*island.*",
            category=UserWarning
        )

    w = lps.Queen.from_dataframe(lsoa_sorted, use_index=True)
    w.transform = "r"

    # Get island LSOA list and print once (not per month)
    island_ids = [nid for nid, nbrs in w.neighbors.items() if len(nbrs) == 0]
    if island_ids:
        print(f"  ℹ️  {len(island_ids)} 个岛状 LSOA（无邻居）: "
              f"{', '.join(island_ids[:5])}"
              f"{' ...' if len(island_ids) > 5 else ''}")

    results = []
    for month, grp in panel.groupby("month"):
        grp_s = (grp.set_index("lsoa_code")[col]
                 .reindex(lsoa_sorted.index)
                 .fillna(0))

        if grp_s.std() < 1e-10:
            results.append({"month": pd.Timestamp(month),
                             "morans_i": np.nan, "p_value": np.nan})
            continue

        try:
            mi = esda.Moran(grp_s.values, w)
            results.append({"month": pd.Timestamp(month),
                             "morans_i": float(mi.I),
                             "p_value":  float(mi.p_sim)})
        except Exception:
            results.append({"month": pd.Timestamp(month),
                             "morans_i": np.nan, "p_value": np.nan})

    df = pd.DataFrame(results)
    df.to_parquet(PROC_DIR / f"morans_i_{crime_type or 'all'}.parquet",
                  index=False)

    valid = df.dropna(subset=["morans_i"])
    print(f"  Moran's I [{col_label}]: {len(valid)}/36 valid months  |  "
          f"range {valid['morans_i'].min():.3f} – {valid['morans_i'].max():.3f}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor('white')

    if len(valid) > 0:
        colors = ["#C0392B" if p < 0.05 else "#95A5A6"
                  for p in valid["p_value"]]
        ax.bar(valid["month"], valid["morans_i"],
               color=colors, width=22, edgecolor='none')

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_title(
        f"Global Moran's I  |  Monthly {col_label} per km²  |  "
        f"Greater London  |  Jun 2023 – May 2026\n"
        f"(red bars = p < 0.05, 999 permutations; Queen contiguity, n = 6,195 LSOAs)",
        fontsize=9, loc='left', pad=8,
    )
    ax.set_xlabel("Month", fontsize=9)
    ax.set_ylabel("Moran's I", fontsize=9)

    ax.set_xlim(df["month"].min() - pd.Timedelta(days=22),
                df["month"].max() + pd.Timedelta(days=22))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C0392B', label='Significant (p < 0.05)'),
        Patch(facecolor='#95A5A6', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=7, framealpha=0.85)

    # Source credit
    fig.text(0.01, -0.06,
             "Data: data.police.uk (OGL v3.0)  |  "
             "Boundaries: ONS BGC V5 (OGL v3.0)  |  "
             "UCL Dept. of Security & Crime Science, 2026",
             fontsize=6, color='#888888')

    plt.tight_layout()
    out_name = f"morans_i_series_{crime_type or 'all'}.png"
    plt.savefig(FIG_DIR / out_name, dpi=150,
                bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Saved: {out_name}")
    return df

def _add_borough_overlay(ax, borough_gdf: gpd.GeoDataFrame) -> None:
    """
    Supervisor feedback item 5: delineate borough boundaries on all hotspot maps
    with a bold black line, clearly demarcating the study area boundary.
    """
    if borough_gdf is None or borough_gdf.empty:
        return
    borough_gdf.boundary.plot(
        ax=ax, color="black", linewidth=1.3,
        linestyle="-", alpha=0.75, zorder=8,
    )

def _fit_label_to_polygon(ax, renderer, name: str, point,
                          polygon, max_fontsize: int = 8,
                          min_fontsize: int = 4,
                          wrap_widths: tuple = (40, 32, 26, 20, 16, 12, 9, 7)):
    """
    Generate an adaptive label for a single borough: starting from a large
    font and wide wrap width, progressively reduce both until the rendered
    text block (converted back to data coordinates) is fully contained within
    the borough polygon.

    Core techniques:
    - polygon.contains(text_box): true geometric containment, not just
      bounding‑box comparison. For concave polygons (e.g., boroughs extending
      along the Thames), bounding‑box checks would incorrectly accept labels
      that actually spill outside the boundary, whereas contains() correctly
      catches these cases.
    - get_window_extent(renderer): uses the actual renderer to obtain
      pixel‑precise text bounding boxes, rather than heuristic character‑width
      estimation, ensuring accuracy across different font‑rendering backends.

    Returns
    -------
    matplotlib.text.Text or None
        If any combination fits perfectly, returns that Text object.
        If none fits (e.g., the extremely small City of London), returns a
        fallback version with the smallest font and tightest wrap, which is
        still preferable to showing no label at all.
    """
    last_txt = None

    for fontsize in range(max_fontsize, min_fontsize - 1, -1):
        for wrap_w in wrap_widths:
            wrapped = textwrap.fill(name, width=wrap_w)

            txt = ax.text(
                point.x, point.y, wrapped,
                fontsize=fontsize, ha="center", va="center",
                color="#1A1A1A", fontweight="bold", zorder=10,
                linespacing=0.95,
                path_effects=[pe.withStroke(linewidth=2.2,
                                            foreground="white")],
            )

            # Pixel coordinates → map data coordinates (metres, BNG)
            bbox_px = txt.get_window_extent(renderer=renderer)
            x0, y0 = ax.transData.inverted().transform(
                (bbox_px.x0, bbox_px.y0))
            x1, y1 = ax.transData.inverted().transform(
                (bbox_px.x1, bbox_px.y1))
            text_box = shapely_box(min(x0, x1), min(y0, y1),
                                   max(x0, x1), max(y0, y1))

            if polygon.contains(text_box):
                if last_txt is not None:
                    last_txt.remove()
                return txt          # perfect fit, accept this version

            if last_txt is not None:
                last_txt.remove()   # remove previous failed attempt
            last_txt = txt

    # No combination fits perfectly (e.g., City of London); retain the last
    # attempt (smallest font + tightest wrap) as a fallback.
    return last_txt

def _add_borough_labels(ax, fig, borough_gdf: gpd.GeoDataFrame,
                        max_fontsize: int = 8,
                        min_fontsize: int = 4) -> None:
    """
    Add [NAME] labels for all boroughs on the map, satisfying the following:
      1. White halo: ensures text remains legible against varied background
         shades.
      2. Automatic line‑wrapping: long names (e.g., "Kensington and Chelsea",
         "Hammersmith and Fulham") are wrapped to fit the polygon boundary.
      3. Use representative_point() rather than centroid, ensuring the label
         point always lies inside the polygon (critical for narrow / concave
         boroughs).

    Parameters
    ----------
    max_fontsize / min_fontsize : adjust according to overall map size.
        For single full‑page maps (e.g., plot_hotspot_map) use 8/4;
        for multi‑panel comparison plots (e.g., plot_crime_type_comparison)
        use 6/3.

    Debug note: the print at the function entry is unconditional (executed
    before any early return), so if this function is called, the log will
    always show "[_add_borough_labels] function entry". If that line is
    missing, the caller is not executing, not a failure inside the function.
    """
    print(f"--- _add_borough_labels called, borough_gdf type: {type(borough_gdf)}")
    print(f"    [_add_borough_labels] function entry — "
          f"borough_gdf rows: {0 if borough_gdf is None else len(borough_gdf)}")

    if borough_gdf is None:
        print("   borough_gdf is None")
        return
    if borough_gdf.empty:
        print("   borough_gdf is empty")
        return
    print(f"   borough_gdf shape: {borough_gdf.shape}")
    print(f"   columns: {list(borough_gdf.columns)}")

    # if borough_gdf is None or borough_gdf.empty:
    #     return

    name_col = next(
        (c for c in borough_gdf.columns
         if c.upper() in ("LAD23NM", "LAD22NM", "LAD21NM", "NAME")),
        None,
    )
    if name_col is None:
        logger.warning(f"  Borough name column not found; skipping labels. "
                    f"Available columns: {list(borough_gdf.columns)}")
        print(f"    [_add_borough_labels] ⚠ Borough name column not found; skipping labels. "
              f"Available columns: {list(borough_gdf.columns)}")
        return

    # Ensure renderer is ready and DPI matches the final savefig,
    # avoiding inconsistencies between pixel measurements and output size.
    print(f"    [_add_borough_labels] using name column: '{name_col}', "
          f"generating labels for {len(borough_gdf)} boroughs...")

    fig.set_dpi(150)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    n_fitted, n_fallback = 0, 0

    for _, row in borough_gdf.iterrows():
        name    = row[name_col]
        polygon = row.geometry
        if polygon is None or polygon.is_empty:
            continue

        # representative_point() always lies inside the polygon;
        # more reliable than centroid for concave boroughs (e.g., those
        # extending along the river).
        point = polygon.representative_point()

        txt = _fit_label_to_polygon(
            ax, renderer, name, point, polygon,
            max_fontsize=max_fontsize, min_fontsize=min_fontsize,
        )

        if txt is not None:
            # Approximate count of perfect vs. fallback labels (only for
            # diagnostic logging; does not affect rendering).
            if txt.get_fontsize() == min_fontsize:
                n_fallback += 1
            else:
                n_fitted += 1

    logger.info(f"  Borough labels: {n_fitted} perfectly fitted, "
             f"{n_fallback} fallback (small areas, e.g., City of London)")
    print(f"    [_add_borough_labels] complete: {n_fitted} perfectly fitted, "
          f"{n_fallback} fallback")

def plot_lisa_cluster_map(panel:      pd.DataFrame,
                          lsoa:       gpd.GeoDataFrame,
                          borough:    gpd.GeoDataFrame,
                          month:      str,
                          crime_type: str  = None,
                          city_name:  str  = "Greater London",
                          scale_km:   int  = 10,
                          out_dir:    Path = FIG_DIR) -> gpd.GeoDataFrame:
    """
    Generate a LISA (Local Indicators of Spatial Association; Anselin, 1995)
    cluster map.

    Supervisor feedback item 5: "use clustering to show where Moran's I is
    locally pronounced in Greater London".

    Global Moran's I only answers whether spatial clustering exists; LISA is
    its local decomposition, computing a local statistic for each LSOA. It
    pinpoints where clustering occurs and distinguishes four patterns:
      HH (High-High) — high‑crime LSOA surrounded by high‑crime neighbours
                       (hotspot core)
      LL (Low-Low)   — low‑crime LSOA surrounded by low‑crime neighbours
                       (cold spot)
      HL (High-Low)  — high‑crime LSOA surrounded by low‑crime neighbours
                       (spatial outlier)
      LH (Low-High)  — low‑crime LSOA surrounded by high‑crime neighbours
                       (spatial outlier)
    """
    col = f"density_{crime_type}" if crime_type else METRIC_COL
    col_label = (STREET_CRIME_TYPES[crime_type]["short_name"]
                 if crime_type else "All Street Crime")

    grp = panel[panel["month"] == pd.to_datetime(month)].copy()
    lsoa_sorted = lsoa.set_index("LSOA21CD").sort_index()

    y = (grp.set_index("lsoa_code")[col]
         .reindex(lsoa_sorted.index).fillna(0).values)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*island.*")
        w = lps.Queen.from_dataframe(lsoa_sorted, use_index=True)
    w.transform = "r"

    lisa = Moran_Local(y, w, permutations=999, seed=42)

    result = lsoa_sorted.reset_index().copy()
    result["local_i"]  = lisa.Is
    result["quadrant"] = lisa.q
    result["p_sim"]    = lisa.p_sim
    result["significant"] = result["p_sim"] < 0.05

    # Map numeric quadrant codes to labels
    quad_map = {1: "HH (Hotspot cluster)",
                2: "LH (Low surrounded by high)",
                3: "LL (Coldspot cluster)",
                4: "HL (High surrounded by low)"}
    result["cluster_type"] = np.where(
        result["significant"],
        result["quadrant"].map(quad_map),
        "Not significant",
    )

    color_map = {
        "HH (Hotspot cluster)":         "#C0392B",
        "LL (Coldspot cluster)":        "#2874A6",
        "HL (High surrounded by low)":  "#F5B041",
        "LH (Low surrounded by high)":  "#A9CCE3",
        "Not significant":              "#EAECEE",
    }

    fig, ax = plt.subplots(figsize=(11, 13))
    fig.patch.set_facecolor('white')
    fig.set_dpi(150)

    for label, color in color_map.items():
        subset = result[result["cluster_type"] == label]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="none",
                       linewidth=0, label=label, zorder=3)

    # ── Borough boundaries + name labels (key focus of this fix) ───
    _add_borough_overlay(ax, borough)

    month_dt  = pd.to_datetime(month)
    month_str = f"{calendar.month_name[month_dt.month]} {month_dt.year}"

    ax.set_title(
        f"Local Spatial Clustering (LISA)\n"
        f"{col_label}  ·  {city_name}  ·  {month_str}\n"
        f"(Local Moran's I; Queen contiguity; 999 permutations, p < 0.05)",
        fontsize=11, fontweight='bold', pad=10, loc='left',
    )

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=l, edgecolor='grey', linewidth=0.3)
                    for l, c in color_map.items()]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=7.5,
              framealpha=0.9, title="LISA Cluster Type",
              title_fontsize=7.5)

    n_hh = int((result["cluster_type"] == "HH (Hotspot cluster)").sum())
    n_ll = int((result["cluster_type"] == "LL (Coldspot cluster)").sum())
    ax.text(0.99, 0.01,
            f"Significant HH clusters: {n_hh} LSOAs ({n_hh/len(result)*100:.1f}%)\n"
            f"Significant LL clusters: {n_ll} LSOAs ({n_ll/len(result)*100:.1f}%)",
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, linewidth=0), zorder=12)

    _add_scale_bar(ax, length_km=scale_km, loc='lower right', fontsize=7)
    _add_north_arrow(ax, x=0.955, y=0.08, size=0.065)
    _add_map_credits(ax, extra_note="LISA: Anselin (1995)")

    ax.set_axis_off()
    plt.tight_layout(pad=0.4)

    fname = f"lisa_{month}_{crime_type or 'all_crime'}.png"
    plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight",
               facecolor='white')
    plt.close()
    print(f"  Saved: {fname}  |  HH clusters: {n_hh}  |  LL clusters: {n_ll}")
    return result

# ══════════════════════════════════════════════════════════════════
# Single‑month, single‑crime‑type hotspot map (professional cartography)
# ══════════════════════════════════════════════════════════════════

def plot_hotspot_map(panel:      pd.DataFrame,
                     lsoa:       gpd.GeoDataFrame,
                     month:      str,
                     crime_type: str   = None,
                     pct:        float = 0.10,
                     city_name:  str   = "Greater London",
                     scale_km:   int   = 10,
                     out_dir:    Path  = FIG_DIR,
                     borough:    gpd.GeoDataFrame = None) -> None:
    """
    Generate a professional crime hotspot map, including scale bar,
    north arrow, legend, and source attribution.

    Parameters
    ----------
    panel      : panel dataset
    lsoa       : LSOA boundaries
    month      : target month (e.g. '2023-12')
    crime_type : key in STREET_CRIME_TYPES; None = all crime
    pct        : hotspot area proportion (default 0.10 = top 10%)
    city_name  : city name for map title
    scale_km   : scale bar length (kilometres)
    out_dir    : output directory
    """
    # ── Select data column ─────────────────────────────────────────────────
    if crime_type and crime_type in STREET_CRIME_TYPES:
        ct_cfg    = STREET_CRIME_TYPES[crime_type]
        col       = f"density_{crime_type}"
        cmap      = ct_cfg["cmap"]
        color_hs  = ct_cfg["color"]
        title_sub = ct_cfg["display_name"]
        fname_tag = crime_type
        note      = ct_cfg.get("description", "")
    else:
        col       = "crime_density"
        cmap      = "YlOrRd"
        color_hs  = "#FF4500"
        title_sub = "All Street Crime"
        fname_tag = "all_crime"
        note      = "Aggregated across all crime categories from data.police.uk."

    grp = panel[panel["month"] == pd.to_datetime(month)].copy()
    merged = lsoa.merge(
        grp[["lsoa_code", col]],
        left_on="LSOA21CD", right_on="lsoa_code", how="left",
    )

    valid_vals = merged[col].dropna()
    n_valid    = len(valid_vals)
    if n_valid == 0:
        print(f"  ⚠ No data for {month} [{fname_tag}], skipping.")
        return
    print(f"  {month} [{fname_tag}]: "
          f"{n_valid}/{len(merged)} LSOAs with data")

    vmax_clip = float(valid_vals.quantile(0.98))
    threshold = float(valid_vals.quantile(1 - pct))
    print(f"  Colour scale: 0 – {vmax_clip:.1f} (98th pct) | "
          f"max = {valid_vals.max():.1f}")

    merged["is_hotspot"] = (
        merged[col].notna() & (merged[col] >= threshold)
    )
    n_hotspot = int(merged["is_hotspot"].sum())

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 13))
    fig.patch.set_facecolor('white')
    fig.set_dpi(150)

    # Base choropleth
    merged.plot(
        column   = col,
        ax       = ax,
        cmap     = cmap,
        vmin     = 0,
        vmax     = vmax_clip,
        legend   = True,
        legend_kwds = {
            "shrink"   : 0.55,
            "label"    : (f"Crime density (crimes per km²)\n"
                          f"[colour capped at 98th percentile = {vmax_clip:.0f}]"),
            "format"   : "%.0f",
            "pad"      : 0.01,
            "aspect"   : 28,
            "ticks"    : np.linspace(0, vmax_clip, 6),
        },
        missing_kwds = {"color": "#D5D8DC", "label": "No data"},
        linewidth    = 0.04,
        edgecolor    = "none",
    )

    # Hotspot boundaries
    hotspot_gdf = merged[merged["is_hotspot"]]
    if not hotspot_gdf.empty:
        hotspot_gdf.boundary.plot(
            ax        = ax,
            color     = color_hs,
            linewidth = 1.0,
            label     = (f"Hotspot boundary — top {int(pct*100)}% area "
                         f"(n = {n_hotspot} LSOAs)"),
            zorder    = 7,
        )

    # ── Add borough overlay and labels ──
    _add_borough_overlay(ax, borough)
    _add_borough_labels(ax, fig, borough, max_fontsize=8, min_fontsize=4)

    # ── Map Title ──────────────────────────────────────────────────
    month_dt  = pd.to_datetime(month)
    month_str = f"{calendar.month_name[month_dt.month]} {month_dt.year}"

    ax.set_title(
        f"{title_sub}\n"
        f"Crime Density Hotspot Map  ·  {city_name}  ·  {month_str}",
        fontsize   = 11,
        fontweight = 'bold',
        pad        = 10,
        loc        = 'left',
    )

    # ── Legend ────────────────────────────────────────────────────
    ax.legend(
        loc        = 'upper left',
        fontsize   = 7.5,
        framealpha = 0.88,
        handlelength = 1.4,
        borderpad    = 0.45,
        title        = "Map Legend",
        title_fontsize = 7.5,
    )

    # ── Professional cartographic elements ──────────────────────────
    _add_scale_bar(ax, length_km=scale_km, loc='lower right', fontsize=7)
    _add_north_arrow(ax, x=0.955, y=0.08, size=0.065)
    extra = (note[:70] + "…") if len(note) > 70 else note
    _add_map_credits(ax, extra_note=extra)

    ax.set_axis_off()
    plt.tight_layout(pad=0.4)

    out_path = out_dir / f"hotspot_{month}_{fname_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"  Saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════
# Four crime types 2×2 comparison
# ══════════════════════════════════════════════════════════════════

def plot_crime_type_comparison(panel:     pd.DataFrame,
                                lsoa:      gpd.GeoDataFrame,
                                month:     str,
                                pct:       float = 0.10,
                                city_name: str   = "Greater London",
                                scale_km:  int   = 10,
                                out_dir:   Path  = FIG_DIR,
                               available_cts: list = None,
                               borough: gpd.GeoDataFrame = None,
                               ) -> None:
    """
    Generate a 2×2 professional comparison of the four street crime types.

    Each subplot shows the spatial density of one crime type, with orange
    boundaries marking the top‑10% hotspot area. Colour scales are calibrated
    independently (each using its 98th percentile) to display spatial patterns
    rather than absolute crime volumes.
    """
    # crime_keys = list(STREET_CRIME_TYPES.keys())   # 4 types
    # ── Determine the crime types to plot ──────────────────────────────────────
    if available_cts is None:
        crime_keys = [
            k for k in STREET_CRIME_TYPES
            if f"density_{k}" in panel.columns
            and panel[f"density_{k}"].notna().any()
        ]
    else:
        crime_keys = [
            k for k in available_cts
            if f"density_{k}" in panel.columns
        ]

    n_types = len(crime_keys)
    if n_types == 0:
        print(f"  ⚠ {month}: no available crime types; skipping comparison plot.")
        return

    # ── 自适应布局 ─────────────────────────────────────────────────
    if n_types == 1:
        nrow, ncol = 1, 1
        fig_w, fig_h = 11, 13
    elif n_types == 2:
        nrow, ncol = 1, 2
        fig_w, fig_h = 22, 13
    elif n_types == 3:
        nrow, ncol = 1, 3
        fig_w, fig_h = 33, 13
    else:  # n_types == 4 (or more, take at most 4)
        crime_keys = crime_keys[:4]
        n_types    = 4
        nrow, ncol = 2, 2
        fig_w, fig_h = 22, 26


    month_dt  = pd.to_datetime(month)
    month_str = f"{calendar.month_name[month_dt.month]} {month_dt.year}"

    # ── Title lists the included types ──────────────────────────────────
    type_labels = " · ".join(
        STREET_CRIME_TYPES[k]["short_name"] for k in crime_keys
    )

    # fig, axes = plt.subplots(2, 2, figsize=(20, 22))
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h),
                              squeeze=False)
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f"Street Crime Type Spatial Comparison\n"
        f"{city_name}  ·  {month_str}  ·  Top {int(pct*100)}% Hotspot Area",
        fontsize = 14 if n_types > 2 else 13,
        fontweight = 'bold',
        y          = 1.01,
    )

    grp = panel[panel["month"] == pd.to_datetime(month)].copy()
    axes_flat = axes.flat

    for idx, (ct_key, ax) in enumerate(zip(crime_keys, axes.flat)):
        ax     = axes_flat[idx]

        ct_cfg = STREET_CRIME_TYPES[ct_key]
        col    = f"density_{ct_key}"
        cmap   = ct_cfg["cmap"]

        merged = lsoa.merge(
            grp[["lsoa_code", col]],
            left_on="LSOA21CD", right_on="lsoa_code", how="left",
        )

        valid_vals = merged[col].dropna()
        if len(valid_vals) == 0:
            ax.text(0.5, 0.5,
                    f"No data available\n{ct_cfg['short_name']}",
                    ha='center', va='center', fontsize=12,
                    transform=ax.transAxes, color='#666666')
            ax.set_title(f"({chr(65+idx)})  {ct_cfg['display_name']}",
                         fontsize=10, fontweight='bold', loc='left')
            ax.set_axis_off()
            continue

        vmax_clip = float(valid_vals.quantile(0.98))
        threshold = float(valid_vals.quantile(1 - pct))
        merged["is_hotspot"] = (
            merged[col].notna() & (merged[col] >= threshold)
        )
        n_hs = int(merged["is_hotspot"].sum())

        merged.plot(
            column   = col,
            ax       = ax,
            cmap     = cmap,
            vmin     = 0,
            vmax     = vmax_clip,
            legend   = True,
            legend_kwds = {
                "shrink"   : 0.65,
                "label"    : f"Crimes per km²\n[98th pct = {vmax_clip:.0f}]",
                "format"   : "%.0f",
                "pad"      : 0.01,
                "aspect"   : 25,
            },
            missing_kwds = {"color": "#D5D8DC", "label": "No data"},
            linewidth    = 0.04,
            edgecolor    = "none",
        )

        hotspot_gdf = merged[merged["is_hotspot"]]
        if not hotspot_gdf.empty:
            hotspot_gdf.boundary.plot(
                ax        = ax,
                color     = ct_cfg["color"],
                linewidth = 0.9,
                label     = f"Hotspot (n={n_hs})",
                zorder    = 7,
            )

        # ── Add borough overlay and labels to each subplot ──
        _add_borough_overlay(ax, borough)
        _add_borough_labels(ax, fig, borough, max_fontsize=6, min_fontsize=3)

        # Subplot title (with letter label)
        panel_label = chr(65 + idx)   # A / B / C / D

        # 统计注释框
        total_crime = int(grp[f"n_{ct_key}"].sum())
        ax.text(0.99, 0.01,
                f"Total crimes: {total_crime:,}\n"
                f"Hotspot LSOAs: {n_hs} ({n_hs/len(merged)*100:.1f}%)",
                transform  = ax.transAxes,
                fontsize   = 6.5, color = '#333333',
                ha = 'right', va = 'bottom',
                bbox = dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.85, linewidth=0),
                zorder = 12)

        ax.set_title(
            f"({panel_label})  {ct_cfg['display_name']}",
            fontsize   = 10,
            fontweight = 'bold',
            pad        = 7,
            loc        = 'left',
        )

        ax.legend(loc='upper left', fontsize=6.5,
                  framealpha=0.85, borderpad=0.4)

        # Scale bar: only in the third panel (lower‑left, index 2)
        if idx == 0:
            _add_scale_bar(ax, length_km=scale_km,
                           loc='lower left', fontsize=6)

        # North arrow: only in the fourth panel (lower‑right, index 3)
        if idx == n_types - 1:
            _add_north_arrow(ax, x=0.955, y=0.07, size=0.062)

        ax.set_axis_off()

    # Hide any unused subplots
    for j in range(n_types, nrow * ncol):
        axes_flat[j].set_visible(False)

    # Figure‑level source credit
    fig.text(
        0.01, 0.002,
        "Crime data: data.police.uk  |  Jun 2023 – May 2026  |  "
        "Open Government Licence v3.0  |  "
        "Boundaries: ONS Open Geography Portal, LSOA Dec 2021 BGC V5 (OGL v3.0)  |  "
        "Cartography: UCL Dept. of Security & Crime Science, July 2026  |  "
        "Note: colour scales are independent per panel (each capped at 98th percentile) "
        "to show spatial patterns rather than absolute crime volume.",
        fontsize    = 5.5,
        color       = '#555555',
        va          = 'bottom',
    )

    plt.tight_layout(pad=0.8)
    out_path = out_dir / f"crime_type_comparison_{month}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Saved: {out_path.name}")


# ══════════════════════════════════════════════════════════════════
# City comparison (London vs Manchester)
# ══════════════════════════════════════════════════════════════════

def plot_city_comparison(panels:     dict,
                          lsoas:      dict,
                          month:      str,
                          crime_type: str   = "theft_from_person",
                          pct:        float = 0.10,
                          out_dir:    Path  = FIG_DIR) -> None:
    """
    Generate side‑by‑side hotspot maps for London and Manchester.

    Parameters
    ----------
    panels     : {"london": DataFrame, "manchester": DataFrame}
    lsoas      : {"london": GeoDataFrame, "manchester": GeoDataFrame}
    month      : target month
    crime_type : key in STREET_CRIME_TYPES
    pct        : hotspot area proportion
    """
    if not panels:
        print("  City comparison: no panel data provided; skipping.")
        return

    ct_cfg    = STREET_CRIME_TYPES[crime_type]
    col       = f"density_{crime_type}"
    cmap      = ct_cfg["cmap"]
    month_dt  = pd.to_datetime(month)
    month_str = f"{calendar.month_name[month_dt.month]} {month_dt.year}"

    city_keys  = list(panels.keys())
    n_cities   = len(city_keys)

    fig, axes = plt.subplots(1, n_cities, figsize=(13 * n_cities, 15))
    if n_cities == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f"City Comparison  |  {ct_cfg['display_name']}\n"
        f"Crime Density Hotspot Map  ·  {month_str}  ·  "
        f"Top {int(pct*100)}% Hotspot Area",
        fontsize=14, fontweight='bold', y=1.01,
    )

    from config import CITIES

    for idx, (city_key, ax) in enumerate(zip(city_keys, axes)):
        panel     = panels[city_key]
        lsoa_gdf  = lsoas[city_key]
        city_cfg  = CITIES.get(city_key, {})
        city_name = city_cfg.get("name", city_key.title())
        scale_km  = city_cfg.get("scale_bar_km", 10)

        grp    = panel[panel["month"] == pd.to_datetime(month)].copy()
        merged = lsoa_gdf.merge(
            grp[["lsoa_code", col]],
            left_on="LSOA21CD", right_on="lsoa_code", how="left",
        )

        valid_vals = merged[col].dropna()
        if len(valid_vals) == 0:
            ax.text(0.5, 0.5, f"No data: {city_name}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        vmax_clip = float(valid_vals.quantile(0.98))
        threshold = float(valid_vals.quantile(1 - pct))
        merged["is_hotspot"] = (
            merged[col].notna() & (merged[col] >= threshold)
        )
        n_hs = int(merged["is_hotspot"].sum())

        merged.plot(
            column=col, ax=ax, cmap=cmap, vmin=0, vmax=vmax_clip,
            legend=True,
            legend_kwds={
                "shrink"  : 0.60,
                "label"   : (f"Crime density (crimes per km²)\n"
                             f"[98th pct = {vmax_clip:.0f}]"),
                "format"  : "%.0f", "pad": 0.01, "aspect": 28,
            },
            missing_kwds={"color": "#D5D8DC"},
            linewidth=0.05, edgecolor="none",
        )

        hotspot_gdf = merged[merged["is_hotspot"]]
        if not hotspot_gdf.empty:
            hotspot_gdf.boundary.plot(
                ax=ax, color=ct_cfg["color"], linewidth=1.0,
                label=f"Hotspot boundary (n={n_hs} LSOAs)", zorder=7,
            )

        panel_label = chr(65 + idx)
        ax.set_title(
            f"({panel_label})  {city_name}\n"
            f"{ct_cfg['short_name']}  ·  Top {int(pct*100)}% Hotspot",
            fontsize=11, fontweight='bold', pad=8, loc='left',
        )
        ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85)
        _add_scale_bar(ax, length_km=scale_km, loc='lower right', fontsize=7)
        _add_north_arrow(ax, x=0.955, y=0.08, size=0.065)
        _add_map_credits(ax)
        ax.set_axis_off()

    plt.tight_layout(pad=0.8)
    city_tag = "_vs_".join(city_keys)
    out_path = out_dir / f"city_comparison_{crime_type}_{month}_{city_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Saved: {out_path.name}")

def plot_study_area_map(lsoa: gpd.GeoDataFrame,
                        borough: gpd.GeoDataFrame,
                        city_name: str = "Greater London",
                        scale_km: int = 10,
                        out_dir: Path = FIG_DIR) -> None:
    """
    Generate a clean study area base map showing LSOA boundaries and borough outlines,
    without crime density coloring. Reuses existing cartographic helper functions.

    Parameters
    ----------
    lsoa : gpd.GeoDataFrame
        LSOA boundaries (already loaded in run())
    borough : gpd.GeoDataFrame
        Borough boundaries (already loaded in run())
    city_name : str
        City name for title
    scale_km : int
        Scale bar length in kilometres
    out_dir : Path
        Output directory for the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    fig.set_dpi(150)

    # Plot LSOA boundaries (light grey, thin lines)
    lsoa.boundary.plot(ax=ax, linewidth=0.15, color='grey', alpha=0.6)

    # Plot borough boundaries (bold black lines)
    borough.boundary.plot(ax=ax, linewidth=1.6, color='black', alpha=0.9)

    # Add borough labels using representative points (avoid centroid for concave polygons)
    name_col = next(
        (c for c in borough.columns if c.upper() in ("LAD23NM", "LAD22NM", "LAD21NM", "NAME")),
        None
    )
    if name_col:
        for _, row in borough.iterrows():
            point = row.geometry.representative_point()
            ax.annotate(
                row[name_col],
                xy=(point.x, point.y),
                ha='center',
                va='center',
                fontsize=5,
                weight='bold',
                color='#1A1A1A',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')],
                zorder=10
            )

    # Set title
    ax.set_title(
        f"Study Area: {city_name}\n"
        f"{len(lsoa):,} LSOAs across {len(borough)} Boroughs (including City of London)",
        fontsize=11,
        fontweight='bold',
        pad=15,
        loc='left'
    )

    # Add professional cartographic elements (reusing existing functions)
    # Fixed coordinates for scale bar and north arrow (using axis-relative positions)
    _add_scale_bar(ax, length_km=scale_km, loc='lower right', fontsize=7)
    _add_north_arrow(ax, x=0.955, y=0.08, size=0.065)
    _add_map_credits(ax, data_period='June 2023 – May 2026', extra_note='Base map for study area')

    ax.set_axis_off()
    plt.tight_layout(pad=0.4)

    out_path = out_dir / "study_area_map.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path.name}")

# ══════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════

def run(panel:             pd.DataFrame,
        lsoa:              gpd.GeoDataFrame,
        city_name:         str  = "Greater London",
        manchester_data:   dict = None) -> None:
    """
    Run the full ESDA analysis, generating all output figures.

    Parameters
    ----------
    panel            : London panel dataset
    lsoa             : London LSOA boundaries
    city_name        : city name for map titles
    manchester_data  : optional, {"panel": DataFrame, "lsoa": GeoDataFrame}
    """

    from src.download_geodata import download_borough_boundaries
    borough = download_borough_boundaries()   # load London borough boundaries

    city_cfg  = CITIES.get(ACTIVE_CITY, {})
    scale_km  = city_cfg.get("scale_bar_km", 10)

    # Quick diagnostics
    nan_density  = panel[METRIC_COL].isna().mean()
    nan_rate     = panel["crime_rate_1k"].isna().mean()
    print(f"  [crime_density] NaN rate: {nan_density:.1%}")
    print(f"  [crime_rate_1k]  NaN rate: {nan_rate:.1%}")

    # ── Phase 0: Study area base map ──────────────────────────────
    plot_study_area_map(lsoa, borough, city_name=city_name, scale_km=scale_km)

    # ── Phase 3a: temporal trends ────────────────────────────────────────
    plot_temporal_trends(panel, city_name=city_name)

    # ── Phase 3b: Global Moran's I (all crimes) ────────────────────────
    compute_morans_series(panel, lsoa, crime_type=None)

    # ── Phase 3c: Moran's I per street crime type ─────────────────────
    for ct_key in STREET_CRIME_TYPES:
        col = f"density_{ct_key}"
        if col in panel.columns and panel[col].notna().any():
            compute_morans_series(panel, lsoa, crime_type=ct_key)
        else:
            print(f"  跳过 Moran's I [{ct_key}]: "
                  f"列{'不存在' if col not in panel.columns else '全为零/NaN'}")

    # ── Phase 3d: three time‑point all‑crime hotspot maps ──────────────
    for m in ["2023-12", "2025-03", "2026-04"]:
        plot_hotspot_map(panel, lsoa, month=m,
                         crime_type=None,
                         city_name=city_name,
                         scale_km=scale_km,
                         borough=borough)

    # ── Phase 3e: per‑crime‑type single‑month hotspot maps ─────────────
    available_cts = [
        k for k in STREET_CRIME_TYPES
        if f"density_{k}" in panel.columns
        and panel[f"density_{k}"].notna().any()
        and panel[f"density_{k}"].sum() > 0
    ]
    print(f"\n  Available street crime types: {available_cts} "
          f"({len(available_cts)}/{len(STREET_CRIME_TYPES)})")

    if not available_cts:
        print("  ⚠ No available density columns; skipping crime type hotspot maps.\n"
              "    → Solution: delete data/processed/panel.parquet and re‑run.")
    else:
        # Single‑month per‑type hotspot maps
        target_month = "2025-03"
        for ct_key in available_cts:
            plot_hotspot_map(panel, lsoa, month=target_month,
                             crime_type=ct_key, city_name=city_name,
                             scale_km=scale_km)

    # ── Phase 3f: LISA cluster maps for all crime (3 time points) ──────
    for m in ["2023-12", "2025-03", "2026-04"]:
        plot_lisa_cluster_map(panel, lsoa, borough, month=m,
                              crime_type=None, city_name=city_name,
                              scale_km=scale_km)

    # ── Phase 3g: per‑crime‑type LISA maps (single month) ──────────────
    target_month = "2025-03"   # representative validation month
    for ct_key in available_cts:
        col = f"density_{ct_key}"
        if col in panel.columns and panel[col].notna().any():
            plot_hotspot_map(panel, lsoa, month=target_month,
                             crime_type=ct_key,
                             city_name=city_name,
                             scale_km=scale_km,
                             borough=borough)
            # Also generate the corresponding LISA map for this crime type
            plot_lisa_cluster_map(panel, lsoa, borough, month=target_month,
                                  crime_type=ct_key, city_name=city_name,
                                  scale_km=scale_km)

    # ── Phase 3h: multi‑type comparison (adaptive layout, 3 time points) ──
    available_cts = all(
        f"density_{k}" in panel.columns and panel[f"density_{k}"].notna().any()
        for k in STREET_CRIME_TYPES
    )
    if available_cts:
        for m in ["2023-12", "2025-03", "2026-04"]:
            plot_crime_type_comparison(panel, lsoa, month=m,
                                        available_cts=available_cts,  # ← 传入实际可用列表
                                       city_name=city_name,
                                       scale_km=scale_km,
                                       borough=borough)
    else:
        print("  ⚠ Missing crime‑type density columns; skipping crime‑type comparison plot. "
              "Please ensure preprocess.py has been updated with CRIME_TYPES and density_* columns.")

    # ── Phase 3h: city comparison (optional, requires Manchester data) ──
    if manchester_data:
        panels_dict = {
            "london":     panel,
            "manchester": manchester_data["panel"],
        }
        lsoas_dict = {
            "london":     lsoa,
            "manchester": manchester_data["lsoa"],
        }
        common_months = sorted(
            set(panel["month"].unique()) &
            set(manchester_data["panel"]["month"].unique())
        )
        if common_months:
            plot_city_comparison(
                panels_dict, lsoas_dict,
                month     = common_months[len(common_months) // 2],
                crime_type = "theft_from_person",
            )
        else:
            print("  ⚠ London and Manchester data have no months in common; skipping city comparison.")
    else:
        print("  Manchester data not provided; skipping city comparison (optional analysis).")

if __name__ == "__main__":
    run()