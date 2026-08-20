"""
Download street-level crime data for London from the data.police.uk API (2023–2026).
Uses a 5×5 grid to subdivide the London bounding box, circumventing the API's
single‑request limit of 10,000 records.

download_police.py  —— 503‑adaptive fix

Root cause (confirmed):
  The blank area in North London on the hotspot maps corresponds to
  row=3,col=2 (Camden/Islington) and row=3,col=3 (Hackney/Tower Hamlets).
  These two cells have ~8,000–10,000+ crimes per month, consistently triggering
  HTTP 503 responses.
  The original code silently skipped such cells after retries → crime_count=0
  for the affected LSOAs → white holes in the hotspot maps.

Fix:
  1. fetch_cell_raw: distinguishes 503 from other errors, returning (data, is_503)
  2. fetch_cell_adaptive: on 503, automatically subdivides the polygon into
     quadrants and retries recursively (up to depth 3 = 64 sub‑cells)
  3. _check_and_clear_stale_cache: a metadata file records download settings;
     when running the fixed code for the first time (no metadata) → stale cache
     is cleared → triggers a fresh download.
"""
import time
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import (LONDON_BBOX, GRID_N, START_YEAR, START_MONTH,
                    END_YEAR, END_MONTH, POLICE_API, API_DELAY,
                    API_RETRIES, RAW_DIR)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

_META_FILE = "download_meta.json"   # metadata file recording GRID_N and other settings

# ══════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════
def date_range(sy, sm, ey, em):
    """Generate a list of 'YYYY-MM' strings."""
    dates, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        dates.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return dates


def grid_polygons(bbox, n):
    """
    Evenly divide the bounding box into n×n rectangular polygons.
    Returns a list of strings in the format expected by data.police.uk:
    'lat1,lng1:lat2,lng2:lat3,lng3:lat4,lng4'
    """
    w, e = bbox["west"], bbox["east"]
    s, n_ = bbox["south"], bbox["north"]
    dlat = (n_ - s) / n
    dlng = (e - w) / n
    polys = []
    for i in range(n):
        for j in range(n):
            s0 = s + i * dlat
            n0 = s0 + dlat
            w0 = w + j * dlng
            e0 = w0 + dlng
            polys.append(
                f"{s0:.6f},{w0:.6f}:{n0:.6f},{w0:.6f}:"
                f"{n0:.6f},{e0:.6f}:{s0:.6f},{e0:.6f}"
            )
    return polys

def _split_polygon(poly_str: str) -> list:
    """
    Split a rectangular polygon into four quadrants (SW, SE, NW, NE).
    Format: 'lat_s,lon_w : lat_n,lon_w : lat_n,lon_e : lat_s,lon_e'
    """
    coords = [c.split(",") for c in poly_str.split(":")]
    lats = [float(c[0]) for c in coords]
    lngs = [float(c[1]) for c in coords]
    s, n = min(lats), max(lats)
    w, e = min(lngs), max(lngs)
    ml = (s + n) / 2    # mid-latitude
    mg = (w + e) / 2    # mid-longitude
    return [
        f"{s:.6f},{w:.6f}:{ml:.6f},{w:.6f}:{ml:.6f},{mg:.6f}:{s:.6f},{mg:.6f}",
        f"{s:.6f},{mg:.6f}:{ml:.6f},{mg:.6f}:{ml:.6f},{e:.6f}:{s:.6f},{e:.6f}",
        f"{ml:.6f},{w:.6f}:{n:.6f},{w:.6f}:{n:.6f},{mg:.6f}:{ml:.6f},{mg:.6f}",
        f"{ml:.6f},{mg:.6f}:{n:.6f},{mg:.6f}:{n:.6f},{e:.6f}:{ml:.6f},{e:.6f}",
    ]

# ══════════════════════════════════════════════════════════════════
# Core fix: 503‑adaptive download
# ══════════════════════════════════════════════════════════════════
def fetch_cell_raw(poly_str: str, date: str) -> tuple:
    """
    Single API call. Returns (crimes_list, is_503).

    Fix: the original fetch_cell() would retry on 503 and then return [],
    conflating a genuine empty result with a 503 overload. This version
    explicitly returns is_503=True so the caller can decide to subdivide.
    """
    got_503 = False
    for attempt in range(API_RETRIES):
        try:
            r = requests.post(
                f"{POLICE_API}/crimes-street/all-crime",
                data={"poly": poly_str, "date": date},
                timeout=40,
            )
            if r.status_code == 200:
                data = r.json()
                return (data if isinstance(data, list) else []), False

            if r.status_code == 503:
                got_503 = True
                log.debug(
                    f"  503 for {date} attempt {attempt+1}/{API_RETRIES}"
                    f" — cell likely exceeds ~10,000 crime limit"
                )
                time.sleep(4 * (attempt + 1))
                continue
            # 404 means no data for that month
            # Other HTTP error
            log.warning(f"  HTTP {r.status_code} for {date}")
            return [], False

        except requests.RequestException as exc:
            log.warning(f"Attempt {attempt+1} failed ({date}): {exc}")
            time.sleep(5)

    # All retries exhausted
    if got_503:
        return [], True     # ← explicit signal: subdivide this polygon
    return [], False

def fetch_cell_adaptive(poly_str: str, date: str, depth: int = 0) -> list:
    """
    Adaptive download: on 503, automatically quad‑split the polygon and
    recurse (up to depth 3 = 64 sub‑cells).

    The two failing central‑London cells (row=3,col=2/3) cover ~105 km² each,
    carrying ~8,000–10,000 crimes per month.
      → depth 1 sub‑cell: ~26 km², ~2,000–3,000 crimes – safe
      → depth 2 sub‑cell: ~6.5 km², ~500–800 crimes – very safe

    Original behaviour (now fixed):
        fetch_cell() → 503 × 3 retries → return [] → data lost forever
    Fixed:
        fetch_cell_raw() → 503 → _split_polygon() → 4 recursive children → full data
    """
    crimes, is_503 = fetch_cell_raw(poly_str, date)

    if not is_503:
        return crimes       # normal return (including empty non‑503 results)

    MAX_DEPTH = 3           # 4^3 = 64 sub‑cells, each ~1.6 km², extremely safe
    if depth >= MAX_DEPTH:
        log.warning(
            f"  503 persists at max depth={MAX_DEPTH} for {date}."
            f" Accepting empty sub-cell."
        )
        return []

    sub_polys = _split_polygon(poly_str)
    log.info(
        f"  503 → subdividing cell (depth={depth}→{depth+1}) "
        f"for {date}"
    )
    result = []
    for sub_poly in sub_polys:
        result.extend(fetch_cell_adaptive(sub_poly, date, depth + 1))
        time.sleep(API_DELAY)
    return result

# ══════════════════════════════════════════════════════════════════
# Cache version management
# ══════════════════════════════════════════════════════════════════

def _check_and_clear_stale_cache(output_dir: Path) -> None:
    """
    Compare the GRID_N stored in the metadata file with the current config.

    ┌─ No metadata (first run with the fixed code) ────────────────┐
    │ Existing caches were generated by the old (non‑adaptive)     │
    │ code and contain data gaps; they must be re‑downloaded.      │
    │ Delete all crimes_*.parquet files → trigger a full redownload.│
    └──────────────────────────────────────────────────────────────┘
    ┌─ Metadata exists but GRID_N differs ─────────────────────────┐
    │ Grid size has changed; historical data may be incomplete.    │
    │ Clear the cache as well.                                    │
    └──────────────────────────────────────────────────────────────┘
    ┌─ Metadata exists and GRID_N matches ─────────────────────────┐
    │ Cache is valid; skip existing files (normal caching).        │
    └──────────────────────────────────────────────────────────────┘

    Compare metadata with current configuration; clear stale caches if any discrepancy.

    New detection: start/end date ranges (previously only checked GRID_N).

    Example use‑cases:
      - 2023-05 → 2023-06: crimes_2023-05.parquet falls outside the new range → delete
      - 2026-04 → 2026-05: crimes_2026-05.parquet not yet downloaded → will be added
    """
    meta_path = output_dir / _META_FILE
    existing_parquets = list(output_dir.glob("crimes_*.parquet"))
    current_start = f"{START_YEAR}-{START_MONTH:02d}"
    current_end = f"{END_YEAR}-{END_MONTH:02d}"

    if not meta_path.exists():
        if existing_parquets:
            log.warning(
                f"\n{'='*60}\n"
                f"No download metadata file found.\n"
                f"There are {len(existing_parquets)} cached files from an older version,\n"
                f"which may contain data gaps due to HTTP 503 in North London.\n"
                f"Clearing old caches and re‑downloading (with 503‑adaptive fix)...\n"
                f"Estimated download time: ~30–50 minutes.\n"
                f"{'='*60}"
            )
            for f in existing_parquets:
                f.unlink()
        return

    with open(meta_path) as fh:
        meta = json.load(fh)

    cached_n = meta.get("grid_n", -1)
    if cached_n != GRID_N:
        log.info(
            f"GRID_N changed ({cached_n} → {GRID_N}), "
            f"clearing {len(existing_parquets)} cache files..."
        )
        for f in existing_parquets:
            f.unlink()
        meta_path.unlink()

    cached_n     = meta.get("grid_n",  -1)
    cached_start = meta.get("start",   "")
    cached_end   = meta.get("end",     "")

    reasons = []
    if cached_n != GRID_N:
        reasons.append(f"GRID_N: {cached_n} → {GRID_N}")
    if cached_start != current_start:
        reasons.append(f"start: '{cached_start}' → '{current_start}'")
    if cached_end != current_end:
        reasons.append(f"end: '{cached_end}' → '{current_end}'")

    if not reasons:
        return  # config matches, cache is valid

    log.info(
        f"Configuration change detected ({', '.join(reasons)}), "
        f"processing caches..."
    )

    # ── Precise deletion: only remove files outside the new date range ────
    # This is more efficient than clearing everything, as the old and new
    # ranges overlap for 35 months.
    valid_dates = set(date_range(START_YEAR, START_MONTH,
                                 END_YEAR,   END_MONTH))
    deleted, kept = 0, 0
    for f in existing_parquets:
        # filename format: crimes_YYYY-MM.parquet
        stem  = f.stem                # e.g. "crimes_2023-05"
        parts = stem.split("_")
        if len(parts) == 2 and parts[1] not in valid_dates:
            f.unlink()
            log.info(f"  Deleting out‑of‑range cache: {f.name}")
            deleted += 1
        else:
            kept += 1

    log.info(f"  Cache cleanup completed: {deleted} deleted, {kept} retained")
    meta_path.unlink()  # remove old metadata; it will be re‑written after download

def _write_meta(output_dir: Path) -> None:
    """Write metadata file, recording the exact configuration used."""
    meta_path = output_dir / _META_FILE
    with open(meta_path, "w") as fh:
        json.dump({
            "grid_n":       GRID_N,
            "api_adaptive": True,
            "start":        f"{START_YEAR}-{START_MONTH:02d}",
            "end":          f"{END_YEAR}-{END_MONTH:02d}",
        }, fh, indent=2)

# ══════════════════════════════════════════════════════════════════
# Main download function
# ══════════════════════════════════════════════════════════════════

def download(output_dir: Path = RAW_DIR / "police"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Clear any stale caches that might contain 503‑induced gaps ──
    _check_and_clear_stale_cache(output_dir)

    dates = date_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    polys = grid_polygons(LONDON_BBOX, GRID_N)

    log.info(
        f"{len(dates)} months × {len(polys)} cells = "
        f"{len(dates) * len(polys)} base API calls "
        f"(+adaptive sub-calls for 503 cells)"
    )

    for date in tqdm(dates, desc="months"):
        out = output_dir / f"crimes_{date}.parquet"
        if out.exists():
            continue                # valid cache, skip

        rows = []
        for poly in polys:
            # ← critical fix: use adaptive download instead of the original fetch_cell()
            rows.extend(fetch_cell_adaptive(poly, date))
            time.sleep(API_DELAY)

        if not rows:
            log.warning(f"No data for {date}")
            continue

        df = pd.DataFrame(rows).drop_duplicates(subset=["id"])

        # Flatten nested 'location' fields
        df["latitude"] = df["location"].apply(
            lambda x: float(x["latitude"]) if x else None)
        df["longitude"] = df["location"].apply(
            lambda x: float(x["longitude"]) if x else None)
        df["street"] = df["location"].apply(
            lambda x: x["street"]["name"] if x else None)
        df = df.drop(columns=["location"], errors="ignore")

        df.to_parquet(out, index=False)
        log.info(f"  {date}: {len(df):,} crimes saved")

    # ── Step 2: Write metadata file for future cache consistency checks ──
    _write_meta(output_dir)
    log.info("Download complete.")

def download_manchester(output_dir: Path = RAW_DIR / "police_manchester"):
    """
    Download street crime data for Manchester (using the city‑specific bounding box).
    Called in the same way as download(), differing only in bounding box and output directory.
    """
    from config import CITIES
    mcr_cfg = CITIES["manchester"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_and_clear_stale_cache(output_dir)

    dates  = date_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    polys  = grid_polygons(mcr_cfg["bbox"], mcr_cfg["grid_n"])

    log.info(f"[Manchester] {len(dates)} months × {len(polys)} cells = "
             f"{len(dates) * len(polys)} base API calls")

    for date in tqdm(dates, desc="Manchester months"):
        out = output_dir / f"crimes_{date}.parquet"
        if out.exists():
            continue
        rows = []
        for poly in polys:
            rows.extend(fetch_cell_adaptive(poly, date))
            time.sleep(API_DELAY)
        if not rows:
            log.warning(f"No data for Manchester {date}")
            continue
        df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
        df["latitude"]  = df["location"].apply(
            lambda x: float(x["latitude"]) if x else None)
        df["longitude"] = df["location"].apply(
            lambda x: float(x["longitude"]) if x else None)
        df["street"]    = df["location"].apply(
            lambda x: x["street"]["name"] if x else None)
        df = df.drop(columns=["location"], errors="ignore")
        df.to_parquet(out, index=False)
        log.info(f"  Manchester {date}: {len(df):,} crimes saved")

    _write_meta(output_dir)
    log.info("Manchester download complete.")

if __name__ == "__main__":
    download()