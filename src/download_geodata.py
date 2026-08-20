"""
Download LSOA boundary files (ONS) and population data (ONS mid-year estimates).
LandScan requires manual registration; this script prioritises ONS LSOA population data.
"""
import io
import zipfile
import time
import logging
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path
from config import RAW_DIR, CRS_BNG, CRS_WGS84

# LSOA_URL = (
#     "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
#     "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/"
#     "FeatureServer/0/query"
# )
RAW_DIR = Path("data/raw")
CRS_BNG = "EPSG:27700"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ── Confirmed valid query endpoint ────────────────────────────────────────
ARCGIS_ENDPOINT = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/"
    "FeatureServer/0/query"
)

# Confirmed actual fields (no LAD field; LSOA21NMW and similar fields are not requested)
OUT_FIELDS = "LSOA21CD,LSOA21NM,BNG_E,BNG_N"
PAGE_SIZE  = 2000   # Confirmed in Figure 3/4: Max Record Count = 2000

# Approximate bounding box for Greater London in British National Grid (BNG),
# with an additional ~5 km buffer on each side to avoid missing cross‑boundary LSOAs.
# Official GLA extent is approximately E:503568-561957, N:155850-200933.
LONDON_BNG_BBOX = {
    "e_min": 498000, "e_max": 567000,
    "n_min": 150000, "n_max": 206000,
}
LONDON_WHERE = (
    f"BNG_E >= {LONDON_BNG_BBOX['e_min']} AND "
    f"BNG_E <= {LONDON_BNG_BBOX['e_max']} AND "
    f"BNG_N >= {LONDON_BNG_BBOX['n_min']} AND "
    f"BNG_N <= {LONDON_BNG_BBOX['n_max']}"
)

# ONS LSOA population (mid‑year estimates, used as normalisation benchmark)
# Download Excel file, sheet = MYE2 - Persons
POP_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
    "populationandmigration/populationestimates/datasets/"
    "lowersuperoutputareamidyearpopulationestimates/"
    "mid2022revisednov2025tomid2024/sapelsoasyoa20222024.xlsx"
)
# Priority order: prefer mid-2024 (aligned with the later part of the crime data window), then descend
TARGET_YEAR_PRIORITY = ["2024", "2023", "2022"]

def _fetch_page(offset: int,
                page_size: int = PAGE_SIZE
                ) -> tuple[gpd.GeoDataFrame | None, bool]:
    """
    Fetch a single page of LSOA boundaries.
    outSR=27700 → directly retrieve British National Grid coordinates;
    the layer extent is natively in 27700, so no subsequent reprojection is required.
    where=BNG range → replaces the non‑existent LAD field filter (confirmed that the field list contains no LAD).
    """
    params = {
        "where":             LONDON_WHERE,
        "outFields":         OUT_FIELDS,
        "outSR":             "27700",
        "f":                 "geojson",
        "returnGeometry":    "true",
        "resultOffset":      offset,
        "resultRecordCount": page_size,
    }

    for attempt in range(3):
        try:
            r = requests.get(
                ARCGIS_ENDPOINT, params=params,
                headers=HEADERS, timeout=90,
            )
            log.debug(f"  offset={offset} | HTTP {r.status_code} "
                      f"| {len(r.content)} bytes")

            if r.status_code != 200:
                log.warning(f"  HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(6 * (attempt + 1))
                continue

            if len(r.content) < 30:
                log.warning(f"  Response body too short: {r.text!r}")
                time.sleep(6 * (attempt + 1))
                continue

            try:
                raw = r.json()
            except Exception as json_err:
                log.warning(f"  JSON parsing failed: {json_err} | "
                            f"first 200 bytes: {r.text[:200]!r}")
                time.sleep(6 * (attempt + 1))
                continue

            if "error" in raw:
                log.warning(f"  API error: {raw['error']}")
                return None, False

            features = raw.get("features", [])
            if not features:
                return None, False

            # Confirmed: Return Exceeded Limit Features defaults to True;
            # this service reliably returns the exceededTransferLimit field.
            exceeded = raw.get("exceededTransferLimit", None)
            has_more = (exceeded if exceeded is not None
                       else len(features) >= page_size)

            gdf = gpd.GeoDataFrame.from_features(features, crs=CRS_BNG)
            gdf.columns = [
                c if c == "geometry" else c.upper()
                for c in gdf.columns
            ]
            return gdf, has_more

        except Exception as exc:
            log.warning(f"  Attempt {attempt+1}/3 failed: {exc}")
            time.sleep(6 * (attempt + 1))

    return None, False


def download_lsoa_boundaries(
    out_dir: Path = RAW_DIR / "boundaries",
) -> gpd.GeoDataFrame:
    """
    Download Greater London LSOA boundaries (2021 version, BGC, V5).

    Service: Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5
    (Confirmed as a real Feature Layer containing full polygon geometries.)

    Filtering strategy:
    The service field list contains no LAD code fields, so administrative filters
    such as 'LAD21CD LIKE E09%' cannot be used. Instead, we use the BNG_E/BNG_N
    fields (British National Grid coordinates) that are present, applying a rectangular
    bounding‑box filter on the LSOA centroids to directly delimit the Greater London area.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "lsoa_2021_london.gpkg"

    if out.exists():
        print(f"LSOA boundaries already exist, loading: {out}")
        return gpd.read_file(out)

    print("=" * 55)
    print("Downloading LSOA boundaries")
    print("Service: ...Boundaries_EW_BGC_V5 ")
    print(f"Filter: BNG_E [{LONDON_BNG_BBOX['e_min']}, "
          f"{LONDON_BNG_BBOX['e_max']}], "
          f"BNG_N [{LONDON_BNG_BBOX['n_min']}, "
          f"{LONDON_BNG_BBOX['n_max']}] (Greater London + 5 km buffer)")
    print("=" * 55)

    pages, offset = [], 0

    while True:
        gdf_page, has_more = _fetch_page(offset)

        if gdf_page is None:
            break

        pages.append(gdf_page)
        cumulative = sum(len(p) for p in pages)
        status = "continuing..." if has_more else "last page"
        print(f"  ✓ offset={offset:>5} → {len(gdf_page):>4} entries"
              f" (cumulative {cumulative:,}) [{status}]")

        if not has_more:
            break

        offset += PAGE_SIZE
        time.sleep(1.0)

    if not pages:
        raise RuntimeError(
            "API returned no data. Please check your network connection, or download manually:\n"
            "https://geoportal.statistics.gov.uk/datasets/"
            "ons::lower-layer-super-output-areas-december-2021-"
            "boundaries-ew-bgc-v5/about"
        )

    gdf = gpd.GeoDataFrame(
        pd.concat(pages, ignore_index=True), crs=CRS_BNG,
    )

    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["LSOA21CD"])
    if len(gdf) < before:
        log.info(f"  Deduplicated: {before} → {len(gdf)} entries")

    null_geom = gdf.geometry.isna().sum()
    if null_geom:
        log.warning(f"  Dropped {null_geom} entries with null geometry")
        gdf = gdf[gdf.geometry.notna()].copy()

    # Data are already in 27700; no reprojection needed.
    gdf["area_m2"] = gdf.geometry.area
    gdf.to_file(out, driver="GPKG")

    print(f"\n✓ {len(gdf):,} LSOAs (within Greater London bounding box) saved → {out}")
    print(f"  Area range: {gdf['area_m2'].min()/1e6:.3f} – "
          f"{gdf['area_m2'].max()/1e6:.1f} km²")
    print(f"  Note: This rectangular bounding‑box filter may include a small number "
          f"of LSOAs from surrounding counties (Surrey/Herts/Essex/Kent). "
          f"They will be naturally zeroed during spatial join with crime data in preprocess.py, "
          f"and do not affect subsequent modelling.")
    return gdf


def _discover_year_sheet(xl: pd.ExcelFile) -> tuple[str, str]:
    """
    Search all sheets in order of TARGET_YEAR_PRIORITY for a sheet matching the year,
    preferring sheets containing "Persons" (excluding Male/Female‑only sheets).
    Returns (sheet_name, matched_year).
    """
    sheet_names = xl.sheet_names
    log.info(f"  Sheet list: {sheet_names}")

    for year in TARGET_YEAR_PRIORITY:
        candidates = [
            s for s in sheet_names
            if year in s and "persons" in s.lower()
        ]
        if not candidates:
            candidates = [
                s for s in sheet_names
                if year in s and "male" not in s.lower()
                and "female" not in s.lower()
            ]
        if candidates:
            return candidates[0], year

    log.warning(f"  No year关键词 matched; falling back to last sheet: "
               f"{sheet_names[-1]}")
    return sheet_names[-1], "unknown"


def download_population(
    out_dir: Path = RAW_DIR / "population",
) -> pd.DataFrame:
    """
    Download and parse ONS LSOA population estimates.
    Data source (confirmed on 2026-06-30): "Mid-2022 revised (Nov 2025) to mid-2024"
    edition, a single xlsx (83.4 MB) with single year of age and sex dimensions,
    combining mid-2022 (revised), mid-2023, and mid-2024 estimates.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "lsoa_population.parquet"

    if out.exists():
        print(f"Population data already exists: {out}")
        return pd.read_parquet(out)

    # ── Local files take precedence (for manual downloads and older ZIP formats) ──────────
    xlsx_files = list(out_dir.glob("*.xlsx"))
    zip_files  = list(out_dir.glob("*.zip"))

    if xlsx_files:
        xl_path = xlsx_files[0]
        print(f"  Using existing local file: {xl_path.name}")
    elif zip_files:
        print(f"  Found ZIP file (old SAPE format), extracting: {zip_files[0].name}")
        z = zipfile.ZipFile(zip_files[0])
        xl_name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        xl_path = out_dir / Path(xl_name).name
        with z.open(xl_name) as src, open(xl_path, "wb") as dst:
            dst.write(src.read())
    else:
        print("Downloading ONS LSOA population data (Mid-2022 revised to Mid-2024)...")
        print(f"  URL: {POP_URL}")
        try:
            r = requests.get(POP_URL, headers=HEADERS, timeout=300,
                             stream=True)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")

            total, chunks = 0, []
            for chunk in r.iter_content(chunk_size=131072):
                chunks.append(chunk)
                total += len(chunk)
                print(f"\r  Downloaded {total/1e6:.1f} / 83.4 MB...",
                      end="", flush=True)
            print()
            content = b"".join(chunks)

            if len(content) < 1_000_000:
                raise RuntimeError(
                    f"File unusually small ({len(content)} bytes). "
                    f"Download may have failed or the ONS page structure may have changed."
                )

            xl_path = out_dir / "sapelsoasyoa20222024.xlsx"
            xl_path.write_bytes(content)
            print(f"  ✓ Saved: {xl_path}")

        except Exception as exc:
            print(f"""
❌ Automatic download failed: {exc}

Please download manually:
  1. Open your browser and go to:
     https://www.ons.gov.uk/peoplepopulationandcommunity/
     populationandmigration/populationestimates/datasets/
     lowersuperoutputareamidyearpopulationestimates

  2. Find the "Mid-2022 revised (Nov 2025) to mid-2024 edition"
     and click the green "xlsx (83.4 MB)" button to download.

  3. Place the downloaded file in:
     {out_dir.resolve()}

  4. Re-run this script.
""")
            raise FileNotFoundError("Population data download failed; manual download required") from exc

    # ── Parse Excel ────────────────────────────────────────────
    print(f"\n  Parsing: {xl_path.name}")
    xl = pd.ExcelFile(xl_path)
    sheet_name, matched_year = _discover_year_sheet(xl)
    print(f"  Target sheet: '{sheet_name}' (matched year: mid-{matched_year})")

    for skip_rows in [4, 5, 6, 3, 7]:
        try:
            df_raw = pd.read_excel(
                xl_path, sheet_name=sheet_name, header=skip_rows,
            )
            # Fix: only match E01/W01 LSOA codes
            lsoa_col = next(
                (c for c in df_raw.columns
                 if df_raw[c].astype(str).str.match(r"^[EW]01\d{6}$").any()),
                None,
            )
            if lsoa_col:
                print(f"  header row = {skip_rows}, LSOA column = '{lsoa_col}'")
                break
        except Exception:
            continue
    else:
        raise ValueError(
            f"Could not automatically detect the table format in sheet '{sheet_name}'.\n"
            f"Available sheets: {xl.sheet_names}\n"
            f"Please manually inspect using pd.ExcelFile('{xl_path}').sheet_names "
            f"and pd.read_excel(...).head(10)."
        )

    total_col = next(
        (c for c in df_raw.columns
         if str(c).strip().lower() in
         ["total", "all ages", "persons", "mid-2024", "mid-2023",
          "mid-2022"]),
        None,
    )
    if total_col is None:
        numeric_cols = df_raw.select_dtypes("number").columns.tolist()
        age_like = [c for c in numeric_cols
                   if str(c).strip().replace("+", "").isdigit()]
        if len(age_like) >= 80:
            df_raw["population"] = df_raw[age_like].sum(axis=1)
            total_col = "population"
            print(f"  No pre‑aggregated total column found; summed {len(age_like)} single‑year age columns.")
        elif numeric_cols:
            total_col = numeric_cols[-1]
            print(f"  Falling back to column '{total_col}' as total population.")
        else:
            raise ValueError(
                f"No usable population numeric column found in sheet '{sheet_name}'."
            )

    df = pd.DataFrame({
        "lsoa_code":      df_raw[lsoa_col].astype(str).str.strip(),
        "population_est": pd.to_numeric(df_raw[total_col],
                                        errors="coerce"),
    })

    df = df[df["lsoa_code"].str.match(r"^E\d{8}$")]
    df = df.dropna(subset=["population_est"])
    df["population_est"] = df["population_est"].astype(int)
    df["reference_year"]  = f"mid-{matched_year}"

    df.to_parquet(out, index=False)
    print(f"\n✓ {len(df):,} LSOA population records saved → {out}")
    print(f"  Reference year: mid-{matched_year}")
    print(f"  Population range: {df['population_est'].min():,} – "
          f"{df['population_est'].max():,}")
    return df

def download_lsoa_boundaries_manchester(
    out_dir: Path = RAW_DIR / "boundaries",
) -> gpd.GeoDataFrame:
    """
    Download LSOA boundaries for Greater Manchester.
    Uses the same BGC V5 service as London, filtering by BNG coordinates for Manchester.
    """
    from config import CITIES
    mcr_cfg = CITIES["manchester"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "lsoa_2021_manchester.gpkg"

    if out.exists():
        print(f"Manchester LSOA boundaries already exist, loading: {out}")
        return gpd.read_file(out)

    # Manchester BNG filter extent
    mcr_where = (
        f"BNG_E >= {mcr_cfg['bng_e_min']} AND BNG_E <= {mcr_cfg['bng_e_max']} AND "
        f"BNG_N >= {mcr_cfg['bng_n_min']} AND BNG_N <= {mcr_cfg['bng_n_max']}"
    )

    print("=" * 55)
    print("Downloading Greater Manchester LSOA boundaries")
    print(f"BNG filter: E[{mcr_cfg['bng_e_min']}, {mcr_cfg['bng_e_max']}], "
          f"N[{mcr_cfg['bng_n_min']}, {mcr_cfg['bng_n_max']}]")
    print("=" * 55)

    # Reuse the London pagination logic, only changing the WHERE clause
    pages, offset = [], 0
    while True:
        params = {
            "where":             mcr_where,
            "outFields":         OUT_FIELDS,
            "outSR":             "27700",
            "f":                 "geojson",
            "returnGeometry":    "true",
            "resultOffset":      offset,
            "resultRecordCount": PAGE_SIZE,
        }
        for attempt in range(3):
            try:
                r = requests.get(ARCGIS_ENDPOINT, params=params,
                                 headers=HEADERS, timeout=90)
                if r.status_code != 200 or len(r.content) < 30:
                    time.sleep(6 * (attempt + 1))
                    continue
                raw = r.json()
                if "error" in raw:
                    break
                features = raw.get("features", [])
                if not features:
                    break
                exceeded = raw.get("exceededTransferLimit",
                                   len(features) >= PAGE_SIZE)
                gdf_page = gpd.GeoDataFrame.from_features(features,
                                                           crs=CRS_BNG)
                gdf_page.columns = [
                    c if c == "geometry" else c.upper()
                    for c in gdf_page.columns
                ]
                pages.append(gdf_page)
                n = sum(len(p) for p in pages)
                print(f"  ✓ offset={offset:>5} → {len(gdf_page):>4} entries"
                      f" (cumulative {n:,}){'[continuing]' if exceeded else '[last page]'}")
                if not exceeded:
                    break
                offset += PAGE_SIZE
                time.sleep(1.0)
                break
            except Exception as exc:
                print(f"  Attempt {attempt+1}/3 failed: {exc}")
                time.sleep(6 * (attempt + 1))
        else:
            break
        if not features or not exceeded:
            break

    if not pages:
        raise RuntimeError("Manchester LSOA boundary download failed. Please check network or download manually.")

    gdf = gpd.GeoDataFrame(
        pd.concat(pages, ignore_index=True), crs=CRS_BNG,
    )
    gdf = gdf.drop_duplicates(subset=["LSOA21CD"])
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf.to_file(out, driver="GPKG")
    print(f"\n✓ {len(gdf):,} Greater Manchester LSOAs → {out}")
    return gdf

def download_borough_boundaries(
    out_dir: Path = RAW_DIR / "boundaries",
) -> gpd.GeoDataFrame:
    """
    Download London's 32 borough boundaries for hotspot map overlay.
    Satisfies supervisor feedback item 5: all hotspot maps must delineate borough boundaries with thick lines.

    Service verified (2026-07 web check):
      Local_Authority_Districts_December_2023_Boundaries_UK_BGC
      Same ArcGIS organisation ID (ESMARspQHYMw9BZ9) as the LSOA service,
      reusing the same proven pagination/parsing logic.
    """
    out = Path(out_dir) / "borough_boundaries_london.gpkg"
    if out.exists():
        print(f"Borough boundaries already exist: {out}")
        return gpd.read_file(out)

    endpoint = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
        "Local_Authority_Districts_December_2023_Boundaries_UK_BGC/"
        "FeatureServer/0/query"
    )
    params = {
        "where":             "LAD23CD LIKE 'E09%'",   # only the 32 London boroughs
        "outFields":         "LAD23CD,LAD23NM",
        "outSR":              "27700",
        "f":                  "geojson",
        "returnGeometry":     "true",
        "resultRecordCount":  100,
    }

    print("Downloading London Borough boundaries (for hotspot map overlay)...")
    r = requests.get(endpoint, params=params, headers=HEADERS, timeout=90)
    if r.status_code != 200 or len(r.content) < 30:
        raise RuntimeError(
            f"Borough boundary download failed (HTTP {r.status_code}).\n"
            f"Please manually access: https://geoportal.statistics.gov.uk/datasets/"
            f"ons::local-authority-districts-december-2023-boundaries-uk-bgc"
        )

    gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs=CRS_BNG)
    gdf.columns = [c if c == "geometry" else c.upper() for c in gdf.columns]
    gdf.to_file(out, driver="GPKG")
    print(f"✓ {len(gdf)} London Borough boundaries → {out}")
    return gdf

if __name__ == "__main__":
    download_lsoa_boundaries()
    download_population()