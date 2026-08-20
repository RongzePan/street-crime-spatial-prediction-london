"""
Complete pipeline: executes all stages sequentially.
Individual stages can be run separately via command-line arguments.
"""
import argparse
import pandas as pd
import geopandas as gpd

from src.download_police   import download        as dl_police
from src.download_geodata  import (download_lsoa_boundaries,
                                   download_population)
from src.preprocess        import run             as preprocess
from src.esda              import run             as esda
from src.kde_baseline      import run_kde
from src.lstm_model        import run_lstm
from src.evaluation        import compare_and_plot
from config                import PROC_DIR


def main(stage: str = "all", include_manchester: bool = False):
    """
    Modification notes:
    - Phase 3 now accepts city_name and optional manchester_data
    - New optional Manchester analysis branch added
    """
    from config import CITIES, ACTIVE_CITY

    city_cfg  = CITIES.get(ACTIVE_CITY, {})
    city_name = city_cfg.get("name", "Greater London")

    if stage in ("all", "download"):
        print("=" * 50)
        print("PHASE 1: Data acquisition")
        dl_police()
        download_lsoa_boundaries()
        download_population()

    if stage in ("all", "preprocess"):
        print("=" * 50)
        print("PHASE 2: Preprocessing")
        preprocess()

    panel = pd.read_parquet(PROC_DIR / "panel.parquet")
    panel["month"] = pd.to_datetime(panel["month"])
    lsoa  = gpd.read_file(f"data/raw/boundaries/{city_cfg.get('gpkg_file', 'lsoa_2021_london.gpkg')}")

    # ── Optional: load Manchester data ─────────────────────────────────────
    manchester_data = None
    if include_manchester:
        try:
            from src.download_police    import download_manchester
            from src.download_geodata   import download_lsoa_boundaries_manchester
            from src.preprocess         import run as preprocess_manchester
            print("=" * 50)
            print("PHASE 1b: Manchester data acquisition")
            download_manchester()
            mcr_lsoa  = download_lsoa_boundaries_manchester()
            # Note: Manchester requires separate preprocessing (passing a distinct police directory).
            # This is a simplified implementation; complete deployment requires
            # full refactoring of preprocess.run().
            print("  Manchester LSOA boundaries prepared; crime data preprocessing requires additional configuration.")
            manchester_data = None   # Complete implementation requires additional configuration; refer to developer documentation
        except Exception as e:
            print(f"  ⚠ Manchester data loading failed: {e}; skipping city comparison.")

    if stage in ("all", "esda"):
        print("=" * 50)
        print("PHASE 3: Exploratory spatial data analysis")
        esda(panel, lsoa,
                 city_name       = city_name,
                 manchester_data = manchester_data)

    if stage in ("all", "model"):
        print("=" * 50)
        print("PHASE 4a: KDE baseline")
        kde_res  = run_kde(panel, lsoa)

        print("=" * 50)
        print("PHASE 4b: LSTM model")
        lstm_res = run_lstm(panel)

        print("=" * 50)
        print("PHASE 5: Evaluation")
        compare_and_plot(kde_res, lstm_res)

    print("\n✓ All phases complete. Check outputs/ for results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
        choices=["all", "download", "preprocess", "esda", "model"])
    parser.add_argument("--manchester", action="store_true",
        help="Run Manchester analysis concurrently (optional; requires additional download time)")
    args = parser.parse_args()
    main(args.stage, include_manchester=args.manchester)