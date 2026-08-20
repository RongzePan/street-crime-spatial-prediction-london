from pathlib import Path

# ── Directory Structure ────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
FIG_DIR     = OUTPUT_DIR / "figures"
RESULT_DIR  = OUTPUT_DIR / "results"

for d in [RAW_DIR/"police", RAW_DIR/"boundaries", RAW_DIR/"population",
          PROC_DIR, MODEL_DIR, FIG_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Research Parameters ─────────────────────────────────────────────
START_YEAR, START_MONTH = 2023, 6
END_YEAR,   END_MONTH   = 2026, 5

# London bounding box (WGS84)
LONDON_BBOX = {"west": -0.510, "east": 0.334, "south": 51.286, "north": 51.692}
GRID_N      = 5          # 5×5 grid; approximately 2,500 crimes per cell per month, below API limit

# Coordinate reference systems
CRS_WGS84 = "EPSG:4326"
CRS_BNG   = "EPSG:27700"  # British National Grid (metric, used for spatial operations)

# ── Temporal Partitioning ─────────────────────────────────────────────
TRAIN_END = "2024-12"   # 2023-05 to 2024-12 = 20 months → (20-6)=14 sequences/LSOA
# Training set cut-off
VAL_END   = "2025-06"   # 2025-01 to 2025-06 = 6 months → 6 sequences/LSOA
# Validation set cut-off
# Test:      2025-07 to 2026-04 = 10 months → 10 sequences/LSOA (fully aligned with KDE test period)

# ── LSTM Hyperparameters ───────────────────────────────────────────
# SEQ_LEN        = 12    # Input sequence length (months)
# ── Root cause fix: SEQ_LEN 12 → 6 ──────────────────────────────────
# With 36 months of data, SEQ_LEN=12 prevents val (6 months) and test (10 months)
# from forming any valid sequences.
# SEQ_LEN=6: train (14 sequences/LSOA) | val (6 sequences/LSOA) | test (10 sequences/LSOA, aligned with KDE)
SEQ_LEN        = 6      # Original 12 → 6
HIDDEN_SIZE    = 64
NUM_LAYERS     = 2
DROPOUT        = 0.2
LR             = 1e-3
EPOCHS         = 150
BATCH_SIZE     = 64
PATIENCE       = 15     # Early stopping patience (epochs)

# ── KDE Parameters ──────────────────────────────────────────────
KDE_BANDWIDTH  = 500   # Unit: metres
KDE_GRID_SIZE  = 100   # Grid resolution (metres)

# ── Evaluation Parameters ─────────────────────────────────────────────
HOTSPOT_PCT    = 0.10  # Top 10% of area defined as hotspot (used for PAI calculation)
RANDOM_SEED    = 42

# ── API ──────────────────────────────────────────────────
POLICE_API     = "https://data.police.uk/api"
API_DELAY      = 1.2   # Seconds (courtesy delay between requests)
API_RETRIES    = 3

# ══════════════════════════════════════════════════════════════════
# Street Crime Type Definitions (focused analysis, mapping to data.police.uk API categories)
# ══════════════════════════════════════════════════════════════════

STREET_CRIME_TYPES = {
    "theft_from_person": {
        "api_categories": ["theft-from-the-person"],
        "display_name":   "Theft from Person (Pickpocketing & Purse Snatching)",
        "short_name":     "Pickpocketing",
        "description":    ("Opportunistic theft directly from a person on the street, "
                           "including pickpocketing, bag snatching and purse theft. "
                           "Highly concentrated in transit hubs and retail zones."),
        "cmap":           "Reds",
        "color":          "#C0392B",
    },
    "robbery": {
        "api_categories": ["robbery"],
        "display_name":   "Street Robbery (inc. Moped-enabled Robbery)",
        "short_name":     "Street Robbery",
        "description":    ("Personal robbery on the street, including moped-enabled "
                           "and bicycle-enabled snatch theft targeting pedestrians "
                           "and cyclists in public spaces."),
        "cmap":           "Purples",
        "color":          "#7D3C98",
    },
    "violence_sexual": {
        "api_categories": ["violence-and-sexual-offences"],
        "display_name":   "Violence & Sexual Offences (inc. Street Harassment)",
        "short_name":     "Violence & Sexual Offences",
        "description":    ("Street-based violence and sexual offences, including assault, "
                           "sexual harassment and sexual violence in public spaces. "
                           "Note: sub-categories not separable via data.police.uk API."),
        "cmap":           "Oranges",
        "color":          "#D35400",
    },
    "public_order": {
        "api_categories": ["public-order"],
        "display_name":   "Public Order Offences",
        "short_name":     "Public Order",
        "description":    ("Offences causing harassment, alarm or distress in public "
                           "spaces, including threatening behaviour, riot and disorder. "
                           "Concentrated in night-time economy zones."),
        "cmap":           "Blues",
        "color":          "#1A5276",
    },
}

# Primary crime type for focused LSTM/KDE modelling
PRIMARY_CRIME_TYPE = "theft_from_person"

# ══════════════════════════════════════════════════════════════════
# City Configurations (supporting London and Manchester comparative analysis)
# ══════════════════════════════════════════════════════════════════

CITIES = {
    "london": {
        "name":         "Greater London",
        "short_name":   "London",
        "bbox":         {"west": -0.510, "east":  0.334,
                         "south": 51.286, "north": 51.692},
        "grid_n":       5,
        "gpkg_file":    "lsoa_2021_london.gpkg",
        # BNG filter extent (for ArcGIS API WHERE clause)
        "bng_e_min":    498000, "bng_e_max": 567000,
        "bng_n_min":    150000, "bng_n_max": 206000,
        "scale_bar_km": 10,
    },
    "manchester": {
        "name":         "Greater Manchester",
        "short_name":   "Manchester",
        "bbox":         {"west": -2.750, "east": -1.900,
                         "south": 53.300, "north": 53.700},
        "grid_n":       4,
        "gpkg_file":    "lsoa_2021_manchester.gpkg",
        "bng_e_min":    330000, "bng_e_max": 420000,
        "bng_n_min":    380000, "bng_n_max": 430000,
        "scale_bar_km": 10,
    },
}

ACTIVE_CITY = "london"   # Switch to "manchester" to run Manchester analysis