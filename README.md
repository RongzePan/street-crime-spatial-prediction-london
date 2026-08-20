# street-crime-spatial-prediction-london

A spatio‑temporal machine learning framework for predicting street‑level crime hotspots in Greater London using LSTM and Kernel Density Estimation (KDE). This repository accompanies the MSc dissertation *“Leveraging Machine Learning for Spatial Crime Prediction: A Comparative Study of KDE and LSTM in Greater London”* (UCL Security & Crime Science, 2026).

---

## Overview

This project implements a complete pipeline for:
- Downloading and processing street‑level crime data from [data.police.uk](https://data.police.uk)
- Aggregating crimes to Lower Layer Super Output Areas (LSOAs) and monthly time steps
- Generating a rich panel dataset with 28 features including historical crime counts, ambient population (LandScan), and temporal cycles
- Computing global Moran’s I and Local Indicators of Spatial Association (LISA) for exploratory spatial data analysis
- Training a Kernel Density Estimation (KDE) baseline model and an LSTM neural network for hotspot prediction
- Evaluating predictive accuracy using the Predictive Accuracy Index (PAI) and producing publication‑ready maps and figures

All code is written in Python 3.9+ and relies on standard open‑source libraries (GeoPandas, PyTorch, libpysal, Matplotlib, etc.).

---

## Data Sources

| Dataset | Source | Licence |
|---------|--------|---------|
| Street‑level crime records (Jun 2023 – May 2026) | [data.police.uk](https://data.police.uk) | Open Government Licence v3.0 |
| LSOA boundaries (Dec 2021, BGC V5) | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) | Open Government Licence v3.0 |
| LSOA mid‑year population estimates (2022–2024) | Office for National Statistics | Open Government Licence v3.0 |
| Ambient population (LandScan Global) | Oak Ridge National Laboratory | Academic use with registration |

> **Note:** Raw data are **not** distributed in this repository due to licensing and size constraints. The download scripts will fetch them automatically from their respective sources (with user registration required for LandScan).

---

## Repository Structure

```
.
├── data/                     # Data directories (created at runtime)
│   ├── raw/                  # Raw downloaded files (not committed)
│   └── processed/            # Processed panel dataset (not committed)
├── outputs/                  # All generated outputs
│   ├── figures/              # Maps and time‑series plots
│   ├── results/              # PAI summaries and model outputs
│   └── models/               # Trained LSTM model weights
├── src/                      # Source code
│   ├── config.py             # Central configuration (paths, parameters)
│   ├── download_police.py    # Downloads crime data with 503‑adaptive retry
│   ├── download_geodata.py   # Downloads boundaries, population, boroughs
│   ├── preprocess.py         # Spatial join, aggregation, feature engineering
│   ├── esda.py               # Moran's I, LISA, hotspot maps (professional cartography)
│   ├── kde_baseline.py       # KDE model with adaptive bandwidth
│   ├── lstm_model.py         # LSTM spatio‑temporal predictor (PyTorch)
│   ├── evaluation.py         # PAI comparison and plots
│   ├── evaluate_by_crime_type.py # Crime‑type‑specific evaluation
│   └── main.py               # Orchestrates the full pipeline
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from version control
└── README.md                 # This file
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/street-crime-spatial-prediction-london.git
cd street-crime-spatial-prediction-london
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU support:** For faster LSTM training, install a CUDA‑compatible version of PyTorch (see [pytorch.org](https://pytorch.org) for instructions). The code automatically detects and uses GPU if available.

---

## Usage

### Quick start (full pipeline)
```bash
python src/main.py --stage all
```
This will:
- Download all required data (crime, boundaries, population)
- Preprocess and build the panel dataset
- Perform ESDA (temporal plots, Moran’s I, LISA, hotspot maps)
- Train and evaluate the KDE and LSTM models
- Generate comparison plots and PAI summaries

### Run individual stages
```bash
python src/main.py --stage download   # Data acquisition only
python src/main.py --stage preprocess # Build panel dataset
python src/main.py --stage esda       # Exploratory spatial analysis
python src/main.py --stage model      # Train KDE and LSTM models
```

### Run per‑crime‑type evaluation
```bash
python -m src.evaluate_by_crime_type
```
This produces separate PAI results for pickpocketing, street robbery, public order, and all‑crime baseline.

### Manchester comparative analysis (experimental)
```bash
python src/main.py --stage all --manchester
```
Requires additional configuration – see `config.py` for details.

---

## Configuration

All modifiable parameters are centralised in `src/config.py`:
- Study period (`START_YEAR`, `END_YEAR`)
- London bounding box and grid division
- LSTM hyperparameters (sequence length, hidden size, learning rate, early stopping)
- KDE bandwidth and grid resolution
- Hotspot area proportion (`HOTSPOT_PCT`)
- Paths to data and output directories

Adjust these values before running the pipeline if you wish to change the study area or model settings.

---

## Outputs

After a successful run, the following files are generated in `outputs/`:

| File | Description |
|------|-------------|
| `figures/temporal_trends.png` | Monthly crime count and density trends |
| `figures/morans_i_series_*.png` | Monthly global Moran’s I for all crimes and each street crime type |
| `figures/hotspot_*.png` | Crime density hotspot maps (choropleth) with borough overlays |
| `figures/lisa_*.png` | LISA cluster maps (HH, LL, HL, LH) |
| `figures/crime_type_comparison_*.png` | 2×2 comparison of four street crime types |
| `figures/pai_comparison.png` | PAI time‑series and boxplot for KDE vs LSTM |
| `results/pai_summary.csv` | Summary statistics of PAI |
| `results/kde_results.parquet` | Monthly PAI results for KDE |
| `results/lstm_results.parquet` | Monthly PAI results for LSTM |
| `models/lstm_best.pt` | Best‑performing LSTM model weights |

All maps include professional cartographic elements: scale bar, north arrow, legend, data credits, and borough boundaries.

---

## Reproducibility

To fully reproduce the results, you will need:
- An active internet connection for data downloads
- A LandScan account (free for academic users) – place the downloaded GeoTIFF in `data/raw/population/` or let the script guide you
- Approximately 30–50 minutes for the crime data download (adaptive 503 handling may add time)
- A machine with at least 8 GB RAM (16 GB recommended for LSTM training)

All random seeds are fixed in `config.py` (`RANDOM_SEED = 42`). The code is designed to be deterministic (where possible) to facilitate replication.

---

## Dependencies

Key libraries are listed in `requirements.txt`. The main ones are:

- `pandas`, `numpy`, `geopandas` – data manipulation and spatial operations
- `matplotlib`, `seaborn` – visualisation
- `scipy`, `scikit‑learn` – KDE and scaling
- `torch` – LSTM implementation
- `esda`, `libpysal` – spatial autocorrelation and weights
- `requests`, `tqdm` – data downloading and progress bars

---

## Citation

If you use this code in your research, please cite the associated dissertation:

> Pan, R. (2026). *Spatiotemporal Crime Hotspot Change Detection Using Machine Learning: A Dynamic Risk Assessment Framework*. MSc dissertation, University College London.

For the software itself, please cite the Zenodo DOI (when available):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22032898.svg)](https://doi.org/10.5281/zenodo.22032898)

---

## Licence

This project is licensed under the MIT Licence – see the [LICENCE](LICENCE) file for details. The data remain under their respective licences (OGL v3.0 and LandScan academic use).

---

## Acknowledgements

- Crime data are provided by the UK Home Office via data.police.uk under OGL v3.0.
- Boundary and population data are from the Office for National Statistics and Ordnance Survey, also under OGL v3.0.
- LandScan data are from Oak Ridge National Laboratory and used with permission for academic research.
- This work was supervised by Dr Yi Dong at UCL Department of Security and Crime Science.

---

## Contact

For questions, please raise an issue on this repository or contact the author directly via UCL email (rongze.pan.24@ucl.ac.uk).

---

*Last updated: August 2026*
