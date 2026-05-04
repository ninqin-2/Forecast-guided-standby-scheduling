# Forecast-guided standby scheduling for modular public fast-charging stations

This repository provides reproducibility materials, sample data and example scripts for the manuscript:

**Forecast-guided standby scheduling for modular public fast-charging stations**

The study investigates whether short-term module-demand forecasting can support standby scheduling in modular public DC fast-charging stations, reducing standby energy loss while maintaining service reliability.

## Repository scope

This repository is intended as a reproducibility and documentation repository. It contains:

1. An anonymized sample dataset at 20-minute resolution.
2. Documentation for data access, data structure and anonymization.
3. Example scripts for module-demand forecasting.
4. Example scripts for multi-step prediction and uncertainty adjustment.
5. Placeholder folders for figure-level source data and future reproducibility materials.

The repository does **not** contain raw transaction-level charging records, user identifiers, vehicle identifiers, VINs, payment records, original station names, exact station coordinates or other sensitive operational records.

## Repository structure

```text
forecast-guided-standby-scheduling/
├── data/
│   ├── sample/
│   │   └── sample_station_6months.xlsx
│   └── source_data/
│
├── docs/
│   ├── data_access.md
│   ├── data_dictionary.md
│   └── privacy_anonymization.md
│
├── scripts/
│   ├── RunMainfor1Hour_sample.py
│   ├── RunProjectModules.py
│   ├── station_list.xlsx
│   ├── Utils/
│   └── models/
│   └── ...
│
├── .gitignore
├── LICENSE
└── README.md
```

## Data

### Sample data

The sample dataset is located at:

```text
data/sample/sample_station_6months.xlsx
```

It contains anonymized station-level time-series data at 20-minute resolution. Each row represents one 20-minute interval for one anonymized public fast-charging station.

The sample data are used for workflow demonstration and code testing. They are not intended to represent the full study dataset.

### Source data

The `data/source_data/` folder is reserved for figure-level source data underlying the main and supplementary figures. These files include numerical values used to generate the published plots after station-level anonymization and aggregation.

### Data restrictions

Raw transaction-level charging records are not publicly released owing to user privacy, station-level commercial sensitivity and third-party data-use restrictions.

More details are provided in:

```text
docs/data_access.md
docs/data_dictionary.md
docs/privacy_anonymization.md
```

## Workflow overview

The analysis workflow consists of four main components:

```text
20-minute station-level data
        ↓
module-demand forecasting
        ↓
multi-step uncertainty adjustment
        ↓
standby-scheduling evaluation
        ↓
energy-saving and service-risk metrics
```

### 1. Module-demand forecasting

The forecasting model predicts short-term module demand directly at the station level. The sample script demonstrates a PatchTST-based forecasting workflow for a 1-hour horizon.

Example script:

```text
scripts/RunMainfor1Hour_sample.py
```

### 2. Multi-step prediction and uncertainty adjustment

The multi-step workflow predicts module demand for multiple 20-minute horizons and combines prediction outputs for uncertainty-aware adjustment.

Example script:

```text
scripts/RunProjectModules.py
```

### 3. Standby-scheduling evaluation

The full study evaluates how forecast-guided activation of charging modules affects standby energy loss and service reliability. The public repository currently provides the sample forecasting and prediction workflow. Additional scheduling-evaluation scripts and figure-level source data will be added as the reproducibility package is finalized.

## Quick start

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/forecast-guided-standby-scheduling.git
cd forecast-guided-standby-scheduling
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the 1-hour sample forecasting workflow:

```bash
python scripts/RunMainfor1Hour_sample.py
```

Run the multi-step prediction workflow:

```bash
python scripts/RunProjectModules.py
```

The generated outputs are saved under:

```text
scripts/PredictResult/
scripts/Model_State_dict/
scripts/Figures/
scripts/outputs/
```


## Notes on file paths

The sample scripts are designed to run within this repository structure. Input data are read from:

```text
data/sample/
```

Running outputs are written to:

```text
scripts/
```

The scripts do not require access to the original non-anonymized station data.

## Reproducibility status

This repository currently provides a sample-level workflow demonstration. The complete anonymized station-level dataset and figure-level source data supporting the main analyses will be deposited in a public data repository with a persistent DOI upon publication.

## Citation

If you use this repository, please cite the associated manuscript and the archived repository release.

```bibtex
@article{author_year_forecast_guided_standby,
  title = {Forecast-guided standby scheduling for modular public fast-charging stations},
  author = {Author, A. and Author, B.},
  journal = {To be added},
  year = {To be added},
  doi = {To be added}
}
```

## License

The code is released under the license specified in the `LICENSE` file.

The sample data and figure-level source data are provided for academic reproducibility and documentation purposes. Data-use conditions may be updated upon publication according to the final public data repository release.

## Contact

For questions about the code or data-access conditions, please contact:

```text
Corresponding author name and email to be added
```