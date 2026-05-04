# Data access

This document describes the data-access scope for the repository associated with the manuscript:

**Forecast-guided standby scheduling for modular public fast-charging stations**

This repository is intended to support reproducibility, documentation and workflow demonstration. It contains anonymized sample data, code-related documentation and example scripts. It does not contain raw transaction-level charging records.

## Data included in this repository

The repository currently includes the following data-related materials.

### Sample data

The `data/sample/` folder contains a small anonymized station-level sample dataset:

```text
data/sample/sample_station_6months.xlsx
```

The sample dataset is processed at 20-minute resolution. Each row represents one 20-minute interval for one anonymized public fast-charging station.

The sample data are provided to illustrate the expected data structure and to support testing of the example scripts. They are not intended to represent the full study dataset.

The sample data include anonymized station identifiers and station-time-level variables, such as module demand, active vehicle count, arriving vehicle count, aggregate charging power, electricity price, weather variables, charging-power class counts and state-of-charge distribution features.

The sample data do not include raw transaction-level charging records, user identifiers, vehicle identifiers, VINs, license-plate information, payment records, original station names, original station codes, operator identifiers or exact station coordinates.

### Figure-level source data

The `data/source_data/` folder is reserved for figure-level source data underlying the main and supplementary figures.

These files will include the numerical values used to generate the published plots after station-level anonymization and aggregation. Transaction-level charging records and sensitive station/user information will not be included.

## Data prepared for public release

An anonymized station-level dataset has been prepared to support reproducibility of the main analyses. The full public dataset will include:

1. Anonymized station-level metadata.
2. 20-minute module-demand time series.
3. 20-minute station-level operational features.
4. Forecasting outputs.
5. Standby-scheduling simulation results.
6. Station-level energy-saving and service-risk metrics.
7. Figure-level source data underlying the main and supplementary figures.

Upon publication, the anonymized station-level dataset and figure-level source data will be deposited in a public data repository with a persistent DOI.

Repository DOI: `to be added upon publication`

## Data not publicly released

The following data are not publicly released:

1. Raw transaction-level charging records.
2. User identifiers.
3. Vehicle identifiers.
4. VINs or license-plate-related information.
5. Payment records.
6. Account-level discount or billing records.
7. Precise user trajectories.
8. Original station names.
9. Raw operator identifiers.
10. Exact station coordinates or commercially sensitive location information.
11. Raw station-level commercial records.

These data are withheld to protect user privacy, vehicle privacy, station-level commercial information and third-party data-use restrictions.

## Data-access statement

The source data underlying the main and supplementary figures are provided with the associated manuscript and will be archived with the public data release.

An anonymized station-level dataset, including 20-minute module demand, station-level features, forecasting outputs and scheduling results, has been prepared for peer review and will be deposited in a public repository with a persistent DOI upon publication.

Transaction-level charging records are not publicly released owing to user privacy, station-level commercial sensitivity and third-party data-use restrictions. Aggregated and anonymized data sufficient to reproduce the main analyses will be made publicly available upon publication.

## Recommended reproducibility path

The recommended reproducibility path starts from anonymized station-level data and figure-level source data, rather than raw transaction-level charging records.

For the sample workflow included in this repository, the workflow is:

```text
anonymized 20-minute sample data
        ↓
module-demand forecasting scripts
        ↓
multi-step prediction and uncertainty adjustment
        ↓
sample prediction outputs
```

For figure reproduction after source-data release, the workflow is:

```text
figure-level source data
        ↓
figure-generation scripts
        ↓
main and supplementary figures
```

For full analysis reproduction after public data release, the workflow is:

```text
anonymized station-level data
        ↓
forecasting outputs
        ↓
standby-scheduling simulation results
        ↓
energy-saving and service-risk metrics
        ↓
figures and tables
```

## Notes on the sample dataset

The sample dataset is designed for code testing and workflow demonstration. It is not a substitute for the full study dataset.

The sample file uses English variable names and anonymized station identifiers. Legacy variable names required by the internal modelling pipeline are handled by the preprocessing utilities in the scripts.

Detailed variable definitions are provided in:

```text
docs/data_dictionary.md
```

Privacy and anonymization principles are described in:

```text
docs/privacy_anonymization.md
```

## Contact

For questions about data access, please contact:

`Corresponding author name and email to be added`