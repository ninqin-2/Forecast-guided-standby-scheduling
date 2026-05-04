
# Data access

This document describes the data-access scope for the repository associated with the manuscript:

**Forecast-guided standby scheduling for modular public fast-charging stations**

This repository is intended to support reproducibility, documentation and workflow demonstration. It contains figure-level source data, sample data and code-related documentation. It does not contain raw transaction-level charging records.

## Data included in this repository

The repository currently includes the following data-related materials.

### Figure-level source data

The `data/source_data/` folder contains figure-level source data underlying the main and supplementary figures. These files include the numerical values used to generate the published plots after station-level anonymization and aggregation.

### Sample data

The `data/sample/` folder contains a small anonymized station-level sample dataset. The sample data are provided to illustrate the expected data structure and to support testing of the example scripts.

The sample dataset is processed at 20-minute resolution and uses anonymized station identifiers and relative temporal indices. It is not intended to represent the full study dataset.

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

The source data underlying the main and supplementary figures are provided in this repository or with the associated manuscript.

An anonymized station-level dataset, including 20-minute module demand, station-level features, forecasting outputs and scheduling results, has been prepared for peer review and will be deposited in a public repository with a persistent DOI upon publication.

Transaction-level charging records are not publicly released owing to user privacy, station-level commercial sensitivity and third-party data-use restrictions. Aggregated and anonymized data sufficient to reproduce the main analyses will be made publicly available upon publication.

## Recommended reproducibility path

The recommended reproducibility path starts from the anonymized station-level datasets and figure-level source data, rather than raw transaction-level charging records.

For figure reproduction, the workflow is:

```text
figure-level source data
        ↓
figure-generation scripts
        ↓
main and supplementary figures
