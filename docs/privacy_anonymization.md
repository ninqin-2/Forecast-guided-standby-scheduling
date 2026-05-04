
# Privacy and anonymization

This document describes the privacy protection and anonymization principles used for the public materials in this repository.

The repository is associated with the manuscript:

**Forecast-guided standby scheduling for modular public fast-charging stations**

The study uses operational data from public fast-charging stations. The public repository is designed to support workflow demonstration and reproducibility while protecting user privacy, vehicle privacy and station-level commercial information.

## Public data included in this repository

The current repository includes an anonymized sample dataset:

```text
data/sample/sample_station_6months.xlsx
```

The sample dataset is processed at 20-minute resolution. Each row represents one 20-minute interval for one anonymized public fast-charging station.

The sample data include station-time-level variables such as:

```text
module_demand
active_vehicle_count
arriving_vehicle_count
aggregate_power_kw
electricity_price
weather variables
charging-power class counts
state-of-charge distribution features
```

The sample data are intended for code testing, workflow demonstration and documentation purposes. They are not intended to represent the full study dataset.

## Data not included in the public release

The public repository does not include:

1. Raw transaction-level charging records.
2. User identifiers.
3. Vehicle identifiers.
4. VINs or license-plate-related information.
5. Payment records.
6. Account-level billing records.
7. Membership or discount account records.
8. Original station names.
9. Original station codes.
10. Operator identifiers.
11. Exact station coordinates.
12. User trajectories.
13. Raw station-level commercial records.

These data are withheld to protect user privacy, vehicle privacy, station-level commercial sensitivity and third-party data-use restrictions.

## Station anonymization

Original station identifiers are replaced with anonymized identifiers.

In the sample dataset, the station is represented as:

```text
station_id_anonymized = S001
station_name_anonymized = sample_station
```

These identifiers are used only to preserve the station-level data structure. They do not reveal the original station code, station name, operator or exact location.

## Temporal information

The sample dataset retains 20-minute timestamps to support sequence construction and workflow testing.

The current sample covers:

```text
2024-12-31 00:00:00 to 2025-06-30 23:40:00
```

The timestamps are provided only for demonstrating the forecasting workflow on station-time-level data. The sample file does not contain user-level charging trajectories or transaction-level event records.

If a more restrictive public release is required, timestamps can be replaced by relative temporal indices, such as:

```text
day_index
interval_index
```

## Spatial anonymization

Exact station coordinates are not included.

The public sample does not contain:

```text
longitude
latitude
address
original station name
operator identifier
```

Station context is represented only through anonymized station identifiers and station-time-level operational variables.

## User and vehicle privacy

User- and vehicle-level identifiers are removed before public release.

The public data do not include:

```text
user_id
vehicle_id
VIN
license_plate
phone number
payment account
membership account
```

Where user or vehicle information is analytically relevant, it is transformed into station-time-level aggregate features, such as counts or binned distributions. Individual-level records are not released.

## Aggregation level

The public sample data are aggregated at the station-time level.

The temporal resolution is:

```text
20 minutes
```

This aggregation level supports testing of module-demand forecasting and standby-scheduling workflows while avoiding exposure of individual charging sessions.

## Charging-power and SOC features

The sample data include aggregate charging-power class and state-of-charge distribution features, such as:

```text
model0
model1
model2
start_soc0
start_soc1
start_soc2
start_soc3
start_soc4
start_model0_soc0
...
start_model2_soc4
```

These variables are aggregated counts or binned features at the station-time level. They do not identify individual users or vehicles.

## Day-type information

Day-type information is retained as a general calendar feature.

The sample data include:

```text
day_type
day_type_label
```

`day_type` is a numeric code used by the modelling workflow. `day_type_label` is a human-readable English label, such as:

```text
Weekday
Weekend
Statutory Holiday
Adjusted Workday
```

These labels do not contain individual-level information.

## Commercially sensitive information

The public repository does not include raw operator-level records, payment records, discount contracts, account-level billing information or other commercially sensitive station-management records.

When pricing or operational features are included, they are provided only as station-time-level variables or transformed features suitable for workflow demonstration.

## Figure-level source data

The `data/source_data/` folder is reserved for figure-level source data underlying the main and supplementary figures.

These files will contain numerical values used to generate the published plots after station-level anonymization and aggregation. They will not include raw transaction-level records or sensitive user, vehicle or station identifiers.

## Intended use

The released materials are intended for:

1. Demonstrating the data structure used in the study.
2. Testing the provided forecasting and prediction scripts.
3. Supporting reproducibility of figure-level and station-level analyses after public data release.
4. Enabling methodological comparison for station-level fast-charging operation research.

The released materials are not intended for:

1. Reconstructing individual charging sessions.
2. Identifying specific users or vehicles.
3. Identifying the original charging station.
4. Inferring commercially sensitive operator information.

## Limitations of the public sample

The sample dataset is provided for workflow demonstration and code testing. It is not a substitute for the full study dataset.

Some analyses requiring raw transaction-level records, exact original station identifiers, detailed billing records or full-network station coverage cannot be reproduced directly from the sample file.

The full anonymized station-level dataset and figure-level source data will be deposited in a public repository with a persistent DOI upon publication.

## Access to restricted data

Access to restricted raw data is not guaranteed. Any request for restricted data would require approval from the data provider and may be subject to a data-use agreement, privacy review, ethical review or institutional approval.

## Contact

For questions about data access or anonymization, please contact:

```text
Corresponding author name and email to be added
```