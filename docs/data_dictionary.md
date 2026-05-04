# Data dictionary

This document describes the sample dataset provided in the `data/sample/` folder.

The sample dataset is an anonymized station-level time-series dataset at 20-minute resolution. Each row represents one 20-minute interval for one anonymized public fast-charging station.

The sample data are provided for workflow demonstration, code testing and documentation purposes. They are not intended to represent the full study dataset.

## Sample data file

Expected location:

```text
data/sample/
```

Example file name:

```text
sample_station_6months.xlsx
```

## Dataset summary

The current sample file contains:

| Item | Value |
|---|---:|
| Number of rows | 13,104 |
| Number of columns | 42 |
| Temporal resolution | 20 minutes |
| Time range | 2024-12-31 00:00:00 to 2025-06-30 23:40:00 |
| Missing values | None |
| Duplicated columns | None |

## Data structure

Each row corresponds to one 20-minute station-time interval.

| Order | Variable | Type | Description |
|---:|---|---:|---|
| 1 | `time` | datetime | Timestamp of the 20-minute interval. |
| 2 | `module_demand` | integer | Required number of charging modules in the interval. This is the main module-demand target. |
| 3 | `active_vehicle_count` | integer | Number of vehicles actively charging in the interval. |
| 4 | `SFC_SW` | float | Surface shortwave radiation or related solar-radiation variable. |
| 5 | `T2M` | float | Air temperature at 2 m. |
| 6 | `QV2M` | float | Specific humidity at 2 m. |
| 7 | `RH2M` | float | Relative humidity at 2 m. |
| 8 | `PS` | float | Surface pressure. |
| 9 | `WS10M` | float | Wind speed at 10 m. |
| 10 | `electricity_price` | float | Electricity price or tariff-related feature. |
| 11 | `start_model0_soc0` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 0. |
| 12 | `start_model0_soc1` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 1. |
| 13 | `start_model0_soc2` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 2. |
| 14 | `start_model0_soc3` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 3. |
| 15 | `start_model0_soc4` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 4. |
| 16 | `start_model1_soc0` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 0. |
| 17 | `start_model1_soc1` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 1. |
| 18 | `start_model1_soc2` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 2. |
| 19 | `start_model1_soc3` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 3. |
| 20 | `start_model1_soc4` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 4. |
| 21 | `start_model2_soc0` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 0. |
| 22 | `start_model2_soc1` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 1. |
| 23 | `start_model2_soc2` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 2. |
| 24 | `start_model2_soc3` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 3. |
| 25 | `start_model2_soc4` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 4. |
| 26 | `start_soc0` | integer | Number of vehicles or sessions in start-of-interval SOC bin 0. |
| 27 | `start_soc1` | integer | Number of vehicles or sessions in start-of-interval SOC bin 1. |
| 28 | `start_soc2` | integer | Number of vehicles or sessions in start-of-interval SOC bin 2. |
| 29 | `start_soc3` | integer | Number of vehicles or sessions in start-of-interval SOC bin 3. |
| 30 | `start_soc4` | integer | Number of vehicles or sessions in start-of-interval SOC bin 4. |
| 31 | `model0` | integer | Number of vehicles or sessions assigned to charging-power class 0. |
| 32 | `model1` | integer | Number of vehicles or sessions assigned to charging-power class 1. |
| 33 | `model2` | integer | Number of vehicles or sessions assigned to charging-power class 2. |
| 34 | `arriving_vehicle_count` | integer | Number of newly arriving vehicles in the interval. |
| 35 | `aggregate_power_kw` | float | Aggregate charging power in the interval, measured in kW. |
| 36 | `weekday` | integer | Day-of-week indicator. |
| 37 | `raw_hour` | integer | Hour component derived from the timestamp. |
| 38 | `raw_minute` | integer | Minute component derived from the timestamp. |
| 39 | `day_type` | integer | Numeric day-type code used for model input. |
| 40 | `day_type_label` | string | English day-type label, such as `Weekday`, `Weekend`, `Statutory Holiday` or `Adjusted Workday`. |
| 41 | `station_id_anonymized` | string | Anonymized station identifier. In the sample data, the station is represented as `S001`. |
| 42 | `station_name_anonymized` | string | Anonymized station name. In the sample data, this is represented as `sample_station`. |

## Variable groups

### Time and calendar variables

```text
time
weekday
raw_hour
raw_minute
day_type
day_type_label
```

`time` records the 20-minute timestamp. The current sample covers the period from 2024-12-31 00:00:00 to 2025-06-30 23:40:00.

`day_type` is the numeric code used by the modelling workflow. `day_type_label` is the human-readable English label.

Current `day_type_label` values are:

```text
Weekday
Statutory Holiday
Weekend
Adjusted Workday
```

### Charging demand and station-operation variables

```text
module_demand
active_vehicle_count
arriving_vehicle_count
aggregate_power_kw
electricity_price
```

`module_demand` is the main prediction target. It represents the required number of charging modules in each 20-minute interval.

`aggregate_power_kw` represents total station charging power in kW.

`electricity_price` is retained as a future-known exogenous feature for the forecasting workflow.

### Weather and environmental variables

```text
SFC_SW
T2M
QV2M
RH2M
PS
WS10M
```

These variables describe weather or environmental conditions associated with each 20-minute station-time interval.

### Charging-power class variables

```text
model0
model1
model2
```

`model0`, `model1` and `model2` are charging-power class labels. They represent aggregate counts of vehicles or sessions belonging to different charging-power classes in each 20-minute interval.

### SOC distribution variables

The following variables describe aggregate start-of-interval state-of-charge distributions:

```text
start_soc0
start_soc1
start_soc2
start_soc3
start_soc4
```

The following variables describe joint distributions by charging-power class and SOC bin:

```text
start_model0_soc0
start_model0_soc1
start_model0_soc2
start_model0_soc3
start_model0_soc4

start_model1_soc0
start_model1_soc1
start_model1_soc2
start_model1_soc3
start_model1_soc4

start_model2_soc0
start_model2_soc1
start_model2_soc2
start_model2_soc3
start_model2_soc4
```

For example, `start_model1_soc3` denotes the number of vehicles or sessions in charging-power class `model1` and SOC bin 3 at the start of the interval.

### Anonymized station variables

```text
station_id_anonymized
station_name_anonymized
```

The sample station is represented as:

```text
station_id_anonymized = S001
station_name_anonymized = sample_station
```

These identifiers are provided only to preserve the station-level data structure. They do not reveal the original station code, station name, operator or exact location.

## Compatibility notes

The public sample data use English, human-readable variable names. The original internal modelling code may use legacy variable names, such as:

```text
num_modular
num_charging
num_coming
Power_kW
electric_price
hour
minute
```

These legacy variables are not included in the public sample file. When running the provided scripts, the preprocessing utilities map the public variable names to the internal names required by the modelling pipeline.

## Privacy notes

The sample data do not include raw transaction-level charging records, user identifiers, vehicle identifiers, VINs, license-plate information, payment records, original station names, original station codes, operator identifiers or exact station coordinates.

The sample data are intended only for demonstrating the data structure and testing the analysis workflow.