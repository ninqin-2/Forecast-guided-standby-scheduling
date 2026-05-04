# Data dictionary

This document describes the sample dataset provided in the `data/sample/` folder.

The sample dataset is an anonymized station-level time-series dataset at 20-minute resolution. Each row represents one 20-minute interval for one anonymized public fast-charging station.

The sample data are provided for workflow demonstration, code testing and documentation purposes. They are not intended to represent the full study dataset.

## Sample data file

Expected location:

```text
data/sample/
```

Example file names:

```text
sample_station_S001_3months.csv
sample_station_S001_3months.xlsx
```

## Data structure

Each row corresponds to one 20-minute station-time interval.

| Variable | Type | Description |
|---|---:|---|
| `station_id_anonymized` | string | Anonymized station identifier. In the sample data, the station is represented as `S001`. |
| `interval_index` | integer | 20-minute interval index within a day. Values typically range from 0 to 71. |
| `hour_of_day` | integer | Hour of day derived from the 20-minute interval. |
| `day_type` | string | English day-type label, such as `Weekday`, `Weekend`, `Statutory Holiday` or `Adjusted Workday`. |
| `weekday` | integer | Day-of-week indicator. |
| `module_demand` | integer | Required number of charging modules in the interval. This is the main module-demand target. |
| `active_vehicle_count` | integer | Number of vehicles actively charging in the interval. |
| `arriving_vehicle_count` | integer | Number of newly arriving vehicles in the interval. |
| `aggregate_power_kw` | float | Aggregate charging power in the interval, measured in kW. |
| `electricity_price` | float | Electricity price or tariff-related feature. |
| `time` | datetime | Timestamp of the 20-minute interval in the sample file. |
| `SFC_SW` | float | Surface shortwave radiation or related solar-radiation variable. |
| `T2M` | float | Air temperature at 2 m. |
| `QV2M` | float | Specific humidity at 2 m. |
| `RH2M` | float | Relative humidity at 2 m. |
| `PS` | float | Surface pressure. |
| `WS10M` | float | Wind speed at 10 m. |
| `start_model0_soc0` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 0. |
| `start_model0_soc1` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 1. |
| `start_model0_soc2` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 2. |
| `start_model0_soc3` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 3. |
| `start_model0_soc4` | integer | Number of vehicles or sessions in charging-power class `model0` and start-of-interval SOC bin 4. |
| `start_model1_soc0` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 0. |
| `start_model1_soc1` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 1. |
| `start_model1_soc2` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 2. |
| `start_model1_soc3` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 3. |
| `start_model1_soc4` | integer | Number of vehicles or sessions in charging-power class `model1` and start-of-interval SOC bin 4. |
| `start_model2_soc0` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 0. |
| `start_model2_soc1` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 1. |
| `start_model2_soc2` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 2. |
| `start_model2_soc3` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 3. |
| `start_model2_soc4` | integer | Number of vehicles or sessions in charging-power class `model2` and start-of-interval SOC bin 4. |
| `start_soc0` | integer | Number of vehicles or sessions in start-of-interval SOC bin 0. |
| `start_soc1` | integer | Number of vehicles or sessions in start-of-interval SOC bin 1. |
| `start_soc2` | integer | Number of vehicles or sessions in start-of-interval SOC bin 2. |
| `start_soc3` | integer | Number of vehicles or sessions in start-of-interval SOC bin 3. |
| `start_soc4` | integer | Number of vehicles or sessions in start-of-interval SOC bin 4. |
| `model0` | integer | Number of vehicles or sessions assigned to charging-power class 0. |
| `model1` | integer | Number of vehicles or sessions assigned to charging-power class 1. |
| `model2` | integer | Number of vehicles or sessions assigned to charging-power class 2. |
| `raw_hour` | integer | Hour component derived from the original timestamp. |
| `raw_minute` | integer | Minute component derived from the original timestamp. |

## Variable groups

### Station and time variables

The following variables describe the anonymized station and the 20-minute time interval:

```text
station_id_anonymized
time
interval_index
hour_of_day
weekday
day_type
raw_hour
raw_minute
```

### Charging demand variables

The following variables describe station-level charging demand and charging activity:

```text
module_demand
active_vehicle_count
arriving_vehicle_count
aggregate_power_kw
electricity_price
```

### Weather variables

The following variables are weather or environmental features:

```text
SFC_SW
T2M
QV2M
RH2M
PS
WS10M
```

### Charging-power class variables

The variables below describe aggregate charging-power classes:

```text
model0
model1
model2
```

Here, `model0`, `model1` and `model2` are charging-power class labels. They represent aggregate counts of vehicles or sessions belonging to different charging-power classes in each 20-minute interval.

### SOC distribution variables

The variables below describe aggregate start-of-interval state-of-charge distributions:

```text
start_soc0
start_soc1
start_soc2
start_soc3
start_soc4
```

The variables below describe joint distributions by charging-power class and SOC bin:

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

## Privacy notes

The sample data do not include raw transaction-level charging records, user identifiers, vehicle identifiers, VINs, license-plate information, payment records, original station names or operator identifiers.

The station is represented by an anonymized station identifier:

```text
S001
```

The sample data are intended only for demonstrating the data structure and testing the analysis workflow.
