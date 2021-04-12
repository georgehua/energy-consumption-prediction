# Energy Consumption Prediction


## The Dataset
---------------

The provided data consists of ~20 mio. rows for training (one year timespan) and ~40 mio. rows for testing (two years timespan). The target variable are the hourly readings from one of four meters {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}. For building the model the data provides following features out of the box:


- building_id --> Foreign key for the building metadata.
- meter ---> The meter id code. Read as {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}
- timestamp --> Hour of the measurement
- site_id --> Identifier of the site the building is on
- primary_use ---> Primary category of activities for the building 
- square_feet --> Floor area of the building
- year_built ---> build year of the building
- floorcount - Number of floors of the building

Further weather data has been provided, which comes with air_temperature, cloud_coverage, dew_temperature, precip_depth_1_hr, sea_level_pressure, wind_direction and wind_speed.

Exploratory Data Analysis

Profiling for each data file:

- building_metadata
- train
- weather

In-depth analysis (missing data, outliers, time-series trend, correlation)
