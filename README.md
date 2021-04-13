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



Installation
---------------

This project run on python >= 3.6

1. **Install Dependencies**

    ```
    # 1. Clone the project
    git clone https://github.com/georgehua/energy-consumption-prediction.git

    # 2. Install project dependencies
    
    # It's recommended to create a virtual environment first, a good option is to use pipenv package
    pip install pipenv
    pipenv shell
    pipenv install -r requirements.txt

    # Or you can install everything gloabally
    pip install requirements.txt
    ```

2. **The Dataset**

   The raw data has to be placed in `data/raw`. A good practice is to download the data via the Kaggle CLI.
    ```
    kaggle competitions download -c ashrae-energy-prediction
    mkdir -p data/raw
    unzip ashrae-energy-prediction.zip -d data/raw
    ```

3. **Data Cleaning and Preprocessing**

    ```
    python src/data/preproc.py data data/interim
    ```

   `preproc.py` will create a clean dataframe for analysis. This script includes: 
   - Load all associated `.csv`-files
   - Localize timestamps based on building location. 
   - Iterative imputation for missing values in weather data. 
   - Join all dataframes and save compressed results in `data/interim`


4. **Feature Engineering**

    ```
    python src/features/build_features.py data data/processed
    ```
    `build_features.py` will conduct the feature engineering process. The result is saved in `data/processed`. This script includes: 
    - Encode all categorical features
    - Encode wind direction (cyclic encoding - cosine)
        - Inspired by: https://stats.stackexchange.com/questions/148380/use-of-circular-predictors-in-linear-regression
    - Extract time based features (time-series pattern on energy usage)
        - hour
        - weekday
        - month
    - Create new features based on existing information: 
        - area_per_floor
        - log_transformed_square_feet
        - log_transform_area_per_floor
        - age_of_buildings
        - relative_humidity
        - feels_like_temperature
        - add lag windows 6, 24 days
    - Labeling outliers: var < mean - 2.5 * std | var > mean + - 2.5 * std
    - Exlucde wrong readings (data anomly)
    - Drop faulty rows
        - Credit for: https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks



5. **Build Models**

    ```
    python src/models/train_model.py <MODEL_NAME> <MODE> data/processed models/<MODEL_NAME>
    
    # Example:
    python src/models/train_model.py lgbm cv data/processed models/lgbm
    ```
    In this script I set up the training program for three ML models LightGBM, CatBoost and XGBoost. From the experiments, the LightGBM renders the best results. At the end of each training, the models will be safed in the equally named directory.
    - `MODEL_NAME` : 
        - lgbm (LightGBM)
        - ctb (CatBoost)
        - xgb (XGBoost)
    - `MODE`: 
        - cv (cross validation, default, best one)
        - full (no validation set, single fold)
        - by_meter (training by meter type)
        - by_building (training by building id)
    

6. **Hyperparameter Tuning**
   
   ```
    python src/models/find_hyperparameter_lgbm.py
    ```
   For tuning hyperparameter, I used bayesian optimization to reduce the time spent on traditional grid-search or random-search.
   Note: This step is time-consuming and requires large RAM space.
   

6. **Make Predictions**

   ```
    python src/models/predict_model.py data/processed <MODEL> <MODEL_PATH> submissions/submission.csv

    # Example:
    python src/models/predict_model.py data/processed lgbm models/lgbm_cv/ submissions/submission.csv
    ```
   The easiest way is to use `make predict MODEL_PATH=<modelpath> MODEL=<model>` where `MODEL_PATH` should point to the directory of the saved models or the model itself. The `MODEL` parameter describes the framework of the model equivalent to the step above. The result is a `.csv` file, which is dumped in the `submission` directory and is ready for uploading to Kaggle. An importen flag is whether to use leaks or not as it heavily influences the resulting submission file.


Note: Data Leaks
------------

Unfortunately a portion of the test labels have been leaked, which stirred the whole competition. If you want to use the leaks for your own experiments, you have the set the respective flags in the config file. Additionally the leaks have to be downloaded from [here](https://www.kaggle.com/yamsam/ashrae-leak-data-station) and be placed in `./data/leaks`.