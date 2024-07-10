# Predicting Used Car Market Values with Machine Learning

## Introduction

Rusty Bargain, a leading used car sales service, is developing a new mobile application aimed at attracting prospective customers. The app's primary feature is the ability to determine the market value of a user's car. To achieve this, the project requires building a machine learning model capable of accurately predicting car values based on historical data encompassing technical specifications, trim versions, and past prices.

Rusty Bargain is interested in:

- Prediction Quality: Develop a model that reliably estimates car market values, ensuring users receive accurate and dependable information.
- Training Efficiency: Streamline model training to minimize time and computational resources while maintaining high prediction accuracy.
- Prediction Speed: Optimize the prediction process for rapid valuation, enhancing user experience within the Rusty Bargain app.

[Notebook](/notebook.ipynb)

## Dataset Description

The dataset used in this project contains historical data of used cars, including their technical specifications, trim versions, and prices. This data is crucial for building a machine learning model to predict the market value of used cars accurately.

The dataset is stored in the file `/datasets/car_data.csv` and includes the following features:

- `DateCrawled`: The date when the profile was downloaded from the database.
- `VehicleType`: The type of vehicle body (e.g., sedan, SUV).
- `RegistrationYear`: The year the vehicle was registered.
- `Gearbox`: The type of gearbox (e.g., manual, automatic).
- `Power`: The power of the vehicle in horsepower (hp).
- `Model`: The model of the vehicle.
- `Mileage`: The mileage of the vehicle in kilometers.
- `RegistrationMonth`: The month the vehicle was registered.
- `FuelType`: The type of fuel the vehicle uses (e.g., petrol, diesel).
- `Brand`: The brand of the vehicle.
- `NotRepaired`: Indicates whether the vehicle has been repaired (`Yes` or `No`).
- `DateCreated`: The date when the profile was created.
- `NumberOfPictures`: The number of pictures of the vehicle.
- `PostalCode`: The postal code of the profile owner.
- `LastSeen`: The date of the last activity of the user.

**Target Variable:**

- `Price`: The price of the vehicle in Euros.

This comprehensive dataset provides all necessary details to build a robust model for predicting the market value of used cars.

## Methodology

### Data Cleaning and Preprocessing

- 262 rows of duplicate data removed.
- `DateCrawled`, `DateCreated`, `LastSeen` not included in the model features because they are unknown at the time of prediction once the model is in production.
- NaN values in the `Model`, `VehicleType`, `Gearbox`, and `NotRepaired` column were relabeled as 'unknown'.
- Due to limited computational resources, only the 200 most popular `Model` values are maintained, the rest are collapsed into the 'unknown' label. This cutoff can easily be changed by passing a different value to the `threshold` argument in the `consolidate_models` function should additional resources become available.
- Significant and impossible outliers in the `Power` column were uncovered. First, some basic research was conducted to determine realistic values for horsepower.

    > "A normal city car might have 90 horsepower, a family hatchback could have 140 horsepower, and midsize cars can have around 200 horsepower. Cars with more than 300 horsepower are usually performance vehicles, and modern supercars can have over 500 horsepower."

    A filtered df was created containing only the rows with `Power` values within the range (50, 600). The Average `Power` per brand, model, fueltype combination was then used to fill values outside the 50 - 600 range.
- The value 0 was determined to represent an 'unkown' month in the `RegistrationMonth` column as was relabeled as such. The column was labeled and treated as a categorical feature.
- Values below 1960 were dropped from the `RegistrationYear` column do to likely data input errors, and to keep analysis within a reasonable date range considering inflation.
- `PostalCode` was treated as categorical and tested for correlation with price. Significant correlation was discovered, but due to limited computational resources and concerns regarding data privacy, the column was dropped from the training features.
- 100% of the values in the `NumberOfPictures` where equal to zero, thus the column was not valuable as a training feature and was subsequently dropped.

### The Final Data Set

| Index | Column             | Non-Null Count | Dtype    |
|-------|--------------------|----------------|----------|
| 0     | Brand              | 320807         | category |
| 1     | Model              | 320807         | category |
| 2     | VehicleType        | 320807         | category |
| 3     | Gearbox            | 320807         | category |
| 4     | Power              | 320807         | int64    |
| 5     | Mileage            | 320807         | int64    |
| 6     | FuelType           | 320807         | category |
| 7     | RepairedStatus     | 320807         | category |
| 8     | RegistrationMonth  | 320807         | category |
| 9     | RegistrationYear   | 320807         | int64    |
| 10    | Price              | 320807         | int64    |

**dtypes:** category(7), int64(4)  
**memory usage:** 14.7 MB

## Model Tunning and Training

4 regression models were tunned within a scope appropriate to Rusty Bargain's computational resources and evaluated based on their RMSE scores for predicting price, as well as their wall clock times for training and prediction. A basic Linear Regression Model was used as the baseline for evaluating the four models.

### Random Forrest

The best Random Forrest parameters were determined according to the following function:

```python
def get_best_RF_params():

    # Define the params dict
    params_dict = {'max_depth': [None] + list(range(1, 21)),
                'min_samples_split': list(range(2, 21)),
                    'min_samples_leaf': list(range(1, 21)),
                    'n_estimators': list(range(10, 31, 10))}
    
    # Initialize the model
    model = RandomForestRegressor(random_state=123456)
    
    # Initialize the RandomizedSearchCV object
    random_search = RandomizedSearchCV(model, param_distributions=params_dict, n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=123456)
    
    # Fit the random search object
    random_search.fit(X_encoded_train, y_encoded_train)
    
    # Get the best parameters
    best_params = random_search.best_params_
    
    return best_params
```

`Best parameters: {'n_estimators': 30, 'min_samples_split': 2, 'min_samples_leaf': 9, 'max_depth': 18}`

### XGBoost

The best XGBoost parameters were determined according to the following function:

```python
def get_best_XGBR_params():
    
  # Define the params dict
  params_dict = {'max_depth': [6,12],
                'n_estimators': list(range(10, 31, 10)),
                'learning_rate': [0.1, 0.3]}
  
  # Initialize the model  
  model = XGBRegressor(random_state=123456)
  
  # Initialize the GridSearchCV object
  grid_search = GridSearchCV(model, param_grid=params_dict, scoring='neg_mean_squared_error', cv=3)
  
  # Fit the grid search object
  grid_search.fit(X_encoded_train, y_encoded_train)
  
  # Get the best parameters
  best_params = grid_search.best_params_
  
  return best_params
```

`Best parameters: {'learning_rate': 0.3, 'max_depth': 12, 'n_estimators': 30}`

### LightGBM

The best LightGBM parameters were determined according to the following function:

```python
def get_best_LGBM_params():
    
    # Define the params dict
    params_dict = {'max_depth': [6,12],
                   'n_estimators': list(range(10, 31, 10)),
                   'learning_rate': [0.1, 0.3]}
    
    # Initialize the model
    model = LGBMRegressor(random_state=123456)
    
    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(model, param_grid=params_dict, scoring='neg_mean_squared_error', cv=3)
    
    # Fit the grid search object
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    
    return best_params
```

`Best parameters: {'learning_rate': 0.3, 'max_depth': 12, 'n_estimators': 30}`

### CatBoost

The best CatBoost parameters were determined according to the following function:

```python
def get_best_CB_params():
    
    # Define the params dict
    params_dict = {'depth': [6,12],
                   'learning_rate': [0.1, 0.3],
                   'iterations': list(range(10, 31, 10))} # similar to n_estimators
                   
    # Initialize the model
    model = CatBoostRegressor(random_state=123456, iterations=300, verbose=0)
       
    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(model, param_grid=params_dict, scoring='neg_mean_squared_error', cv=3)
    
    # Fit the grid search object
    grid_search.fit(X_train, y_train, cat_features=cat_features)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    
    return best_params
```

`Best parameters: {'depth': 12, 'iterations': 30, 'learning_rate': 0.3}`

## Model Analysis

Relevant statistics on performance were stored in the results_df.
|   | Description         | RMSE        | TrainTime_ms | PredictTime_ms | TrainTime_s | PredictTime_s |
|---|---------------------|-------------|--------------|----------------|-------------|---------------|
| 0 | Linear Regression   | 2675.445352 | 2580.0       | 25.70          | 2.580       | 0.02570       |
| 1 | Random Forest       | 1796.647142 | 39700.0      | 239.00         | 39.700      | 0.23900       |
| 2 | XGBoost             | 1715.013808 | 19500.0      | 52.00          | 19.500      | 0.05200       |
| 3 | LightGBM            | 1844.570307 | 327.0        | 29.40          | 0.327       | 0.02940       |
| 4 | CatBoost            | 1779.668107 | 1090.0       | 6.36           | 1.090       | 0.00636       |


![RMSE Scores](/images/rmse.png)

Excluding the baseline check of the Linear Regression model, each tested model has similar performance within a range of 130 Euros. The best performing model (XGBoost) had an RMSE value of 960 Euros better than the baseline test.

![Train Times](/images/train.png)

In terms of training time, our baseline model is actually in the middle of the pack. Our best performing model in terms of RMSE takes 7.56 times longer to train than the baseline model, and Random Forest takes even longer. Conversely, LightGBM trains 7.89 times faster than our baseline and is the fastest training model.

![Prediction Time](/images/predict.png)

Again, out baseline model performs rather quickly but is beaten by the CatBoost model, which predicts in about half the time. It's important to note that all models are predicting in the validation set (64,161 rows) in less than half a second.

## Conclusions

The three gradient boosting regressor models (XGBoost, LightGMB, and CatBoost) as well as the Random Forest model returned improved RMSE scores compared to the baseline Linear Regression model. XGBoost scored the best with an RMSE of 960 Euros better than the baseline and 65 Euros better than the next best performing model.

However, Rusty Bargain used car sales service is also concerned with the speed of training and prediction for these models. Unfortunately, XGBoost takes 19.5 seconds to train, 7.56 times longer than our baseline test. Our next best performing model, CatBoost, trains in just over a second and is the fastest model when making predictions.

Given the full scope of Rusty Bargains business needs, I suggest they implement the CatBoost model into their new application. It offers high performance with exceptionally fast training and prediction times.

## Further Recommendations

Allocating more computational resources would likely improve performance due to two main factors:

1. Maintaining data integrity. The entire postal code feature needed to be removed from the dataset in order to make the regression problem simple enough for the limited computational resources. Similarly, car models outside the 200 most popular had to be collapsed into the 'unknown' column for the same reason.
2. More and deeper trees. The hyperparameter tuning process had to remain limited to the low end of computational demand in order to not crash Rusty Bargain's infrastructure. In each model trained, the best parameters met the ceiling for many arguments, specifically the n_estimators. It is likely that each of these models would perform better with n_estimators >= 100. 
