#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:52:48 2020

@author: Jacob Evarts, Nolan Rudolph
"""

# Quick processing
import numpy as np

# CSV Reading
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Machine Learning Prebuilt Methods/Classes
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def main():
    """--- DATA ACQUISITION ---"""
    # Requirements: pandas
    
    # Importing the dataset
    dataset = pd.read_csv('Datasets/covid_19_data.csv')
    
    # Dataframe of relevant features -- Matrix that is really easy to work with
    dataFrame = pd.DataFrame()
    dataFrame = dataset.copy()
    
    # Replaces unknown province/state with the country name
    for index, row in dataFrame.iterrows():
        if pd.isna(dataFrame.iloc[index, 2]):
            dataFrame.iloc[index, 2] = dataFrame.iloc[index, 3]
    
    dataFrame.insert(3, "Region", dataFrame.loc[:, "Province/State"])
    dataFrame = dataFrame.rename(columns={"Province/State" : "Encoded_Region"})
    
    
    """--- ENCODE CATEGORICAL DATA ---"""
    # Requirements: LabelEncoder, OneHotEncoder
    
    # Label encoder takes each region and converts it to a unique number for each region
    # One Hot makes a column for every region (for boolean value) -- Adds new Column for every region
    label_encoder = LabelEncoder()
    onehotencoder = OneHotEncoder()
    
    # Just the Province/State column
    # Encodes Province/State as numbers
    regions = dataFrame.loc[:, "Region"]
    dataFrame.loc[:, "Encoded_Region"] = label_encoder.fit_transform(dataFrame.loc[:, "Encoded_Region"])
    
    # Creates dummy variables for the Province/States
    # Each column represents one region (boolean values) due to HotEncoder
    regions = regions.values.reshape(-1, 1)
    regions = onehotencoder.fit_transform(regions).toarray()
    column_names = onehotencoder.get_feature_names(['Region'])
    regions_dataFrame = pd.DataFrame(regions, columns = column_names)
    
    dataFrame = pd.concat([dataFrame, regions_dataFrame], axis=1)
    
    # drop last column of encoding for dependancy issues
    dataFrame = dataFrame.iloc[:,:-1]
    
    
    """--- FINAL PREPROCESSING ---"""
    # Convert dates into numerical value (days since first date given)
    dates = dataFrame.loc[:, "ObservationDate"]
    num_days = pd.Series()
    for date in dates:
        month = date[0:2]
        day = date[3:5]
        # January
        if month == "01":
            days_since = (int(day) - 22)
        # Febuary
        else:
            days_since = (int(day) + 9)
        num_days = num_days.append(pd.Series(days_since), ignore_index=True)
    
    dataFrame.insert(2, "Days_Elapsed", num_days)
    
    # Dataframe with unnecessary features removed
    dataFrame_model = dataFrame.copy()
    confirmed = dataFrame_model["Confirmed"]
    dataFrame_model.drop(["SNo", "ObservationDate", "Encoded_Region", "Region", \
                       "Country/Region", "Last Update", "Confirmed"], inplace=True, axis=1)
            
    
    """--- TEST/TRAIN SPLIT ---"""
    # Requirements: sklearn.model_selection.train_test_split
    
    # X_test - 70% of data, X_val - 10% of data, X_test - 20% of data
    # Splits data into train/validation data and labels, and test data
    X_train_val, X_test, y_train_val, y_test = train_test_split(dataFrame_model, confirmed, test_size = 0.2, random_state = 0)
    
    # Splits taining and validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.125, random_state = 0)
    
    
    """--- MODEL HYPERTUNING ---"""
    # Requirements: sklearn.ensemble.GradientBoostingRegressor, sklearn.model_selection.GridSearchCV
    
    params = {"n_estimators" : [3000],        # tested [500, 1000, 3500]
                  "max_depth" : [3],          # tested [3, 4, 6, 8]
                  "learning_rate" : [0.01]}   # tested [0.01, 0.05, 0.1]
    
    gb = GradientBoostingRegressor()
    classifier = GridSearchCV(estimator=gb, param_grid=params, cv=5, n_jobs=10)
    classifier.fit(X_train, y_train)
    print("Best Parameters: ", classifier.best_params_)
    
    model = classifier.best_estimator_
    
    # Model fit
    print("Fitment Score: ", model.score(X_train, y_train))
    
    # Predicting the validation set results
    y_pred_val = model.predict(X_val)
    
    mse = metrics.mean_squared_error(y_val, y_pred_val)
    print("Mean square error: %.4f" % mse)
    
    plt.figure(figsize=(12,6))
    plt.title("Graident Boosting Model")
    plt.scatter(y_train, y_train)
    plt.plot(y_train, model.predict(X_train), color = 'black')
    plt.show()
    
    path = "Figures/regression.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12,6))
    plt.title("Graident Boosting Model")
    plt.scatter(y_train, y_train)
    plt.plot(y_train, model.predict(X_train), color = 'black')
    plt.ylim(0, 10000)
    plt.xlim(0, 10000)
    plt.show()
    path = "Figures/regression_scaled.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    
    """--- METRICS --"""
    # Requirements: sklearn.model_selection 
    
    # Predicting the validation set results
    y_pred_val = model.predict(X_val)
    
    mse = metrics.mean_squared_error(y_val, y_pred_val)
    print("Mean square error: %.4f" % mse)
    
    # R-squared metric with cross validation
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(gb, X_train, y_train, cv=kfold, scoring='r2')
    print("R2 Cross Val Mean - " + str(results.mean()))
    print("R2 Cross Val Std Dev - " + str(results.std()))
        
    # Plot feature importance
    feature_importance = model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance[0:5])
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()
    path = "Figures/feature_importance.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    
    """--- BASELINE ---"""
    # Requirements: from sklearn.linear_model import LinearRegression
    mean = pd.Series()
    median = pd.Series()
    for val in y_val:
        mean = mean.append(pd.Series(y_val.mean()), ignore_index=True)
        median = median.append(pd.Series(y_val.median()), ignore_index=True)
    
    #lin_reg = LinearRegression()
    #lin_reg.fit(X_train, y_train)
    #lin_pred = lin_reg.predict(y_val)
    
    # Mean
    mse_mean = metrics.mean_squared_error(y_val, mean)
    print("Mean square error (Mean prediction): %.4f" % mse_mean)
    mse_median = metrics.mean_squared_error(y_val, median)
    # Medain
    print("Mean square error (Median prediction): %.4f" % mse_median)
    # Linear Regression
    #mse_lin_reg =  metrics.mean_squared_error(y_val, median)
    #print("Mean square error (Linear regression prediction): %.4f" % mse_median)
    
    #barlist = plt.bar([mse, mse_mean, mse_median, mse_lin_reg], ["Gradient Boosted Regression", "Mean", "Median", "Linear Regression"])
    #plt.show()

# Encase we need several modules in the future
if __name__ == "__main__":
    main()
