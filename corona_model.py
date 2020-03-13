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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

def main():

    """--- DATA ACQUISITION ---"""
    # Requirements: pandas

    # Importing the dataset
    dataset = pd.read_csv('novel-corona-virus-2019-dataset/covid_19_data.csv')

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
    dataFrame_model.drop(["SNo", "ObservationDate", "Encoded_Region", "Region", \
                   "Country/Region", "Last Update"], inplace=True, axis=1)
        

    """--- TEST/TRAIN SPLIT ---"""
    # Requirements: sklearn.model_selection.train_test_split

    # X_test - 70% of data, X_val - 10% of data, X_test - 20% of data
    # Splits data into train/validation data and labels, and test data
    X_train_val, X_test, y_train_val, y_test = train_test_split(dataFrame_model, dataFrame_model["Deaths"], test_size = 0.2, random_state = 0)

    # Splits taining and validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.125, random_state = 0)


    """--- MODEL ---"""
    # Requirements: sklearn.metrics, sklearn.ensemble.GradientBoostingRegressor, sklearn.model_selection.GridSearchCV

    # Parameters to tune
    params = {'n_estimators': 3000, 'max_depth': 3, 'min_samples_split': 2,
              'learning_rate': 0.05, 'loss': 'ls'}

    classifier = GradientBoostingRegressor(**params)
    classifier.fit(X_train, y_train)

    # Predicting the validation set results
    y_pred_val = classifier.predict(X_val)

    mse = metrics.mean_squared_error(y_val, y_pred_val)
    print("Mean square error: %.4f" % mse)


    """--- HYPERTUNING ---"""
    # Requirements: sklearn.ensemble.GradientBoostingRegressor, sklearn.model_selection.GridSearchCV

    # Model fit
    print("Fitment Score: ", classifier.score(X_train, y_train))

    plt.figure(figsize=(12,6))
    plt.title("Graident Boosting Model")
    plt.scatter(y_train, y_train)
    plt.plot(y_train, classifier.predict(X_train), color = 'black')
    plt.show()

    # Tests different numbers of estimator values
    estimators = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000]
    est_mse = []

    params = {"n_estimators" : [500],
              "max_depth" : [3, 4, 6, 8],
              "learning_rate" : [0.01, 0.05, 0.1],
              "min_samples_leaf" : [20, 50, 100]}

    gb = GradientBoostingRegressor()
    classifier = GridSearchCV(estimator=gb, param_grid=params, cv=5)

    classifier.fit(X_train, y_train)


    """--- FOR POTENTIAL LATER USE ---"""
    """
    for n_estimators in estimators:
        params = {'n_estimators': n_estimators, 'max_depth': 3, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}

        classifier = GradientBoostingRegressor(**params)
        classifier.fit(X_train, y_train)

        mse = metrics.mean_squared_error(y_val, classifier.predict(X_val))
        est_mse.append(mse)

    plt.plot(estimators, est_mse)
    plt.show()


    # Tests different learning rate values
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    lr_mse = []

    for lr in learning_rate:
        params = {'n_estimators': 500, 'max_depth': 3, 'min_samples_split': 2,
              'learning_rate': lr, 'loss': 'ls'}

        classifier = GradientBoostingRegressor(**params)
        classifier.fit(X_train, y_train)

        mse = metrics.mean_squared_error(y_val, classifier.predict(X_val))
        lr_mse.append(mse)

    plt.plot(learning_rate, lr_mse)
    plt.show()
    """

    """ 
    ***UNCESSESARY SHIT WE PROBABLY DONT NEED***


    dataFrame = confirmed_tseries.copy()
    dataFrame = dataFrame.rename(columns={"Province/State" : "Region"})

    # Replaces unknown province/state with the country name
    for index, row in dataFrame.iterrows():
        if pd.isna(dataFrame.iloc[index, 0]):
            dataFrame.iloc[index, 0] = dataFrame.iloc[index, 1]

    # Rename date columns to specify they refer to the number of confirmed cases
    for i in range(30):
        date = ""
        if(i < 10):
            # Month
            date += ("1/")
            # Day
            date += (str(i + 22) + "/20")
        else:
            date += ("2/")
            date += (str(i - 9) + "/20")
        dataFrame = dataFrame.rename(columns={date : ("conf-" + date)})
        
    # Add number of recovered individuals on each date
    for i in range(30):
        date = ""
        if(i < 10):
            # Month
            date += ("1/")
            # Day
            date += (str(i + 22) + "/20")
        else:
            date += ("2/")
            date += (str(i - 9) + "/20")
        dataFrame["rec-" + date] = recovered_tseries[date]
        
    # Add number of individuals that died on each date
    for i in range(30):
        date = ""
        if(i < 10):
            # Month
            date += ("1/")
            # Day
            date += (str(i + 22) + "/20")
        else:
            date += ("2/")
            date += (str(i - 9) + "/20")
        dataFrame["dead-" + date] = recovered_tseries[date]
        
    dataFrame.drop("Country/Region", inplace=True, axis=1)

    # Remove region text data for model
    dataFrame_model = dataFrame.drop("Region", axis=1)
    """

# Encase we need several modules in the future
if __name__ == "__main__":
    main()
