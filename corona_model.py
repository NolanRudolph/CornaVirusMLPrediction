#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:52:48 2020

@author: Jacob Evarts, Nolan Rudolph
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""--- Data Acquisition ---"""
# Importing the dataset
dataset = pd.read_csv('novel-corona-virus-2019-dataset/covid_19_data.csv')

# Right now model predicts number of deaths due to novel coronavirus
confirmed_tseries = pd.read_csv('novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
recovered_tseries = pd.read_csv('novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
death_tseries = pd.read_csv('novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

# Dataframe of relevant features 
df = pd.DataFrame()
df = dataset.copy()

# Replaces unknown province/state with the country name
for index, row in df.iterrows():
    if pd.isna(df.iloc[index, 2]):
        df.iloc[index, 2] = df.iloc[index, 3]

df.insert(3, "Region", df.loc[:, "Province/State"])
df = df.rename(columns={"Province/State" : "Encoded_Region"})

"""--- Encode Categorical Data ---"""
# sklearn
label_encoder = LabelEncoder()
regions = df.loc[:, "Region"]
# Just the Province/State column
# Encodes Province/State as numbers
df.loc[:, "Encoded_Region"] = label_encoder.fit_transform(df.loc[:, "Encoded_Region"])

# Creates dummy variables for the Province/States
onehotencoder = OneHotEncoder()
regions = regions.values.reshape(-1, 1)
regions = onehotencoder.fit_transform(regions).toarray()
column_names = onehotencoder.get_feature_names(['Region'])
regions_df = pd.DataFrame(regions, columns = column_names)

df = pd.concat([df, regions_df], axis=1)
# drop last column of encoding for dependancy issues
df = df.iloc[:,:-1]

"""--- Final Preprocessing ---"""
# Convert dates into numerical value (days since first date given)
dates = df.loc[:, "ObservationDate"]
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

df.insert(2, "Days_Elapsed", num_days)

# Dataframe with unnecessary features removed
df_model = df.copy()
df_model.drop(["SNo", "ObservationDate", "Encoded_Region", "Region", \
               "Country/Region", "Last Update"], inplace=True, axis=1)

"""--- Test/Train Split ---"""
from sklearn.model_selection import train_test_split
# Splits data into train/validation data and labels, and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(df_model, df_model["Deaths"], test_size = 0.2, random_state = 0)

# Splits taining and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = .125, random_state = 0)
# X_test - 70% of data, X_val - 10% of data, X_test - 20% of data


"""--- Model ---"""
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
# Parameters to tune
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

classifier = GradientBoostingRegressor(**params)
classifier.fit(X_train, y_train)

# Predicting the validation set results
y_pred_val = classifier.predict(X_val)

mse = metrics.mean_squared_error(y_val, y_pred_val)
print("Mean square error: %.4f" % mse)



"""--- Hypertuning ---"""
estimators = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000]
est_mse = []

# Tests different numbers of estimator values
for n_estimators in estimators:
    params = {'n_estimators': n_estimators, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

    classifier = GradientBoostingRegressor(**params)
    classifier.fit(X_train, y_train)

    mse = metrics.mean_squared_error(y_val, classifier.predict(X_val))
    est_mse.append(mse)

plt.plot(estimators, est_mse)
plt.show()


learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
lr_mse = []

# Tests different learning rate values
for lr in learning_rates:
    params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': lr, 'loss': 'ls'}

    classifier = GradientBoostingRegressor(**params)
    classifier.fit(X_train, y_train)

    mse = metrics.mean_squared_error(y_val, classifier.predict(X_val))
    lr_mse.append(mse)

plt.plot(estimators, lr_mse)
plt.show()
    

""" 
***UNCESSESARY SHIT WE PROBABLY DONT NEED***


df = confirmed_tseries.copy()
df = df.rename(columns={"Province/State" : "Region"})

# Replaces unknown province/state with the country name
for index, row in df.iterrows():
    if pd.isna(df.iloc[index, 0]):
        df.iloc[index, 0] = df.iloc[index, 1]

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
    df = df.rename(columns={date : ("conf-" + date)})
    
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
    df["rec-" + date] = recovered_tseries[date]
    
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
    df["dead-" + date] = recovered_tseries[date]
    
df.drop("Country/Region", inplace=True, axis=1)

# Remove region text data for model
df_model = df.drop("Region", axis=1)
"""
    










