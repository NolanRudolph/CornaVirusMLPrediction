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
        



"""
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

"""--- Test/Train Split ---"""
from sklearn.model_selection import train_test_split
# Splits data into train/validation data and labels, and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(df_model, df_model["dead-2/20/20"], test_size = 0.2, random_state = 0)

# Splits taining and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = .125, random_state = 0)

# X_test - 70% of data, X_val - 10% of data, X_test - 20% of data

"""--- Model ---"""
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
    
# Predicting the validation set results
y_pred_val = model.predict(X_val)
    
# Making the Confusion Matrix
cm = metrics.confusion_matrix(y_val, y_pred_val)
print("Correct negatives: " + str(cm[0,0]) + " False positives: " + str(cm[0,1]))
print("False negatives: " + str(cm[1,0]) + " Correct positives: " + str(cm[1,1]))
    
# Compute accuracy based on confusion matrix
accuracy = (cm[0,0] + cm[1,1]) / (cm[0,1] + cm[1,0] + cm[0,0] + cm[1,1])
print("Accuracy: " + str(accuracy))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Cross Val Mean - " + str(accuracies.mean()))
print("Cross Val Std Dev - " + str(accuracies.std()))






