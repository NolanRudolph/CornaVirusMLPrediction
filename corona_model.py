#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:52:48 2020

@author: Jacob Evarts, Nolan Rudolph
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

"""--- DATA ACQUISITION ---"""
# Importing the dataset
dataset = pd.read_csv('novel-corona-virus-2019-dataset/covid_19_data.csv')

# Right now model predicts number of deaths due to novel coronavirus
confirmed_tseries = pd.read_csv('novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
recovered_tseries = pd.read_csv('novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
death_tseries = pd.read_csv('novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

# Dataframe of relevant features 
df = pd.DataFrame()
df = confirmed_tseries.copy()
df = df.rename(columns={"Province/State" : "Region"})

# Replaces unknown province/state with t0e country name
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

"""--- Test/Train Split ---"""
from sklearn.model_selection import train_test_split
# Splits data into train/validation data and labels, and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(df, df["dead-2/20/20"], test_size = 0.2, random_state = 0)

# Splits taining and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = .125, random_state = 0)

# X_test - 70% of data, X_val - 10% of data, X_test - 20% of data






