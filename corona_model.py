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
dataset = pd.read_csv('novel-corona-virus-2019-dataset/2019_nCoV_data.csv')