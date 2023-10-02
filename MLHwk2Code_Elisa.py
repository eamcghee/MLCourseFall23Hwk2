#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:57:39 2023

@author: em
"""
#Hwk2, ATS 787A7
#Fully-connected or standard CNN to predict quantity  y, given inputs X
#as a classification task

#initial code copied from "Feed-forward neural network using ozone data at Joshua Tree"
#by Jamin Rader at CSU

#////////0. Set Up Environments////////

import sys
import numpy as np
import seaborn as sb

import pandas as pd
import datetime
import tensorflow as tf

# import tensorflow.keras as keras
import sklearn

# import pydot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %matplotlib inline

# tf.compat.v1.disable_v2_behavior()

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"tensorflow version = {tf.__version__}")

#////////1. Data Preparation////////
#1.1 Data Overview

# Read in data from url
#data = pd.read_csv(url, parse_dates=["DATE_TIME"], infer_datetime_format=True)

# Add hour and day of year
#data["HOUR"] = data["DATE_TIME"].dt.hour
#data["MONTH"] = data["DATE_TIME"].dt.month
#data["YEAR"] = data["DATE_TIME"].dt.year
#data["DAYOFYEAR"] = data["DATE_TIME"].dt.dayofyear
#data.sort_values("DATE_TIME", inplace=True, ignore_index=True)

#1.2 Define Input and Output
#take a look at the data...what we have for the inputs, x1, x2, x3, x4

#The 2015 benchmark for human health ozone condition is shown here. 
#Let us predict whether the _______ will be classified as 
# Y1: good, Y2:fair, or Y3:poor over _-hour periods.
#Define classes
#Y1 = ____ value            #Good  ≤  54.9 ppb
#Y2 = ____ value            #Fair 55.0 - 70.9 ppb
#Y3 = ____ value            #Poor  ≥  71.0 ppb

#Let's start out by training our model using Temperature, Relative Humidity, Windspeed, and Day of Year.

## Here are all the different variables that we could use for training our neural
# network (except ozone, of course)
# insert code to display data       #data.columns