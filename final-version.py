# Import python libraries
import streamlit as st
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Read csv file
df = pd.read_csv('london.csv')

df

# Get data year range
months = [str(i) for i in range(202201, 202213)]

# Get unique values of offence column
offence = df['Major Category'].unique()
offence

# Get unique values for borough column
borough = df['Borough'].unique()
borough

# Plot graph
for area in borough:
    fig = plt.figure(figsize = (20, 10), dpi = 100, facecolor = 'w', edgecolor = 'k')
    plt.title(area)
    plt.xlabel('Month')
    plt.ylabel('No of offences')
    for crime in offence:
        temp_df = df[(df['Borough'] == area) & (df['Major Category'] == crime)]
        n_crime = [temp_df[c].values[0] for c in months]
        plt.plot(months, n_crime)
        plt.legend(offence)