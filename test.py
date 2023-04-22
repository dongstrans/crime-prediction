# Import python libraries
import streamlit as st
#import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Read csv file
df = pd.read_csv('london.csv')

df

st.text("Test")