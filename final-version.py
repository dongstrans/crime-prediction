# Import python libraries
import streamlit as st
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

st.write("""
# Stop Search Crime Prediction
**The data shows all the London Boroughs and the associated Crimes.**
""")

# Read csv file
df = pd.read_csv('london.csv')

df

# Get data year range
months = [str(i) for i in range(202201, 202213)]

# Get unique values of offence column
offence = df['Major Category'].unique()
offence
print(offence)

# Get unique values for borough column
borough = df['Borough'].unique()
borough
print(borough)

st.write("""
![alt text](/assets/images/Barking and Dagenham.PNG)
""")

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
        
# Plot graph for a particular offence
fig = plt.figure(figsize = (20, 10), dpi = 100, facecolor = 'w', edgecolor = 'k')
plt.title('Violence Against the Person for each Borough')
plt.xlabel('Month')
plt.ylabel('No of Crime')
for area in borough:
    temp_df = df[(df['Borough'] == area) & (df['Major Category'] == 'Violence Against the Person')]
    n_crime = [temp_df[c].values[0] for c in months]
    plt.plot(months, n_crime)
    plt.legend(borough)
    
# Print out number of features
for c_name in df.columns:
    if df[c_name].dtypes == 'object':
        unique_cat = len(df[c_name].unique())
        print("'{c_name}' has {unique_cat} features".format(c_name = c_name, unique_cat = unique_cat))
        
# Creating labels
lab = preprocessing.LabelEncoder()

# Scale the training data
df['Major Category'] = lab.fit_transform(df['Major Category'])

df.head()

# Setting number of clusters
kmeans = KMeans(n_clusters = 10)

# Fitting the data
kmeans.fit(df.iloc[:,1:])

# Output clusters
kmeans.cluster_centers_

# Apply labels to the clusters
labels = kmeans.labels_

labels

# Get unique values
unique, counts = np.unique(kmeans.labels_, return_counts = True)

# Store in a dictionary
d_data = dict(zip(unique, counts))

d_data

# Applying the label
df['cluster'] = kmeans.labels_

# Plot graph
sns.lmplot('202201', '202202', data = df, hue = 'cluster', size= 10, aspect = 1, fit_reg = False)

# Inertia get the sum of squared error for the cluster
# Checks how close the cluster points are
kmeans.inertia_

# K-Means score added to column
kmeans.score

df

# Display bar plot
f, ax = plt.subplots(figsize = (24, 15))

# Load the data
stats = df.sort_values(['cluster', 'Borough'], ascending = True)
sns.set_color_codes('pastel')
sns.barplot(y = 'Borough', x = '202201', data = stats)

sns.despine(left = True, bottom = True)

# Split dataset for testing and training
X = df.iloc[:,1:15]
y = df.iloc[:,df.columns=='cluster']

# Print out headers
print(X.head())
y.head()

# Dividing dataset to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Creating an instance of Random Forest
random_forest = RandomForestClassifier(n_estimators = 100)

# Creating test and training data
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

# Get report scores
from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(y_pred, y_test)
print("Accuracy: ", acc)

# Calculate classification
clf = classification_report(y_pred, y_test)
print(clf)

# Create an instance of logistic regression
lg = LogisticRegression()

# Fitting the training data
lg.fit(X_train, y_train)

# Calculate prediction
predic = lg.predict(X_test)

# Get accuracy score
acc2 = accuracy_score(predic, y_test)
print("Accuracy: ", acc2)

# Get classification score
clf2 = classification_report(predic, y_test)
print(clf2)

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=10)

# Train the classifier on the training set
knn.fit(X_train, y_train)

# Use the classifier to predict the labels of the test set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the classifier on the test set
acc3 = accuracy_score(y_test, y_pred)
print("Accuracy:", acc3)

# Get classification
clf3 = classification_report(y_pred, y_test)
print(clf3)

# Compare the results of all different ML Algorithms

# Create label to plot on graph
objects = ('Random Forest', 'Logistic Regression', 'K-Nearest Neightbor')
y_pos = np.arange(len(objects))
performance = [acc, acc2, acc3]

plt.bar(y_pos, performance, align = 'center', alpha = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Random Forest vs Logistic Regression vs K-Nearest Nieghbor')
plt.show()

# Generate the confusion matrix for knn
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Create a Random Forest classifier with n_estimators=100
rf = RandomForestClassifier(n_estimators=100)

# Train the classifier on the training set
rf.fit(X_train, y_train)

# Use the classifier to predict the labels of the test set
y_pred = rf.predict(X_test)

# Generate the confusion matrix for random forest
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Create a logistic regression classifier with default parameters
lr = LogisticRegression()

# Train the classifier on the training set
lr.fit(X_train, y_train)

# Use the classifier to predict the labels of the test set
y_pred = lr.predict(X_test)

# Generate the confusion matrix logistic regression
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)