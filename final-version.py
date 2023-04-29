# Import python libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

st.write("""
# Stop Search Crime Prediction
**This project demonstrates how the data can be used for predictions.**
""")

st.write("Firstly all of the python libraries needs to be imported.")

code = '''import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image'''
st.code(code, language='python')

st.write("Data was taken from London DataStore, the data was manipulated to get all the total crime for each crime category and organized by the London Borough. The code below is used to read the csv file and display the table.")

with st.echo():

    df = pd.read_csv('london.csv')
    df

# Read csv file
# df = pd.read_csv('london.csv')
# df

st.write("I create a variable called months to store a month range.")
code = '''
months = [str(i) for i in range(202201, 202213)]
'''
st.code(code, language='python')

# Get data year range
months = [str(i) for i in range(202201, 202213)]

# Get unique values of offence column
st.write("I get the unique values of the Mayor Category column.")
code = '''
offence = df['Major Category'].unique()
offence
print(offence)
'''
st.code(code, language='python')
st.write("**List of Crimes**")
offence = df['Major Category'].unique()
offence
print(offence)

# Get unique values for borough column
st.write("This gets the unique values of the Borough column.")
code = '''
borough = df['Borough'].unique()
borough
print(borough)
'''
st.code(code, language='python')
st.write("**List of London Boroughs**")
borough = df['Borough'].unique()
borough
print(borough)

st.write("Plotting the graph for to display all of the London Borough Crimes.")
code = '''
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
            '''
st.code(code, language='python')

# Load all images
image = Image.open('Barking and Dagenham.jpg')
st.image(image, caption = 'Barking and Dagenham Crimes')
image = Image.open('Barnet.jpg')
st.image(image, caption = 'Barnet Crimes')
image = Image.open('Bexley.jpg')
st.image(image, caption = 'Bexley Crimes')
image = Image.open('Brent.jpg')
st.image(image, caption = 'Brent Crimes')
image = Image.open('Bromley.jpg')
st.image(image, caption = 'Bromley Crimes')
image = Image.open('Camden.jpg')
st.image(image, caption = 'Camden Crimes')
image = Image.open('Croydon.jpg')
st.image(image, caption = 'Croydon Crimes')
image = Image.open('Ealing.jpg')
st.image(image, caption = 'Ealing Crimes')
image = Image.open('Enfield.jpg')
st.image(image, caption = 'Enfield Crimes')
image = Image.open('Greenwich.jpg')
st.image(image, caption = 'Greenwich Crimes')
image = Image.open('Hackney.jpg')
st.image(image, caption = 'Hackney Crimes')
image = Image.open('Hammersmith and Fulham.jpg')
st.image(image, caption = 'Hammersmith and Fulham Crimes')
image = Image.open('Haringey.jpg')
st.image(image, caption = 'Haringey Crimes')
image = Image.open('Harrow.jpg')
st.image(image, caption = 'Harrow Crimes')
image = Image.open('Havering.jpg')
st.image(image, caption = 'Havering Crimes')
image = Image.open('Hillingdon.jpg')
st.image(image, caption = 'Hillingdon Crimes')
image = Image.open('Hounslow.jpg')
st.image(image, caption = 'Hounslow Crimes')
image = Image.open('Islington.jpg')
st.image(image, caption = 'Islington Crimes')
image = Image.open('Kensington and Chelsea.jpg')
st.image(image, caption = 'Kensington and Chelsea Crimes')
image = Image.open('Kingston upon Thames.jpg')
st.image(image, caption = 'Kingston upon Thames Crimes')
image = Image.open('Lambeth.jpg')
st.image(image, caption = 'Lambeth Crimes')
image = Image.open('Lewisham.jpg')
st.image(image, caption = 'Lewisham Crimes')
image = Image.open('Merton.jpg')
st.image(image, caption = 'Merton Crimes')
image = Image.open('Newham.jpg')
st.image(image, caption = 'Newham Crimes')
image = Image.open('Redbridge.jpg')
st.image(image, caption = 'Redbridge Crimes')
image = Image.open('Richmond upon Thames.jpg')
st.image(image, caption = 'Richmond upon Thames Crimes')
image = Image.open('Southwark.jpg')
st.image(image, caption = 'Southwark Crimes')
image = Image.open('Sutton.jpg')
st.image(image, caption = 'Sutton Crimes')
image = Image.open('Tower Hamlets.jpg')
st.image(image, caption = 'Tower Hamlets Crimes')
image = Image.open('Waltham Forest.jpg')
st.image(image, caption = 'Waltham Forest Crimes')
image = Image.open('Wandsworth.jpg')
st.image(image, caption = 'Wandsworth Crimes')
image = Image.open('Westminster.jpg')
st.image(image, caption = 'Westminster Crimes')

st.write("Plotting a graph to show all London Boroughs with the Drug Offences Crime.")
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
code = '''
for area in borough:
fig = plt.figure(figsize = (20, 10), dpi = 100, facecolor = 'w', edgecolor = 'k')
plt.title('Drug Offences for each Borough')
plt.xlabel('Month')
plt.ylabel('No of Crime')
for area in borough:
    temp_df = df[(df['Borough'] == area) & (df['Major Category'] == 'Drug Offences')]
    n_crime = [temp_df[c].values[0] for c in months]
    plt.plot(months, n_crime)
    plt.legend(borough)
'''
st.code(code, language='python')

# Plot graph for a particular offence
fig = plt.figure(figsize = (20, 10), dpi = 100, facecolor = 'w', edgecolor = 'k')
plt.title('Drug Offences for each Borough')
plt.xlabel('Month')
plt.ylabel('No of Crime')
for area in borough:
    temp_df = df[(df['Borough'] == area) & (df['Major Category'] == 'Drug Offences')]
    n_crime = [temp_df[c].values[0] for c in months]
    plt.plot(months, n_crime)
    plt.legend(borough)

image = Image.open('drug offences.jpg')
st.image(image, caption = 'Drug Offences for each Borough')

st.write("Print out the number of features.")
# Print out number of features
for c_name in df.columns:
    if df[c_name].dtypes == 'object':
        unique_cat = len(df[c_name].unique())
        print("'{c_name}' has {unique_cat} features".format(c_name = c_name, unique_cat = unique_cat))
code = '''
for c_name in df.columns:
        if df[c_name].dtypes == 'object':
            unique_cat = len(df[c_name].unique())
            print("'{c_name}' has {unique_cat} features".format(c_name = c_name, unique_cat = unique_cat))
'''
st.code(code, language='python')
image = Image.open('num of features.jpg')
st.image(image, caption = '')

st.write("Creating labels for the Major Category attribute, fitting the data printing out the headers.")
# Creating labels
lab = preprocessing.LabelEncoder()
with st.echo():
    lab = preprocessing.LabelEncoder()
    df['Major Category'] = lab.fit_transform(df['Major Category'])
    df.head()
    df

# Scale the training data
df['Major Category'] = lab.fit_transform(df['Major Category'])

#df.head()
#st.write("**Added the column Major Category into the tablet**")
#df

st.write("Setting the no of clusters, fitting the data and displaying the output of the clusters.")
code = '''
kmeans = KMeans(n_clusters = 10)
kmeans.cluster_centers_
labels = kmeans.labels_
labels
'''
st.code(code, language='python')
# Setting number of clusters
kmeans = KMeans(n_clusters = 10)

# Fitting the data
kmeans.fit(df.iloc[:,1:])

# Output clusters
st.write("**The output clusters for the 12 months**")
kmeans.cluster_centers_

# Apply labels to the clusters
labels = kmeans.labels_
st.write("**Apply labels to the clusters**")
labels

st.write("Get the unique values, store these in a dictionary and applying the labels.")
# Get unique values
unique, counts = np.unique(kmeans.labels_, return_counts = True)

# Store in a dictionary
d_data = dict(zip(unique, counts))
d_data

# Applying the label
df['cluster'] = kmeans.labels_

code = '''
unique, counts = np.unique(kmeans.labels_, return_counts = True)
d_data = dict(zip(unique, counts))
d_data
df['cluster'] = kmeans.labels_
'''
st.code(code, language='python')

# Plot graph
st.write("Plotting a graph to show all of the clusters from the Major Category.")
sns.lmplot('202201', '202202', data = df, hue = 'cluster', size= 10, aspect = 1, fit_reg = False)
code = '''
sns.lmplot('202201', '202202', data = df, hue = 'cluster', size= 10, aspect = 1, fit_reg = False)
'''
st.code(code, language='python')
image = Image.open('seaborn.jpg')
st.image(image, caption = '')

# Inertia get the sum of squared error for the cluster
# Checks how close the cluster points are
st.write("**The inertia to get the sum of squared errors for the cluster**")
kmeans.inertia_

# K-Means score added to column
st.write("Add the cluster column into the table and display it.")
code = '''
kmeans.score
df
'''
st.code(code, language='python')
#kmeans.score
df

# Display bar plot
st.write("Displaying a bar chart of all London Boroughs of the clusters of January.")
code = '''
f, ax = plt.subplots(figsize = (24, 15))
stats = df.sort_values(['cluster', 'Borough'], ascending = True)
sns.set_color_codes('pastel')
sns.barplot(y = 'Borough', x = '202201', data = stats)

sns.despine(left = True, bottom = True)
'''
st.code(code, language='python')
f, ax = plt.subplots(figsize = (24, 15))
stats = df.sort_values(['cluster', 'Borough'], ascending = True)
sns.set_color_codes('pastel')
sns.barplot(y = 'Borough', x = '202201', data = stats)
image = Image.open('barchart.jpg')
st.image(image, caption = '')

sns.despine(left = True, bottom = True)

# Split dataset for testing and training
st.write("Split the data for training and testing and printing out the headers for x and y.")
X = df.iloc[:,1:15]
y = df.iloc[:,df.columns=='cluster']
print(X.head())
y.head()
code = '''
X = df.iloc[:,1:15]
y = df.iloc[:,df.columns=='cluster']
print(X.head())
y.head()
'''
st.code(code, language='python')
image = Image.open('splitdata.jpg')
st.image(image, caption = '')

# Dividing dataset to training and testing data
st.write("Dividing the datasets to training and testing data.")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
code = '''
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
'''
st.code(code, language='python')

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Creating an instance of Random Forest
st.write("""## Using Random Forest Algorithm to train and test prediction""")
st.write("Creating an instance of random forest, fitting the data and get the prediction scores.")
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print("Accuracy: ", acc)
clf = classification_report(y_pred, y_test)
print(clf)
code = '''
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print("Accuracy: ", acc)
clf = classification_report(y_pred, y_test)
print(clf)
'''
st.code(code, language='python')
image = Image.open('randomforestscore.jpg')
st.image(image, caption = '')

# Create an instance of logistic regression
st.write("""##Using the Logistic Regression Algorithm to train and test prediction""")
st.write("Create an instance of logistic regression, fitting the data and get the prediction scores.")
lg = LogisticRegression()
lg.fit(X_train, y_train)
predic = lg.predict(X_test)
acc2 = accuracy_score(predic, y_test)
print("Accuracy: ", acc2)
clf2 = classification_report(predic, y_test)
print(clf2)
code = '''
lg = LogisticRegression()
lg.fit(X_train, y_train)
predic = lg.predict(X_test)
acc2 = accuracy_score(predic, y_test)
print("Accuracy: ", acc2)
clf2 = classification_report(predic, y_test)
print(clf2)
'''
st.code(code, language='python')
image = Image.open('logisticregressionscore.jpg')
st.image(image, caption = '')

# Create a KNN classifier with k=5
st.write("""## Using the K-Nearest Neighbor Algorithm to train and test prediction""")
st.write("Create an instance of k-nearest neighbor, fitting the data and get the prediction scores.")
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc3 = accuracy_score(y_test, y_pred)
print("Accuracy:", acc3)
clf3 = classification_report(y_pred, y_test)
print(clf3)
code = '''
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc3 = accuracy_score(y_test, y_pred)
print("Accuracy:", acc3)
clf3 = classification_report(y_pred, y_test)
print(clf3)
'''
st.code(code, language='python')
image = Image.open('knearestneighborscore.jpg')
st.image(image, caption = '')

# Compare the results of all different ML Algorithms
st.write("""## Compare the results""")
st.write("Create labels to plot on graph.")
objects = ('Random Forest', 'Logistic Regression', 'K-Nearest Neightbor')
y_pos = np.arange(len(objects))
performance = [acc, acc2, acc3]
plt.bar(y_pos, performance, align = 'center', alpha = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Random Forest vs Logistic Regression vs K-Nearest Nieghbor')
plt.show()
code = '''
objects = ('Random Forest', 'Logistic Regression', 'K-Nearest Neightbor')
y_pos = np.arange(len(objects))
performance = [acc, acc2, acc3]
plt.bar(y_pos, performance, align = 'center', alpha = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Random Forest vs Logistic Regression vs K-Nearest Nieghbor')
plt.show()
'''
st.code(code, language='python')
image = Image.open('compare.jpg')
st.image(image, caption = '')

# Generate the confusion matrix for knn
st.write("""## Generate confusion matrix for K-Nearest Neighbor""")
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
code = '''
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
'''
st.code(code, language='python')
image = Image.open('knnmatrix.jpg')
st.image(image, caption = '')

# Create a Random Forest classifier with n_estimators=100
st.write("""## Generate confusion matrix for Random Forest""")
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
code = '''
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
'''
st.code(code, language='python')
image = Image.open('randmatrix.jpg')
st.image(image, caption = '')

# Create a logistic regression classifier with default parameters
st.write("""## Generate confusion matrix for Logistic Regression""")
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
code = '''
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
'''
st.code(code, language='python')
image = Image.open('logmatrix.jpg')
st.image(image, caption = '')
