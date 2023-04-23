#!/usr/bin/env python
# coding: utf-8

# # INSTALING REQUIRED LIBRARIES

# In[2]:


get_ipython().system('pip install missingno')
get_ipython().system('pip install tensorflow')


# # IMPORTS

# In[3]:


import pandas as pd
import glob
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#  libraries for tensor flow
import tensorflow as tf
from tensorflow import keras




# # READING .CSV FILES

# WE FIRST IMPORT DATASET FOR THE CONTROL GROUP, I.E CONTROL GROUP

# In[4]:


# Get data file names
path = r"C:\Users\Akshay Mohan Nair\OneDrive - University of Essex\Spring Term\Decision Making\Assignment_2\unzipped\EyeT"
filenames = glob.glob(path + "/*dataset_II*.csv")
filenames = [file for file in filenames if "dataset_III" not in file]

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame_dataset_II = pd.concat(dfs, ignore_index=True)


# EXTRACTING NUMERICAL VALUE FOR THE PARTICIPANT CODE

# In[5]:


# Extract the last numeric characters from 'Participant name' column
big_frame_dataset_II['Participant name'] = big_frame_dataset_II['Participant name'].str[-2:]


# In[6]:


big_frame_dataset_II.head()


# In[7]:


big_frame_dataset_II.shape


# # DROPPING ROWS WITH NULL VALUES IN DIAMETER COLUMN

# In[8]:


big_frame_dataset_II = big_frame_dataset_II.dropna(subset=['Pupil diameter left', 'Pupil diameter right'])
big_frame_dataset_II.shape

big_frame_dataset_II['Pupil diameter left', 'Pupil diameter right'] = big_frame_dataset_II['Pupil diameter left', 'Pupil diameter right'].fillna(0)
# In[9]:


big_frame_dataset_II.to_csv('big_frame_dataset_II.csv')


# In[10]:


big_frame_dataset_II.head()


# # GRAPHS

# BELOW GRAPH SHOWS PERCENTAGE OF NULL VALUES IN ALL THE COLUMNS OF THE DATASET.

# In[11]:


msno.bar(big_frame_dataset_II)


# GRAPH FOR THE FLUCTUATION IN LEFT EYE'S PUPIL DIAMETER WITH RESPECT TO TIMESTAMP.

# In[12]:


big_frame_dataset_II.plot(x="Pupil diameter left",y= "Eyetracker timestamp", kind="line", figsize=(5, 5))


# GRAPH FOR THE FLUCTUATION IN RIGHT EYE'S PUPIL DIAMETER WITH RESPECT TO TIMESTAMP.

# In[13]:


big_frame_dataset_II.plot(x="Pupil diameter right",y= "Eyetracker timestamp", kind="line", figsize=(5, 5))


# CORELATION MATRIX FOR THE WHOLE DATASET

# In[14]:


matrix_co=big_frame_dataset_II.corr()
sns.heatmap(matrix_co, cmap="gist_gray",linewidth=.5) #cmap = 'crest' for B and W heatmap
plt.show()


# BAR GRAPH FOR THE COLUMNS IN THE DATASET.

# In[15]:


big_frame_dataset_II.hist(bins=50, figsize=(20,15))


# PIE CHART FOR THE DIFFERENT VALUES IN THE COLUMN 'Eye movement type'

# In[16]:


pie_chart = big_frame_dataset_II['Eye movement type'].value_counts(dropna=False)
print(pie_chart)
pie_chart.plot(kind="pie", autopct='%1.1f%%',    ylabel='EYE MOVEMENT TYPE VALUE COUNTS')


# BELOW TABLE SHOWS THAT THE ADDED VALUE OF "GAZE POINT LEFT X" AND "SUM OF GAZE POINT RIGHT X" IS EQUAL TO "GAZE POINT X".

# In[17]:


#Checking Values of Gaze Point X and Its Combined Value of Left and Right Eye
check_gaze = pd.DataFrame()
check_gaze['Added_value']=(big_frame_dataset_II['Gaze point left X']+ big_frame_dataset_II['Gaze point right X'])/2
check_gaze['Gaze point X'] = big_frame_dataset_II['Gaze point X']
check_gaze.head()


# BELOW CODE SHOWS THAT THE VALUE OF SUM OF "GAZE POINT LEFT X" AND "SUM OF GAZE POINT RIGHT X" IS EQUAL TO GAZE POINT X. THEREFORE THERE IS NO POINT IN TAKING THESE INDIVIDUAL COLUMNS INTO OUT FINAL DATASET.
# 
# THE VIOLET LINE JUST SHOWS THAT THE VALUES PERFECTLY OVERLAP EACH OTHER AS THE INDIVUAL COLOURS VALUES FOR THE LINES WERE RED AND BLUE.

# In[18]:


fig, ax = plt.subplots(figsize=(5,5))
check_gaze.plot(x="Added_value",kind="line", ax=ax)
check_gaze.plot(x="Gaze point X",kind="line", color="r", ax=ax,alpha=0.5)
plt.show()


# # SELECTING FEATURES

# FROM THE ABOVE GRAPH WE DECIDE TO SELECT THE BELOW FEATURES FROM OUR INTIAL DATASET FOR THE MODEL TO TRAIN UPON.

# In[19]:


Selected_features = ['Participant name', 
                      'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)', 'Gaze event duration',
                     'Pupil diameter left','Pupil diameter right']


# WE CONVERT THE PARTICIPANT NAME COLUMN'S DATATYPE TO INT FOR EASIER DATA MANIPULATION.

# In[20]:


main_df_dataset_II = big_frame_dataset_II[Selected_features].copy(deep=True)
main_df_dataset_II['Participant name'] = main_df_dataset_II['Participant name'].astype(int)


# In[21]:


main_df_dataset_II.head()


# In[22]:


main_df_dataset_II.info()


# In[23]:


print(main_df_dataset_II.shape)


# In[24]:


main_df_dataset_II.isna().sum()


# BELOW GRAPH SHOWS THE PERCENTAGE OF NULL VALUES IN THE SELECTED FEATURES.

# In[25]:


msno.bar(main_df_dataset_II)


# # REMOVING DUPLICATE ROWS

# THE BELOW CODE SNIPPET REMOVES ALL THE DUPLICATE ROWS IN THE DATASET.

# In[26]:


main_df_dataset_II.drop_duplicates(inplace = True)
print('Shape After Removing Duplicates: ',main_df_dataset_II.shape)

main_df_dataset_II.info()
# # CLEANING DATA

# WE CONVERT THE SELECTED FEATURES INTO FLOAT VALUES FOR BETTER DATA MANIPULATION AND INPUTE THE MISSING VALUES IN THE COLOUMN WITH THEIR MEAN VALUES.

# In[27]:


remove_comma_column = ['Gaze point X (MCSnorm)','Gaze point Y (MCSnorm)','Pupil diameter left','Pupil diameter right']

for column_name in remove_comma_column:
    
    #Converting Coloumn to Float and Removing Commas(,) and Replacing with Dot(.)
    main_df_dataset_II[column_name] = main_df_dataset_II[column_name].str.replace(',','.')
    main_df_dataset_II[column_name] = main_df_dataset_II[column_name].astype(float)
    
    #Replacing the NAN values with Mean
    mean = main_df_dataset_II[column_name].mean()
    main_df_dataset_II[column_name] = main_df_dataset_II[column_name].fillna(mean)
    
    print('Mean of Column:',column_name,' is: ',main_df_dataset_II[column_name].mean())
    
main_df_dataset_II


# WE GROUP ALL THE ROWS WITH SAME PARTICIPANTS NAME AND TAKE MEAN OF ALL THEIR OTHER COLUMNS IN ORDER TO REDUCE THE DATASET SIZE.

# In[28]:


main_df_dataset_II = main_df_dataset_II.groupby('Participant name').mean().reset_index()


# In[29]:


main_df_dataset_II.to_csv('main_df_dataset_II.csv')


# In[30]:


main_df_dataset_II.info()


# In[31]:


main_df_dataset_II.isna().sum()


# In[32]:


main_df_dataset_II.head()


# # READING QUESTIONAIRE DATA

# THE RESEARCH HAD ANOTHER DATASET WHICH INCLUDED THE ANSWERS TO A QUESTIONNAIRE BY THE PARTICIPANTS.
# THE BELOW CODE SNIPPET READY THE BEFOREMENTIONED DATASET INTO A DF CALLED "quest_df".

# In[33]:


quest_df = pd.read_csv(r"C:\Users\Akshay Mohan Nair\OneDrive - University of Essex\Spring Term\Decision Making\Assignment_2\unzipped\questionnaire\Questionnaire_datasetIB.csv",encoding='latin-1')
quest_df


# DROPPING THE UNREQUIRED COLUMNS FROM THE QUESTIONNAIRE DATASET AS THEY INCLUDE CREATED DATE, MODIFIED DATE AND NR VALUES.

# In[34]:


quest_df = quest_df.drop(quest_df.columns[[1, 2, 3, 4, 5, 48]], axis=1)


# THE BELOW CODE SNIPPET MERGE THE SELECTED FEAUTRES DATASET WITH THE QUESTIONNAIRE DATASET ON THE BASIS OF THE PARTICIPANT NAME.

# In[35]:


main_df_dataset_II = main_df_dataset_II.merge(quest_df, left_on='Participant name', right_on='Participant nr')


# In[36]:


main_df_dataset_II = main_df_dataset_II.drop('Participant nr', axis=1)


# In[37]:


main_df_dataset_II.to_csv('merged_main_df_dataset_II.csv')


# # SPLITTING THE MERGED DATASET FOR TRAINING AND TEST PURPOSES

# In[38]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(main_df_dataset_II.drop(['Participant name','Total Score original', 'Total Score extended'], axis=1), main_df_dataset_II['Total Score extended'], test_size=0.3, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# # MODEL CREATION FOR DATASET-II

# ## MODEL 1: LINEAR REGRESSION MODEL

# In[39]:


# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


# TRAINING AND MAKING PREDICTIONS USING THE LINEAR REGRESSION MODEL.

# THE BELOW CODE SNIPPET PRIDICTS EMPATHY SCORE USING ALL THE SELECTED FEATURES AND THE QUESTIONNARIE DATASET.

# In[40]:


# Evaluate the model on the testing data
y_pred = lr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)


# BELOW CODE SNIPPED TRIES TO PREDICT EMPATHY SCORE USING JUST THE PUPIL DIAMETER.

# In[41]:


# Extract the input features and the target variable
X = main_df_dataset_II[['Pupil diameter left', 'Pupil diameter right']]
y = main_df_dataset_II['Total Score extended']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and train it on the training set
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Use the trained model to predict on the test set
y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('mse',mse)
# Evaluate the model's performance on the test set using mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
print(f"R SQUARED: {r2}")


# In[42]:


#print(main_df_dataset_II.info())
inputs = main_df_dataset_II.columns[0:4]
target_outputs = main_df_dataset_II.columns[-1:]
inputs


# MAKING EMPAHY SCORE PREDICTIONS USING THE PUPIL DIAMETERS AND THE QUESTIONNAIRE AS SUGGESTED IN THE PAPER.

# In[43]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(main_df_dataset_II.drop(inputs, axis=1), main_df_dataset_II[target_outputs], test_size=0.3, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# In[44]:


print(main_df_dataset_II.drop(inputs, axis=1))


# EMPATHY PREDICTION USING PUPIL DIAMETER AND QUESTIONNAIRE USING LINEAR REGRESSION MODEL.

# In[45]:


# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
# Evaluate the model on the testing data
y_pred = lr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)


# ## MODEL 2: NEURAL NETWORKS

# In[112]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(main_df_dataset_II.drop(['Participant name','Total Score original', 'Total Score extended'], axis=1), main_df_dataset_II['Total Score extended'], test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

main_df_dataset_II
inputs= main_df_dataset_II.columns[4:-2]
outputs = main_df_dataset_II.columns[-1:]
inputs

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(big_frame_dataset_II[inputs], big_frame_dataset_II[outputs], test_size=0.2, random_state=42)

# In[ ]:





# In[104]:


# Defining the model
model = keras.Sequential([
    keras.layers.Conv1D(32, kernel_size=7, activation='relu', input_shape=[X_train_scaled.shape[1], 1]),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=1000, batch_size=4)

# Evaluate the model on test data
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# In[105]:


# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)


# # READING .CSV FILES

# In[148]:


# Get data file names
path = r"C:\Users\Akshay Mohan Nair\OneDrive - University of Essex\Spring Term\Decision Making\Assignment_2\unzipped\EyeT"
filenames = glob.glob(path + "/*dataset_III*.csv")
#filenames = [file for file in filenames if "dataset_II" not in file]

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

# Concatenate all data into one DataFrame
big_frame_dataset_III = pd.concat(dfs, ignore_index=True)


# EXTRACTING NUMERICAL VALUE FOR THE PARTICIPANT CODE

# In[149]:


# Extract the last numeric characters from 'Participant name' column
big_frame_dataset_III['Participant name'] = big_frame_dataset_III['Participant name'].str[-2:]


# In[150]:


big_frame_dataset_III = big_frame_dataset_III.dropna(subset=['Pupil diameter left', 'Pupil diameter right'])
big_frame_dataset_III.shape


# ## GRAPHS FOR DATASET-III (TEST GROUP)

# BELOW GRAPH SHOWS PERCENTAGE OF NULL VALUES IN ALL THE COLUMNS OF THE DATASET-III.

# In[151]:


msno.bar(big_frame_dataset_III)


# GRAPH FOR THE FLUCTUATION IN LEFT EYE'S PUPIL DIAMETER WITH RESPECT TO TIMESTAMP FOR TEST GROUP DATASET.

# In[152]:


big_frame_dataset_III.plot(x="Pupil diameter left",y= "Eyetracker timestamp", kind="line", figsize=(5, 5))


# GRAPH FOR THE FLUCTUATION IN RIGHT EYE'S PUPIL DIAMETER WITH RESPECT TO TIMESTAMP FOR TEST GROUP DATASET.

# In[153]:


big_frame_dataset_III.plot(x="Pupil diameter right",y= "Eyetracker timestamp", kind="line", figsize=(5, 5))


# CORELATION MATRIX FOR THE WHOLE DATASET OF TEST GROUP

# In[154]:


matrix_co=big_frame_dataset_III.corr()
sns.heatmap(matrix_co, cmap="gist_gray",linewidth=.5) #cmap = 'crest' for B and W heatmap
plt.show()


# BAR GRAPH FOR THE COLUMNS IN THE DATASET-III.

# In[155]:


big_frame_dataset_III.hist(bins=50, figsize=(20,15))


# PIE CHART FOR THE DIFFERENT VALUES IN THE COLUMN 'Eye movement type' IN DATASET-III

# In[156]:


pie_chart = big_frame_dataset_III['Eye movement type'].value_counts(dropna=False)
print(pie_chart)
pie_chart.plot(kind="pie", autopct='%1.1f%%',    ylabel='EYE MOVEMENT TYPE VALUE COUNTS')


# BELOW TABLE SHOWS THAT THE ADDED VALUE OF "GAZE POINT LEFT X" AND "SUM OF GAZE POINT RIGHT X" IS EQUAL TO "GAZE POINT X" IN DATASET-III

# In[157]:


#Checking Values of Gaze Point X and Its Combined Value of Left and Right Eye
check_gaze = pd.DataFrame()
check_gaze['Added_value']=(big_frame_dataset_III['Gaze point left X']+ big_frame_dataset_III['Gaze point right X'])/2
check_gaze['Gaze point X'] = big_frame_dataset_III['Gaze point X']
check_gaze.head()


# BELOW CODE SHOWS THAT THE VALUE OF SUM OF "GAZE POINT LEFT X" AND "SUM OF GAZE POINT RIGHT X" IS EQUAL TO GAZE POINT X. THEREFORE THERE IS NO POINT IN TAKING THESE INDIVIDUAL COLUMNS INTO OUT FINAL DATASET.
# 
# THE VIOLET LINE JUST SHOWS THAT THE VALUES PERFECTLY OVERLAP EACH OTHER AS THE INDIVUAL COLOURS VALUES FOR THE LINES WERE RED AND BLUE.

# In[158]:


fig, ax = plt.subplots(figsize=(5,5))
check_gaze.plot(x="Added_value",kind="line", ax=ax)
check_gaze.plot(x="Gaze point X",kind="line", color="r", ax=ax,alpha=0.5)
plt.show()


# # FEATURE SELECTION FOR DATASET-III

# FROM THE ABOVE GRAPHS WE DECIDE TO SELECT THE BELOW FEATURES FROM OUR INTIAL DATASET FOR THE MODEL TO TRAIN UPON.

# In[159]:


Selected_features = ['Participant name', 
                      'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)', 'Gaze event duration',
                     'Pupil diameter left','Pupil diameter right']


# WE CONVERT THE PARTICIPANT NAME COLUMN'S DATATYPE TO INT FOR EASIER DATA MANIPULATION.

# In[160]:


big_frame_dataset_III = big_frame_dataset_III[Selected_features].copy(deep=True)
big_frame_dataset_III['Participant name'] = big_frame_dataset_III['Participant name'].astype(int)


# In[161]:


big_frame_dataset_III.info()


# BELOW GRAPH SHOWS THE PERCENTAGE OF NULL VALUES IN THE SELECTED FEATURES IN DATASET-III.

# In[162]:


msno.bar(big_frame_dataset_III)


# # REMOVING DUPLICATE ROWS

# THE BELOW CODE SNIPPET REMOVES ALL THE DUPLICATE ROWS IN THE DATASET-III.

# In[163]:


big_frame_dataset_III.drop_duplicates(inplace = True)
print('Shape After Removing Duplicates: ',big_frame_dataset_III.shape)


# # CLEANING DATA FOR DATASET-III

# WE CONVERT THE SELECTED FEATURES INTO FLOAT VALUES FOR BETTER DATA MANIPULATION AND INPUTE THE MISSING VALUES IN THE COLOUMN WITH THEIR MEAN VALUES.

# In[164]:


remove_comma_column = ['Gaze point X (MCSnorm)','Gaze point Y (MCSnorm)','Pupil diameter left','Pupil diameter right']

for column_name in remove_comma_column:
    
    #Converting Coloumn to Float and Removing Commas(,) and Replacing with Dot(.)
    big_frame_dataset_III[column_name] = big_frame_dataset_III[column_name].str.replace(',','.')
    big_frame_dataset_III[column_name] = big_frame_dataset_III[column_name].astype(float)
    
    #Replacing the NAN values with Mean
    mean = big_frame_dataset_III[column_name].mean()
    big_frame_dataset_III[column_name] = big_frame_dataset_III[column_name].fillna(mean)
    
    print('Mean of Column:',column_name,' is: ',big_frame_dataset_III[column_name].mean())
    
big_frame_dataset_III


# WE GROUP ALL THE ROWS WITH SAME PARTICIPANTS NAME AND TAKE MEAN OF ALL THEIR OTHER COLUMNS IN ORDER TO REDUCE THE DATASET SIZE.

# In[165]:


big_frame_dataset_III = big_frame_dataset_III.groupby('Participant name').mean().reset_index()


# In[166]:


big_frame_dataset_III.to_csv('main_df_dataset_III.csv')


# # MERGING DATASET-III WITH QUESTIONNAIRE

# THE BELOW CODE SNIPPET MERGE THE SELECTED FEAUTRES DATASET WITH THE QUESTIONNAIRE DATASET ON THE BASIS OF THE PARTICIPANT NAME.

# In[167]:


big_frame_dataset_III = big_frame_dataset_III.merge(quest_df, left_on='Participant name', right_on='Participant nr')
big_frame_dataset_III = big_frame_dataset_III.drop('Participant nr', axis=1)


# In[168]:


big_frame_dataset_III.to_csv('merged_main_df_dataset_III.csv')


# # SPLITTING THE MERGED DATASET FOR TRAINING AND TEST PURPOSES

# In[173]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(big_frame_dataset_III.drop(['Participant name','Total Score original', 'Total Score extended'], axis=1), big_frame_dataset_III['Total Score extended'], test_size=0.3, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # MODEL CREATION FOR TEST GROUP

# ## MODEL 1: LINEAR REGRESSION MODEL FOR TEST GROUP

# In[174]:


# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


# TRAINING AND MAKING PREDICTIONS USING THE LINEAR REGRESSION MODEL.
# THE BELOW CODE SNIPPET PRIDICTS EMPATHY SCORE USING ALL THE SELECTED FEATURES AND THE QUESTIONNARIE DATASET.

# In[175]:


# Evaluate the model on the testing data
y_pred = lr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)


# BELOW CODE SNIPPED TRIES TO PREDICT EMPATHY SCORE USING JUST THE PUPIL DIAMETER FOR TEST GROUP DATA.

# In[177]:


# Extract the input features and the target variable
X = big_frame_dataset_III[['Pupil diameter left', 'Pupil diameter right']]
y = big_frame_dataset_III['Total Score extended']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and train it on the training set
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Use the trained model to predict on the test set
y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('mse',mse)
# Evaluate the model's performance on the test set using mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")
print(f"R SQUARED: {r2}")


# In[179]:


inputs = big_frame_dataset_III.columns[0:4]
target_outputs = big_frame_dataset_III.columns[-1:]


# MAKING EMPAHY SCORE PREDICTIONS USING THE PUPIL DIAMETERS AND THE QUESTIONNAIRE AS SUGGESTED IN THE PAPER.

# In[180]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(big_frame_dataset_III.drop(inputs, axis=1), big_frame_dataset_III[target_outputs], test_size=0.3, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# EMPATHY PREDICTION USING PUPIL DIAMETER AND QUESTIONNAIRE USING LINEAR REGRESSION MODEL.

# In[181]:


# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
# Evaluate the model on the testing data
y_pred = lr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)


# ## MODEL 2: NEURAL NETWORKS FOR TEST GROUP

# In[182]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(big_frame_dataset_III.drop(['Participant name','Total Score original', 'Total Score extended'], axis=1), big_frame_dataset_III['Total Score extended'], test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[197]:


# Defining the model
model = keras.Sequential([
    keras.layers.Conv1D(32, kernel_size=7, activation='relu', input_shape=[X_train_scaled.shape[1], 1]),
    keras.layers.MaxPooling1D(pool_size=2),
    #keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    #keras.layers.MaxPooling1D(pool_size=2),
    #keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    #keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=1000, batch_size=4)

# Evaluate the model on test data
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# In[198]:


# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)


# In[ ]:





# In[ ]:




