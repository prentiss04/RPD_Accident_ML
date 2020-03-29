#Install Java, Spark, and Findspark
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://www-us.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
!tar xf spark-2.4.5-bin-hadoop2.7.tgz
!pip install -q findspark

# Set Environment Variables
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.5-bin-hadoop2.7"
# Start a SparkSession
import findspark
findspark.init()
!wget https://jdbc.postgresql.org/download/postgresql-42.2.9.jar
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("BigDataHW").config("spark.driver.extraClassPath","/content/postgresql-42.2.9.jar").getOrCreate()

# Import Dependencies
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib as plt
import tensorflow as tf
import time

# Configure settings for RDS
#mode = "append"
from config import db_password

jdbc_url="jdbc:postgresql://accident-viz.c4cdhyeva5ut.us-east-1.rds.amazonaws.com:5432/Accident-ETL"
config = {"user":"postgres",
          "password": db_password,
          "driver":"org.postgresql.Driver"}

# Read in dataset from SQL with join between accident table and accident_location table
sql_query = "(SELECT acc.*, acc_loc.highway, acc_loc.timezone FROM accidents acc JOIN accident_location acc_loc on acc.coordinates = acc_loc.coordinates) as accdnts"
spark_df = spark.read.jdbc(url=jdbc_url, table = sql_query, column = 'severity', lowerBound =  1, upperBound = 4, numPartitions = 4, properties=config)

######## Convert Spark DF to Pandas DF
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Convert the Spark DataFrame to a pandas DataFrame
pandas_df = pd.DataFrame()
n = spark_df.count()

chunksize = 100000
for i in range (0, n, chunksize):
  chunk = spark_df.filter(spark_df["rowid"].between(i, i + chunksize))
  pd_df = chunk.toPandas()
  pandas_df= pandas_df.append(pd_df)

# Check the size of the converted DataFrame
len(pandas_df)

# Drop columns
X_df = pandas_df.drop(['coordinates','accident_id', 'rowid'], axis=1)

# Determine column datatypes for conversion
X_df.dtypes

# Determine number of unique weather_conditions
X_df['weather_condition'].nunique()

# for binary column features, lambda fxn true=1, f=0 (crossing, junction, amenity, railway, station, stop, traffic_signal, highway)
X_df['crossing'] = X_df['crossing'].replace({'True':1, 'False':0})
X_df['highway'] = X_df['highway'].replace({'Y':1, 'N':0})
X_df['junction'] = X_df['junction'].replace({'True':1, 'False':0})
X_df['amenity'] = X_df['amenity'].replace({'True':1, 'False':0})
X_df['railway'] = X_df['railway'].replace({'True':1, 'False':0})
X_df['station'] = X_df['station'].replace({'True':1, 'False':0})
X_df['stop'] = X_df['stop'].replace({'True':1, 'False':0})
X_df['traffic_signal'] = X_df['traffic_signal'].replace({'True':1, 'False':0})
X_df['civil_twilight'] = X_df['civil_twilight'].replace({'Night':1, 'Day':0})

# Create column for start time in seconds
df_time = pd.to_datetime(pandas_df["start_time"])
X_df["time_in_second"] = (df_time.dt.hour*60+df_time.dt.minute)*60 + df_time.dt.second

# Create separate columns for start/end date/time
# Start date & time
X_df['new_start_date'] = [d.date() for d in X_df["start_time"]] 
X_df['new_start_time'] = [d.time() for d in X_df["start_time"]] 

# End date & time
X_df['new_end_date'] = [d.date() for d in X_df["end_time"]] 
X_df['new_end_time'] = [d.time() for d in X_df["end_time"]]

# Convert newly created date objects to date format
X_df['new_start_date'] =  X_df['new_start_date'].astype('datetime64[ns]')
X_df['new_end_date'] =  X_df['new_end_date'].astype('datetime64[ns]')

# Create columns for day of the week
X_df["start_day_of_week"] = X_df['new_start_date'].dt.weekday
X_df["end_day_of_week"] = X_df['new_end_date'].dt.weekday

# Determine duration of traffic event
X_df['duration'] = X_df['end_time'] - X_df['start_time']
X_df['duration_seconds'] = X_df['duration'].dt.total_seconds()

# Create DF of unique weather conditions and counts, write to csv for categorization
# From earlier determination of abundance of unique weather_conditions (>100)
# Manual preprocessing to group weather_classifications based on personal judgement into ten categories.
weather_unique = X_df['weather_condition'].value_counts().rename_axis('unique_values').reset_index(name='counts')
weather_unique.to_csv(r'./weather_condition.csv', index = False, header = True)

# Read in weather_condition_categories.csv file
conditions_cat_df = pd.read_csv('./weather_condition_categories.csv')

# Merge existing X_df and new conditions_cat_df on the X_df.weather_condition & conditions_cat_df.unique_values
X_df = pd.merge(X_df, conditions_cat_df, how="left", left_on="weather_condition", right_on="unique_values")

# Rename category to weather_category
X_df.rename(columns= {'category': "weather_category"}, inplace=True)

# Cast string features as float in preparation for scaling
X_df['wind_speed'] = X_df['wind_speed'].astype(str).astype(float)
X_df['precipitation'] = X_df['precipitation'].astype(str).astype(float)
X_df['amenity'] = X_df['amenity'].astype(str).astype(float)
X_df['crossing'] = X_df['crossing'].astype(str).astype(float)
X_df['junction'] = X_df['junction'].astype(str).astype(float)
X_df['railway'] = X_df['railway'].astype(str).astype(float)
X_df['station'] = X_df['station'].astype(str).astype(float)
X_df['civil_twilight'] = X_df['civil_twilight'].astype(str).astype(float)

# Drop unnecessary columns and create separate DataFrame
X_df2 = X_df.drop(['start_time', 'end_time', 'weather_condition', 'new_start_date', 'new_start_time', 'new_end_date', 'new_end_time', 'end_day_of_week', 'Unnamed: 0','unique_values','counts', 'duration'], axis=1)

# Apply get_dummies to wind_direction, timezone, weather_condition & create separate DF for ML
X_dummies_df = pd.get_dummies(X_df2, columns=["side","wind_direction", "timezone","start_day_of_week"])

###################################################################
########### Machine Learning ######################################
###################################################################

# Create target and feature sets
y = X_dummies_df["severity"]
X = X_dummies_df.drop(columns="severity")

# Split dataset into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Determine the shape of our training and testing sets.
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Creating a StandardScaler instance.
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

###################################
##### Linear Regression Model #####
###################################

classifier = LogisticRegression(solver='lbfgs', random_state=1)
classifier

# Train the model with training set
classifier.fit(X_train, y_train)

# Create predictions and assemble results
predictions_LR = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions_LR, "Actual": y_test})

# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions_LR)

# Create a DataFrame from the confusion matrix
cm_df = pd.DataFrame(cm, index=["Actual 1", "Actual 2", "Actual 3", "Actual 4"], columns=["Predicted 1", "Predicted 2", "Predicted 3", "Predicted 4"])

# Calculating the accuracy score.
acc_score_LR = accuracy_score(y_test, predictions_LR)

# Displaying results
print("LR Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score_LR}")
print("Classification Report")
print(classification_report(y_test, predictions_LR))

####################
### Oversampling ###
####################
# Low recall for Levels 1,3 & 4 in LR model suggest a model with better distribution may improve results

# Check incident count for training set
Counter(y_train)

# Instantiate RandomOverSample instance
ros = RandomOverSampler(random_state = 78)

# Resample the original training set & rename
X_resampled, y_resampled, = ros.fit_resample(X_train, y_train)

# Confirm resampled training set size
Counter(y_resampled)

# Instantiate & train the Oversampled model
model_OS = LogisticRegression(solver='lbfgs', random_state=78, max_iter=1000)
model_OS.fit(X_resampled, y_resampled)

# Create predictions
y_pred_OS = model_OS.predict(X_test)
confusion_matrix(y_test, y_pred_OS)

# Determine accuracy score
balanced_accuracy_score(y_test, y_pred_OS)

# Provide summary classification report
print(classification_report_imbalanced(y_test, y_pred_OS))

#######################################
###### Decision Tree Classifier #######
#######################################

# Creating the decision tree classifier instance.
model_tree = tree.DecisionTreeClassifier()

# Fitting the model.
model_tree = model_tree.fit(X_train_scaled, y_train)

# Making predictions using the testing data.
predictions_tree = model_tree.predict(X_test_scaled)

# Calculating the confusion matrix
cm_tree = confusion_matrix(y_test, predictions_tree)

# Create a DataFrame from the confusion matrix
cm_tree_df = pd.DataFrame(cm_tree, index=["Actual 1", "Acutal 2", "Actual 3", "Actual 4"], columns=["Predicted 1", "Predicted 2", "Predicted 3", "Predicted 4"])

# Calculate the accuracy score.
acc_score_tree = accuracy_score(y_test, predictions_tree)

# Displaying results
print("DTC Confusion Matrix")
display(cm_tree_df)
print(f"Accuracy Score : {acc_score_tree}")
print(f"Accuracy Score : %f" % acc_score_tree)
print("Classification Report")
print(classification_report(y_test, predictions_tree))

# Calculate feature importance in the model
importances_tree = model_tree.feature_importances_
importances_tree

# Sort the features by their importances
sorted(zip(model_tree.feature_importances_, X.columns), reverse=True)

################################
######## Neural Network ########
################################

# Define the model - deep neural net
number_input_features = len(X_train.columns)
hidden_nodes_layer1 = 20
hidden_nodes_layer2 = 10

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="softmax"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="softmax"))

# Output layer
nn.add(tf.keras.layers.Dense(units=4, activation="softmax"))

# Check the structure of the model
nn.summary

# Decrease y_train results by 1 to achieve lowest score = zero
# NN expects min classification to be zero. As Severity ranges from 1:4, adjustment is necessary
y_train2 = y_train -1

# Convert labels to categorical one-hot encoding
one_hot_labels = tf.keras.utils.to_categorical(y_train2, num_classes=4)

# Compile the model
nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn.fit(X_train_scaled, one_hot_labels, epochs=20)

# Decrease y_test results by 1 to achieve lowest score = zero (similar to train set modification)
y_test2 = y_test -1

# Convert y_test to categorical one-hot encoding
one_hot_labels_test = tf.keras.utils.to_categorical(y_test2, num_classes=4)
print(one_hot_labels_test)

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, one_hot_labels_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Create Predictions array
predictions_nn = nn.predict(X_test_scaled)

# Create matrix of test results array and predictions array
matrix = tf.math.confusion_matrix(one_hot_labels_test.argmax(axis=1), predictions_nn.argmax(axis=1), num_classes=4)

# Displaying results
with tf.Session():
  print('Confusion Matrix: \n\n', tf.Tensor.eval(matrix))

print(f"Accuracy Score : %f" % model_accuracy)