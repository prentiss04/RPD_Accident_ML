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
import numpy as np
import pandas as pd
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Import Dependencies
import time

# Convert the Spark DataFrame to a pandas DataFrame using Arrow
pandas_df = pd.DataFrame()
#result_pdf = spark_df.toPandas()

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

X_df.dtypes
#X_df.filter(X_df("distance")>0.5).show()

# Determine number of unique weather_conditions
X_df['weather_condition'].nunique()

# for binary, lambda fxn true=1, f=0 (side, crossing, junction, amenity, railway, station, stop, traffic_signal, highway - check civil_twilight)
def convert_binary(feature):
  if "False" in feature:
    return 0
  else:
    return 1

X_df['crossing'] = X_df['crossing'].replace({'True':1, 'False':0})
X_df['highway'] = X_df['highway'].replace({'Y':1, 'N':0})
X_df['junction'] = X_df['junction'].replace({'True':1, 'False':0})
X_df['amenity'] = X_df['amenity'].replace({'True':1, 'False':0})
X_df['railway'] = X_df['railway'].replace({'True':1, 'False':0})
X_df['station'] = X_df['station'].replace({'True':1, 'False':0})
X_df['stop'] = X_df['stop'].replace({'True':1, 'False':0})
X_df['traffic_signal'] = X_df['traffic_signal'].replace({'True':1, 'False':0})
X_df['civil_twilight'] = X_df['civil_twilight'].replace({'Night':1, 'Day':0})

X_df.head()

# Create column for start time in seconds
df_time = pd.to_datetime(pandas_df["start_time"])

X_df["time_in_second"] = (df_time.dt.hour*60+df_time.dt.minute)*60 + df_time.dt.second

# Create separate columns for start/end date/time
X_df['new_start_date'] = [d.date() for d in X_df["start_time"]] 
X_df['new_start_time'] = [d.time() for d in X_df["start_time"]] 

X_df['new_end_date'] = [d.date() for d in X_df["end_time"]] 
X_df['new_end_time'] = [d.time() for d in X_df["end_time"]]

# Convert newly created date objects to date formate
X_df['new_start_date'] =  X_df['new_start_date'].astype('datetime64[ns]')
X_df['new_end_date'] =  X_df['new_end_date'].astype('datetime64[ns]')

# Create column for day of the week
X_df["start_day_of_week"] = X_df['new_start_date'].dt.weekday
X_df["end_day_of_week"] = X_df['new_end_date'].dt.weekday

# Determine duration of traffic event
X_df['duration'] = X_df['end_time'] - X_df['start_time']
X_df.head()
# convert to seconds?

X_df['duration_seconds'] = X_df['duration'].dt.total_seconds()
X_df.head()

# Create DF of unique weather conditions and counts, write to csv for categorization
# Manual preprocessing to group weather_classifications based on personal judgement into ten categories.
# change to read.csv (for .py)

weather_unique = X_df['weather_condition'].value_counts().rename_axis('unique_values').reset_index(name='counts')
#weather_unique.to_csv('weather_condition.csv')
weather_unique.to_csv(r'./weather_condition.csv', index = False, header = True)

# Read in weather_condition_categories.csv file (just the weather_condition_categories tab not the pivot table tab)

# Create DF of unique weather conditions and counts, write to csv for categorization
# Read sorted weather_conditions_categories from folder
conditions_cat_df = pd.read_csv('./weather_condition_categories.csv')

X_df = pd.merge(X_df, conditions_cat_df, how="left", left_on="weather_condition", right_on="unique_values")
X_df.head()

X_df.rename(columns= {'category': "weather_category"}, inplace=True)

#X_df['side'] = X_df['side'].astype(str).astype(float)
X_df['wind_speed'] = X_df['wind_speed'].astype(str).astype(float)
X_df['precipitation'] = X_df['precipitation'].astype(str).astype(float)
X_df['amenity'] = X_df['amenity'].astype(str).astype(float)
X_df['crossing'] = X_df['crossing'].astype(str).astype(float)
X_df['junction'] = X_df['junction'].astype(str).astype(float)
X_df['railway'] = X_df['railway'].astype(str).astype(float)
X_df['station'] = X_df['station'].astype(str).astype(float)
X_df['civil_twilight'] = X_df['civil_twilight'].astype(str).astype(float)
#X_df['highway'] = X_df['highway'].astype(str).astype(float)

X_df2 = X_df.drop(['start_time', 'end_time', 'weather_condition', 'new_start_date', 'new_start_time', 'new_end_date', 'new_end_time', 'end_day_of_week', 'Unnamed: 0','unique_values','counts', 'duration'], axis=1)

X_df2.head()

# apply get_dummies to wind_direction, timezone, weather_condition
X_dummies_df = pd.get_dummies(X_df2, columns=["side","wind_direction", "timezone","start_day_of_week"])

X_dummies_df.head()

# principal comp analysis - when there are untenable # of features

"""# Determine which columns to drop, add and which to create dummies"""

# Create target and feature sets
y = X_dummies_df["severity"]
X = X_dummies_df.drop(columns="severity")

"""# **Machine Learning**"""

# Dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split dataset into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Determine the shape of our training and testing sets.
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# Creating a StandardScaler instance.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaler = scaler.fit(X_train)

# Scaling the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', random_state=1)
classifier

# Train the model with training set
classifier.fit(X_train, y_train)

# Create predictions and assemble results
predictions_LR = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions_LR, "Actual": y_test})

# Calculating the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, predictions_LR)

# Create a DataFrame from the confusion matrix
cm_df = pd.DataFrame(cm, index=["Actual 1", "Actual 2", "Actual 3", "Actual 4"], columns=["Predicted 1", "Predicted 2", "Predicted 3", "Predicted 4"])

cm_df

# Calculating the accuracy score.
acc_score_LR = accuracy_score(y_test, predictions_LR)

# Displaying results
print("LR Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score_LR}")
print("Classification Report")
print(classification_report(y_test, predictions_LR))

# Calculate feature importance in the model

# Sort the features by their importances

"""# **Oversampling**"""

# Oversample to see if results improve

from collections import Counter
Counter(y_train)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = 78)
X_resampled, y_resampled, = ros.fit_resample(X_train, y_train)

Counter(y_resampled)

start_time = time.time()
from sklearn.linear_model import LogisticRegression
model_OS = LogisticRegression(solver='lbfgs', random_state=78, max_iter=1000)
model_OS.fit(X_resampled, y_resampled)
print ("My program took", time.time() - start_time, "to run")

from sklearn.metrics import confusion_matrix
y_pred_OS = model_OS.predict(X_test)
confusion_matrix(y_test, y_pred_OS)

from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred_OS)

from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred_OS))

"""# **Decision Tree Classifier**"""

# Creating the decision tree classifier instance.
from sklearn import tree

model_tree = tree.DecisionTreeClassifier()

# Fitting the model.
model_tree = model_tree.fit(X_train_scaled, y_train)

# Making predictions using the testing data.
predictions_tree = model_tree.predict(X_test_scaled)

predictions_tree

# Calculating the confusion matrix
cm_tree = confusion_matrix(y_test, predictions_tree)

# Create a DataFrame from the confusion matrix
cm_tree_df = pd.DataFrame(cm_tree, index=["Actual 1", "Acutal 2", "Actual 3", "Actual 4"], columns=["Predicted 1", "Predicted 2", "Predicted 3", "Predicted 4"])

cm_tree_df

# Calculating the accuracy score.
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

"""# **Neural network (19.4.4)**"""

# Install dependencies
import matplotlib as plt
import tensorflow as tf

# Define the model - deep neural net
number_input_features = len(X_train.columns)
hidden_nodes_layer1 = 20
hidden_nodes_layer2 = 10

nn = tf.keras.models.Sequential()

#from keras.layers import LeakyReLU
# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="softmax"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="softmax"))

# Output layer
nn.add(tf.keras.layers.Dense(units=4, activation="softmax"))

# Check the structure of the model
nn.summary

# Decrease y_train results by 1 to achieve lowest score = zero
y_train2 = y_train -1
y_train2.head()

# Convert labels to categorical one-hot encoding
one_hot_labels = tf.keras.utils.to_categorical(y_train2, num_classes=4)

# Compile the model
nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn.fit(X_train_scaled, one_hot_labels, epochs=20)

# Decrease y_test results by 1 to achieve lowest score = zero
y_test2 = y_test -1
y_test2.head()

# Convert y_test to categorical one-hot encoding
one_hot_labels_test = tf.keras.utils.to_categorical(y_test2, num_classes=4)
print(one_hot_labels_test)

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, one_hot_labels_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

predictions_nn = nn.predict(X_test_scaled)
predictions_nn

matrix = tf.math.confusion_matrix(one_hot_labels_test.argmax(axis=1), predictions_nn.argmax(axis=1), num_classes=4)
matrix

# Displaying results

with tf.Session():
  print('Confusion Matrix: \n\n', tf.Tensor.eval(matrix))

print(f"Accuracy Score : %f" % model_accuracy)
print("Classification Report")
#print(classification_report(y_test, predictions))

# https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
# Precision, Recall & F-1 Scores
#precision_nn = precision_score(y_test, y_col)
import sklearn
from sklearn.metrics import recall_score
sklearn.metrics.recall_score(y_test2, predictions_nn)

from sklearn.metrics import classification_report

y_pred = nn.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool))

Precision_nn = matrix.metrics.Precision()

# tf.math.confusion_matrix(one_hot_labels_test.argmax(axis=1), predictions_nn.argmax(axis=1), num_classes=4)

# Matrix Recall & Precision math
import numpy as np

#recall = np.diag(matrix) / np.sum(matrix, axis =1)
#precision = np.diag(matrix) / np.sum(matrix, axis =0)
# https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co

np.mean(recall)
np.mean(precision)

"""# **Perform LR on smaller set**"""

# Copy DF
# Filter out 1s and 2s
# Reattempt nn with just two traffic categories

# Create filtered DF with only 3 & 4 level accidents
bad_traffic_df = X_dummies_df[X_dummies_df.severity >= 3]
bad_traffic_df.head()

# Drop the severity column for the output
y_bt = bad_traffic_df["severity"]
X_bt = bad_traffic_df.drop(columns="severity")

# Split dataset into train & test sets
X_bt_train, X_bt_test, y_bt_train, y_bt_test = train_test_split(X_bt, y_bt, random_state=78)

X_bt_scaler = scaler.fit(X_bt_train)
#does the test set need to be fit as well?

# Scaling the data
X_bt_train_scaled = X_bt_scaler.transform(X_bt_train)
X_bt_test_scaled = X_bt_scaler.transform(X_bt_test)

#classifier = LogisticRegression(solver='lbfgs', random_state=1)
#classifier

# Create the logistic regression model
#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None, penalty='12', random_state=1, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

# Train the model with training set
classifier.fit(X_bt_train, y_bt_train)

# Create predictions and assemble results
predictions_bt = classifier.predict(X_bt_test)
pd.DataFrame({"Prediction": predictions_bt, "Actual": y_bt_test})

# Calculating the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm_bt = confusion_matrix(y_bt_test, predictions_bt)

# Create a DataFrame from the confusion matrix
cm_bt_df = pd.DataFrame(cm_bt, index=["Actual 3", "Actual 4"], columns=["Predicted 3", "Predicted 4"])

cm_bt_df

# Calculating the accuracy score.
acc_score_bt = accuracy_score(y_bt_test, predictions_bt)

# Displaying results
print("Confusion Matrix")
display(cm_bt_df)
print(f"Accuracy Score : {acc_score_bt}")
print("Classification Report")
print(classification_report(y_bt_test, predictions_bt))