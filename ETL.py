##########################################################################################
############# This file performs ETL on the US Road Accidents dataset from Kaggle ########
##########################################################################################

# Import dependencies
import pandas as pd

# Import SQL Load dependencies
import psycopg2
from config import db_password

#####################################################################
#########  DATA EXTRACTION  #########################################
#####################################################################

# Extract Kaggle Data

file_dir = '.'
kaggle_metadata = pd.read_csv(f'{file_dir}/US_Accidents_Dec19.tar.gz', 
                              compression='gzip', error_bad_lines=False, low_memory=False)

### Turn off SettingWithCopyWarning ##############
pd.options.mode.chained_assignment = None

# Check total rows extraced
len(kaggle_metadata)

# Extract only Columns relevant for analysis
df_subset = kaggle_metadata[['US_Accidents_Dec19.csv','Severity','Start_Time','End_Time',
                             'Start_Lat','Start_Lng','Distance(mi)', 'Street','Side','City',
                             'County','State','Zipcode','Timezone', 
                             'Temperature(F)','Humidity(%)','Pressure(in)',
                             'Visibility(mi)','Wind_Direction','Wind_Speed(mph)','Precipitation(in)',
                             'Weather_Condition','Amenity','Crossing','Junction','Railway',
                             'Station','Stop','Traffic_Signal','Civil_Twilight'
                            ]]

#####################################################################
#########  TRANSFORM  ###############################################
#####################################################################

# Check Null Values
df_subset.isnull().sum()

# Fill NA with zero values for Precipitation column
df_subset["Precipitation(in)"].fillna(0, inplace = True) 

# Drop rows with other NA values
df_subset.dropna(inplace=True)

# Check the resulting dataset length
len(df_subset)

# Sort the dataframe on Severity so that when removing duplicates the one with higher severity is retained
sorted_df = df_subset.sort_values('Severity',ascending=False)

# Check how many duplicates exist in the dataset
len(sorted_df[['Severity', 'Start_Time', 'Start_Lat', 'Start_Lng']].drop_duplicates())

# Remove duplicates
sorted_df.drop_duplicates(subset=['Severity', 'Start_Time', 'Start_Lat', 'Start_Lng'], inplace = True) 

# Extract first 5 digits of zipcode where zip code is in postal format of ZIP-4
sorted_df['Zipcode'] = sorted_df['Zipcode'].str.replace(r"-.*","")

# Check length of the remaining dataset after removing duplicate and Null value rows
len(sorted_df)

# check datatypes for corrections
sorted_df.dtypes

# Change datatype for Start and End Time columns
sorted_df['Start_Time'] = pd.to_datetime(sorted_df.Start_Time)
sorted_df['End_Time'] = pd.to_datetime(sorted_df.End_Time)

# Change Datatype for Severity from float to integer
sorted_df['Severity'] = sorted_df['Severity'].astype(int)

#########################   Create Highway and Coordinates Column   ######################


searchfor = ['highway', 'Tollway', 'expy', 'fwy', 'hwy', 'Interstate', 
             'Tpke', 'Pkwy', 'Parkway', '-', 'US', 'Route', 
             'FM', 'Byp', 'Trwy', 'Beltway', 'Skyway', 'Skwy', ]
sorted_df.loc[sorted_df['Street'].str.contains('|'.join(searchfor), case=False), 'Highway'] = 'Y'

# Fill NA with zero values for Highway column
sorted_df["Highway"].fillna('N', inplace = True) 

# Create Coordinates column
sorted_df['Coordinates'] = sorted_df['Start_Lat'].map(str) + ':' + sorted_df['Start_Lng'].map(str)

##################  Prep Data and Create Dataframes for different SQL tables #############

# Rename columns
sorted_df = sorted_df.rename(index=str,columns={'US_Accidents_Dec19.csv': 'Accident_ID'})

# Create Dataframes for Loading into SQL Tables 
table1_df = sorted_df[['Accident_ID','Severity','Start_Time','End_Time',
                             'Start_Lat','Start_Lng','Coordinates', 'Distance(mi)', 'Side', 
                             'Temperature(F)','Humidity(%)','Pressure(in)',
                             'Visibility(mi)','Wind_Direction','Wind_Speed(mph)','Precipitation(in)',
                             'Weather_Condition','Amenity','Crossing','Junction','Railway',
                             'Station','Stop','Traffic_Signal','Civil_Twilight'
                             ]]
table2_df = sorted_df[['Coordinates', 'Street','City','County','State','Zipcode',
                       'Timezone', 'Highway']]

# Drop duplicate columns from table 2 dataframe
table2_df.drop_duplicates(subset=['Coordinates'], inplace=True)

# Rename dataframe columns to match the table column names which are all in lower case
table2_df.columns = map(str.lower, table2_df.columns)

# Rename dataframe columns to match the table column names
table1_df = table1_df.rename(index=str,columns={'Distance(mi)':'distance', 
                             'Temperature(F)':'temperature',
                             'Humidity(%)':'humidity',
                             'Pressure(in)':'pressure',
                             'Visibility(mi)':'visibility',
                             'Wind_Speed(mph)':'wind_speed',
                             'Precipitation(in)':'precipitation',
                             })
table1_df.columns = map(str.lower, table1_df.columns)

# Set Index
table1_df.set_index('accident_id', inplace=True)
table2_df.set_index('coordinates', inplace=True)

# Since very large dataset Create CSVs to load data into Postgres SQL tables
table1_df.to_csv('table1.csv', sep='|')
table2_df.to_csv('table2.csv',sep='|')

########################################################################################
##############  LOAD  DATA to Postgres SQL Tables on AWS ###############################
########################################################################################

# Create connection string and engine
conn = psycopg2.connect(
    host="accident-viz.c4cdhyeva5ut.us-east-1.rds.amazonaws.com", 
    port='5432', 
    dbname="Accident-ETL", 
    user="postgres", 
    password=db_password
)
cur = conn.cursor()

# Load data from table2_df into the accident_location table
with open('table2.csv', 'r') as f:
    next(f) # Skip the header row.
    cur.copy_from(f, 'accident_location', sep='|')
    conn.commit()

# Load data from table1_df into the accident table
with open('table1.csv', 'r') as f:
    next(f) # Skip the header row.
    cur.copy_from(f, 'accidents', sep='|')
    conn.commit()


##############  THE END  ##################################################################
###########################################################################################