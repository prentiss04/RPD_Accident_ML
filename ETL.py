##########################################################################################
############# This file performs ETL on the US Road Accidents dataset from Kaggle ########
##########################################################################################

# Import dependencies
import pandas as pd

#####################################################################
#########  DATA EXTRACTION  #########################################
#####################################################################

# Extract Kaggle Data

file_dir = 'C:/Users/ruchi/Desktop/Berkley Extension Learning Docs/Final Project'
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

#########################   Create Highway and Coordinates Column   ######################


searchfor = ['highway', 'Tollway', 'expy', 'fwy', 'hwy', 'Interstate', 
             'Tpke', 'Pkwy', 'Parkway', '-', 'US', 'Route', 
             'FM', 'Byp', 'Trwy', 'Beltway', 'Skyway', 'Skwy', ]
sorted_df.loc[sorted_df['Street'].str.contains('|'.join(searchfor), case=False), 'Highway'] = 'Y'

# Fill NA with zero values for Highway column
sorted_df["Highway"].fillna('N', inplace = True) 

# Create Coordinates column
sorted_df['Coordinates'] = sorted_df['Start_Lat'].map(str) + ', ' + sorted_df['Start_Lng'].map(str)

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

# Drop DUPLICATE rows from Table 2
table2_df.drop_duplicates(inplace=True)

# Set Index
table1_df.set_index('Accident_ID', inplace=True)
table2_df.set_index('Coordinates', inplace=True)

########################################################################################
##############  LOAD  ##################################################################
########################################################################################
sorted_df.columns
len(sorted_df)

##############  THE END  ##################################################################
########################################################################################