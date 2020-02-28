##########################################################################################
############# This file performs ETL on the US Road Accidents dataset from Kaggle ########
##########################################################################################

# Import dependencies
import pandas as pd

#####################################################################
#########  DATA EXTRACTION  #########################################
#####################################################################

# Extract Kaggle Data
kaggle_metadata = pd.read_csv('./US_Accidents_Dec19.tar.gz', compression='gzip', error_bad_lines=False)

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
#########  TRANSFORM  #########################################
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

# Check length of the remaining dataset after removing duplicate and Null value rows
len(sorted_df)

# check datatypes for corrections
sorted_df.dtypes

########################################################################################
### Split Date and Time Column into Date, Time, Time in Seconds, and Day of week columns
########################################################################################

# new data frame with split value Start date time column
newstart = sorted_df["Start_Time"].str.split(" ",expand = True) 
  
# making separate Start Time column from new data frame 
sorted_df["Start_Time_of_Day"]= newstart[1] 

# new data frame with split value End date time column
newend = sorted_df["End_Time"].str.split(" ",expand = True) 

# making separate Start Time column from new data frame 
sorted_df["End_Time_of_Day"]= newend[1] 

# Convert Time to seconds for Start Time and End Time
sorted_df['Start_seconds'] = pd.to_timedelta(sorted_df['Start_Time_of_Day']).dt.seconds

# Convert Time to seconds for Start Time and End Time
sorted_df['End_seconds'] = pd.to_timedelta(sorted_df['End_Time_of_Day']).dt.seconds

sorted_df['Start_Time'] = pd.to_datetime(sorted_df.Start_Time)
sorted_df['End_Time'] = pd.to_datetime(sorted_df.End_Time)

# Get Day of the week for the accident
sorted_df['Day_of_Week'] = sorted_df['Start_Time'].dt.weekday

########################################################################################
##############  LOAD  ##################################################################
########################################################################################
sorted_df.columns
len(sorted_df)

##############  THE END  ##################################################################
########################################################################################