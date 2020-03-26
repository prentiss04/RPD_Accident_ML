# RPD_Accident_ML

## Selected topic
US Road Accidents and their impact on Traffic (Delays)

## Reason topic selected / Inspiration
US-Accidents can be used for numerous applications such as real-time accident prediction, studying accident hotspot locations, casualty analysis and extracting cause and effect rules to predict accidents, and studying the impact of precipitation or other environmental stimuli on accident occurrence. 

## Description of data source
### - Source: Kaggle
### - Datasize: ~3M entries
### - Time period: February 2016 to December 2019
### - Scope: 49 US states 
####  (No Hawaii and Alaska)

While not all inputs are complete for each line item, there are nearly 50 features presented. Some of these are unsuitable for our project while others lack enough information to be worth including. Prior to cleaning and transforming data, there are approximately 3M entries.  

More details about the dataset can be found at Kaggle: https://www.kaggle.com/sobhanmoosavi/us-accidents

#### Acknowledgements for the dataset as required by the source [Please cite the following papers if you use this dataset]: 
- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.
- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019. 

## Questions hope to answer with data

1.) Can we accurately predict accident severity based on certain features?

2.) What are the key features that cause accidents with severe traffic delays?

3.) Can an Early Warning System be developed to avoid traffic hotspots based on certain conditions such as weather, time of day

## Description of the data exploration phase of the project
The data was explored for relevant features, duplicates, excessive null values etc. 

![Data Exploration Sample image](https://github.com/prentiss04/RPD_Accident_ML/blob/Master/Data_Exploration.png)

And following transformation was performed, on the select data columns extracted, for use in the project:
- Handling of Null values: such as replaced nulls with 0 for precipitation, dropped columns with significant null values, dropped rows with null values
- Drop duplicate rows: since accidents were collected from multiple sources there were duplicates
- Transform/Create columns
     - Extract 5 digit zipcode from postal zipcodes
     - Create additional columns such as Highway
- Correct column datatypes such as converting start date and time column to datetime type
- Rename columns to match the database table column names

## Description of the analysis phase of the project
Using the tools listed below, we used open source automobile accident information to create several machine learning models to predict post-accident traffic severity. We leveraged Logistic Regression, Oversampling, Decision Tree Classifier and Neural Networks to build models of varying accuracy. 
Users should have current versions of Python, TensorFlow, PySpark loaded to successfully run files ETL.py & accidents.py.

Tools:
- Postgres for data storage
- PySpark for data reading/reading
- Sklearn, TensorFlow & Keras for machine learning
- Github, Google Colab & Docs for collaboration
- Python, MS Excel, pandas for preprocessing 

## Technologies, languages, tools, and algorithms used throughout the project
- Database Storage: AWS Postgres RDS
- Program to Extract, Transform, Load and Query data: Pyspark and Psycopg2 for Postgres connectivity
- Python Machine Learning algorithms for classification: Oversampling, Decision Tree Classifier, Tested and rejected Neural Network and Logistic Regression
- Tableau for Analysis and Visualization


## Result of analysis
**Decision Tree Classifier outperformed other models tested**<br />
From a technical level we found that the Decision Tree Classifier (DTC) was the best predictor of accuracy (81%) with our Neural Network (NN) model a close second (75%). With more experience using neural networks and the vast number of variables, it’s reasonable to expect that NN performance would exceed DTC. 

**Low count classification levels are difficult to predict**<br />
With two of the four severity classification scores (Level 2 & 3) accounting for >80% of the sample data, it stands to reason that most of the models we tested performed reasonably well predicting those severity levels. One issue at hand is that Level 4 accidents have the most severe traffic response, but only account for 2% of accidents. With such a small accident count in a model that is performing very well with much more benign accidents, it’s not unexpected, though disappointing, that we couldn’t get better results in the accidents that likely have the most catastrophic results.  

**Accident location is the strongest feature to predicting post-accident traffic severity**<br />
When predicting traffic severity, it’s reasonable to want to know what factors have the greatest impact on the duration of the post-accident traffic. With the data and the decision tree classifier that we used, Location (longitude and latitude) has the greatest influence (19.2% and 17.4% respectively) in predicting accident severity. This is not altogether shocking given that traffic severity is highest in dense population centers and accident frequency will be higher in those locations. Thus, in places like Atlanta, Chicago, New York, Los Angeles have considerably more traffic than remote highways in the middle of the US, so when an accident happens in those locations, the resulting traffic impact can be sizable compared to a similar collision on a remote interstate. 

**Severity follows in importance to predicting severity**<br />
Accident duration (time difference between initiation of accident to completion of accident) is the feature that has the next biggest influence (10.5%) on severity. This should be obvious given that if the duration of an accident is long, it’s likely that the resulting traffic will also be long. 

**Road type classification features highly in predicting traffic severity**<br />
The Highway (or not) feature was the next greatest influence (8.9%) on predicting severity. Similar to the location features, for an accident to be rated as “most severe” it will have a very long traffic delay. In most cases, that will be in places where there is already an abundance of traffic (i.e. highways). 

**Knowing when an accident takes place is a good predictor how long the backup is present**<br />
Time in Seconds (i.e. time of day) was the 5th most influential factor. Tying back to the post-accident traffic severity would be most common at times when traffic is the highest, it makes sense that frequency would spike around heavy commute hours (8-10 AM, 4-6 PM). One issue with this is that this is not a linear relationship; i.e. traffic frequency and accident severity will peak at 9 AM, but it’s unlikely that it would continue to rise at 10 AM, etc. Upon further review, this is a variable that we may have wanted to encode another way. 

**Counter to expectations, the weather is not a not a great predictor of post-accident traffic**<br />
One surprise from the analysis is that the weather conditions have very little bearing on accident severity. The reasons for this could be several, some of them data while others are likely human behavior. 1) There are relatively few accidents recorded when conditions are the most dangerous so the models are likely not “seeing” those events as influential to traffic severity. 2) People tend to not drive, or drive more cautiously, when conditions  are poor. As a result, when accidents do occur, fewer people are on the road leading to shorter delays. 


## Recommendation for future analysis
For future analysis, including, where possible, other features of accidents such as driver age and experience, number of vehicles involved, speed of initiating driver and possible impairments would be our next step. 

Furthermore, as the dataset is very large, we'd want to see if there are other ways to streamline the analysis. This could be done by looking at features that offer little to no benefit to the model or consider possible outliers that may be creating more noise than advantate. 

As it stands, we are looking at the traffic severity as the target. It would be interesting to see how the data correlates to other targets (not currently provided) such as predicting financial cost of accidents or accidents/fatalities. 

## Anything the team would have done differently
While time of day and day of week were both features considered during the analysis, we overlooked the creation of a weekday/weekend. We expect that this would be similar to the the Civil_twilight feature where most high severity accidents would occur during the weekdays when traffic is the highest during commute hours. This may have helped increase the models in the Level 4 accidents. 

We used Accuracy for the arbiter for machine learning model success with Recall being the second metric we focused on. Regardless of model selection, the Level 4 (most severe) accidents were the most difficult to predict with ~50% recall & precision the maximum achievement. 

With the goal of improving Level 4 metrics, we would focus energies on determining which features have little influence in accurately predicting accident severity and drop them from the pre-split analysis dataframe. 

Furthermore, we barely scratched the surface of the world of neural network models. Given the number of variables to modify (nodes, layers, activation, etc.) it’s reasonable that we could achieve an accuracy that parallels the results from the Decision Tree Classifier with better Recall for the most extreme accidents.  


## Notes
 Presentation: https://docs.google.com/presentation/d/1c3YTJ279FInRyDMTLCzxcWQR16KaX0t2oJ0nIwjrDpo/edit?usp=sharing,<br />
 Visualization: https://public.tableau.com/profile/ruchi7973#!/vizhome/FinalProject-USRoadAccidents/Storyboard<br />
 Machine Learning Observations: https://docs.google.com/document/d/1oLN1u9kHJTzFKctZW1csRmOUX3zS_cXikEgI1IjUg3c/edit
