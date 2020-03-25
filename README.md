# RPD_Accident_ML

## Selected topic
US Road Accidents and their impact on Traffic (Delays)

## Reason topic selected / Inspiration
US-Accidents can be used for numerous applications such as real-time accident prediction, studying accident hotspot locations, casualty analysis and extracting cause and effect rules to predict accidents, and studying the impact of precipitation or other environmental stimuli on accident occurrence. 

This Accident dataset is additionally attractive to us because, we are able to apply deep learning models to the data set due to its large number of features. 

## Description of data source
This data is located on Kaggle and is a countrywide traffic accident dataset, which covers 49 states of the United States. This data has been collected in real-time, using multiple Traffic APIs. Currently, it contains data that is collected from February 2016 to December 2019 for the Contiguous United States.

These APIs broadcast traffic data captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road-networks. 

While not all inputs are complete for each line item, there are nearly 50 features presented. Some of these are unsuitable for our project while others lack enough information to be worth including. Prior to cleaning and transforming data, there are approximately 3M entries.  

#### Acknowledgements for the dataset as required by the source [Please cite the following papers if you use this dataset]: 
- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.
- Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019. 

## Questions hope to answer with data
There are 2 parts to this project:

(i) Predict US Road Accident Severity using a Machine Learning model, where severity is defined as impact on traffic (delays)

(ii) Create a US heatmap visualization based on the severity of the accidents

## Description of the data exploration phase of the project - RS


## Description of the analysis phase of the project - PPD


## Technologies, languages, tools, and algorithms used throughout the project - RS


## Result of analysis - PPD (bulleted key takeaways from each)
From a technical level we found that the Decision Tree Classifier (DTC) was the best predictor of accuracy (81%) with our Neural Network (NN) model a close second (75%). With more experience using neural networks and the vast number of variables, it’s reasonable to expect that NN performance would exceed DTC. 

With two of the four severity classification scores (Level 2 & 3) accounting for >80% of the sample data, it stands to reason that most of the models we tested performed reasonably well predicting those severity levels. One issue at hand is that Level 4 accidents have the most severe traffic response, but only account for 2% of accidents. With such a small accident count in a model that is performing very well with much more benign accidents, it’s not unexpected, though disappointing, that we couldn’t get better results in the accidents that likely have the most catastrophic results.  

When predicting traffic severity, it’s reasonable to want to know what factors have the greatest impact on the duration of the post-accident traffic. With the data and the decision tree classifier that we used, Location (longitude and latitude) has the greatest influence (19.2% and 17.4% respectively) in predicting accident severity. This is not altogether shocking given that traffic severity is highest in dense population centers and accident frequency will be higher in those locations. Thus, in places like Atlanta, Chicago, New York, Los Angeles have considerably more traffic than remote highways in the middle of the US, so when an accident happens in those locations, the resulting traffic impact can be sizable compared to a similar collision on a remote interstate. 

Accident duration (time difference between initiation of accident to completion of accident) is the feature that has the next biggest influence (10.5%) on severity. This should be obvious given that if the duration of an accident is long, it’s likely that the resulting traffic will also be long. 

The Highway (or not) feature was the next greatest influence (8.9%) on predicting severity. Similar to the location features, for an accident to be rated as “most severe” it will have a very long traffic delay. In most cases, that will be in places where there is already an abundance of traffic (i.e. highways). 

Time in Seconds (i.e. time of day) was the 5th most influential factor. Tying back to the post-accident traffic severity would be most common at times when traffic is the highest, it makes sense that frequency would spike around heavy commute hours (8-10 AM, 4-6 PM). One issue with this is that this is not a linear relationship; i.e. traffic frequency and accident severity will peak at 9 AM, but it’s unlikely that it would continue to rise at 10 AM, etc. Upon further review, this is a variable that we may have wanted to encode another way. 

One surprise from the analysis is that the weather conditions have very little bearing on accident severity. The reasons for this could be several, some of them data while others are likely human behavior. 1) There are relatively few accidents recorded when conditions are the most dangerous so the models are likely not “seeing” those events as influential to traffic severity. 2) People tend to not drive, or drive more cautiously, when conditions  are poor. As a result, when accidents do occur, fewer people are on the road leading to shorter delays. 


## Recommendation for future analysis - PPD/RS
note: 5 PM weekday is very different from 5 PM weekend but we don't separate that out


## Anything the team would have done differently - PPD/RS
We used Accuracy for the arbiter for machine learning model success with Recall being the second metric we focused on. Regardless of model selection, the Level 4 (most severe) accidents were the most difficult to predict with ~50% recall & precision the maximum achievement. 

With the goal of improving Level 4 metrics, we would focus energies on determining which features have little influence in accurately predicting accident severity and drop them from the pre-split analysis dataframe. 

Furthermore, we barely scratched the surface of the world of neural network models. Given the number of variables to modify (nodes, layers, activation, etc.) it’s reasonable that we could achieve an accuracy that parallels the results from the Decision Tree Classifier with better Recall for the most extreme accidents.  


## Notes
 add link the google preso
 add link to tableau file
 add ML txt file (link and commit)


