# Data Preprocessing & Machine Learning Background

## Description of data preprocessing
* Dataset was read in from SQL with a join between the accident and accident_location tables
    * Spark dataframe created for dataset
    * Spark dataframe converted to a Pandas dataframe to allow for additional preprocessing
        * Given the volume of data information (~2.5M rows), dataframe converted by chunks
    * Several feature columns dropped due to redundancy or no value in future modeling
    * By using “nunique”, we determined which features need encoding application and which could be binary conversions.
        * Many accident features were described as “True”/”False”, “Y”/”N” or “Night”/”Day” so those were converted to 1/0 respectively.
        * Features such as “wind direction”, “timezone”, “side” and “day of week”, etc. had 7 or more options, so those features were encoded
    * Existing dataset provided accident start/end time so new columns were created to determine day of week as well as the duration of the accident.
    * Over 100 varied weather conditions were provided with no straightforward manner to group via simple script.
        * For example, there are 26 different conditions featuring “snow” from “light snow grains” to “heavy thunderstorms and snow”. Ergo, creating a filtered classification based on the word “snow” would likely cause more issues that solve them.
        * To reconcile this, the team performed manual processing based on personal judgement. With a list of ~125 categories this was a manageable task. Had the list been longer, we likely would have looked to automate this step, possibly with performing an additional machine learning model to determine correlation between traffic severity to weather conditions.
        * Unique conditions and frequencies were exported to a separate .csv file for distinct grouping into ten categories.
        * Groupings .csv file was re-read back into the script and joined to the existing dataframe using the existing weather condition as the common feature.
    * String features were cast as float while existing float & integer features were left as is.
    * Superfluous and redundant columns dropped

## Description of feature engineering and the feature selection, including the team's decision-making process
* Severity is the feature we deemed as the target as it is used to describe the impact the accident had on traffic. All other traffic features are inputs to determine what was the outcome.
    * The team exercised judgement to decide which features were unlikely to aid in the modeling of the data.
    * While the precise date did not seem to add value in predicting the extent of traffic delay, the day of week and time seemed to provide considerable value.
    * Furthermore, the starting date provided context but the ending date does not except to determine the duration of the actual accident.
    * ID, Source, Traffic Message Channel (TMC) and Description are not suitable descriptors for the data. ID is simply a unique code for the accident, Source is where the information was gleaned and TMC provides a detailed description of the accident.
    * Much of the location information is provided with longitude and latitude information so specific address information is not included.
    * Much of the specific weather information is not included in the dataset, so while it may have been valuable, the volume of missing information rendered it useless.
    * Weather conditions such as wind direction, precipitation and conditions were abundant and valuable so it was included.
    * Location information such as junctions, specific signage, speed bumps, etc. were provided and available in many cases, so those rows were included.
    * Civil Twilight was the only daytime/nighttime feature included as anything extra seemed redundant and unnecessary.

## Description of how data was split into training and testing sets
* The data was split using sklearn with a 70/30 split between training and test set.

## Explanation of model choice, including limitations and benefits
* We started with a multinomial logistic regression model to analyze our dataset. Rather than a logistic expression which would use features (x) to determine if a particular outcome (y) occurred or not, a multinomial considers multiple values for the variable y. That is necessary for this application since post-accident traffic severity is ranked 1-4.
    * Limitations - More detail will be provided in the explanation of the confusion matrix but one limitation of the model was that over ⅔ of the data was one of the four traffic classifications and over ⅘ was two of the classifications. As a result, there were relatively few instances of level 1 and level 4 traffic severity and the model struggled to accurately predict those classifications with so little information to draw on to train the model.
    * Benefits - With an accuracy score of 67%, the logistic regression is effective in accurately predicting level 2 traffic severity. This stands to reason since it is the largest category of accident classifications. Regrettably, it is poor with both precision and recall with the other groupings.

## Explanation of changes in model choice (if changes occurred between the Segment 2 and Segment 3 deliverables)
* Oversampling - With so few accidents classified as Level 1 in the test set (as well as the trained set), oversampling the data set was performed to determine if a different model may improve predictions of less frequent, but severe traffic problems (Level 4). With an accuracy score of 41%, this did not improve our position but did provide some interesting results.
    * The oversampling model accomplished the goal of improving recall of the Level 1, 3 and 4 accidents though precision was still low. Many Level 2 and Level 3 accidents are mis-diagnosed as Level 1 accidents. The problem with these {false positives?} is that if the data suggests that the accident is relatively minor (i.e. Level 1) and resources are not allocated appropriately, the actual traffic delay will likely increase longer than if it had been diagnosed correctly as Level 2 & 3 in the first place.
    * While the model is overly “casual” with grouping Level 2 & 3 accidents as Level 1, it is also very conservative by improperly predicting that Level 2 & 3 accidents as Level 4. Once again the allocation of resources is misaligned in that more equipment and manpower will be provided for an accident that may not even warrant an ambulance.
    * The consequence of the oversampling is that the model lost both precision and recall for the more frequent Level 2 and 3 accidents such that the overall accuracy dropped to 39.7%. Another challenge was that the model took nearly an  hour to run compared to the Logistic Regression which was less than ten minutes.
* Decision Tree Classifier (DTC) -  In an effort to improve both precision and recall so that resources are allocated in a suitable manner without over or under committing, we employed a decision tree classifier. Given that there are dozens of features to manage, the decision tree seemed like a logical next step. With an accuracy score of 81% it appears the decision tree is doing a better job with a larger portion of the accidents.
    * The DTC model performs well with the Level 2 & 3 accidents and returns the highest metrics for the Level 4 accidents of any of the previous models.
    * Very poor predictions (~0% )for the least severe accidents are unfortunate but an over-allocation of resources for a minor fender-bender is preferable than under-providing following a 50-car pile-up in snowy, icy conditions.
    * With just 2% of the accident population categorized as a Level 4 accident, the model struggles to accurately predict better than a coin-flip (~50% for both precision and recall).
* Deep Neural Network - With the goal of improving on DTC performance several DNNs were attempted. Given nascent experience and the variety of options to “tinker” with, this felt a bit more uncertain.
    * Model parameter choice
        * Several activation functions were attempted but the softmax function provided the best performance.
        * 20 epochs was settled on mostly because accuracy improvements plateaued after 10 epochs.
        * Layer choice and node count was trial and error.
        * 4 Units selected given the four severity classifications.
    * The DNN achieved similar performance scores to the DTC but with slightly lower accuracy (75% vs 81%) with similar precision and recall scores.
    * Ultimately, with more experience, data feature improvements and more knowledge around model parameter choice we expect that a DNN model will exceed DTC performance. In the end, a model needs to do a better job predicting the most severe accidents to be viable.

## Description of how model was trained (or retrained, if they are using anexisting model)
* After the data was scaled, data was trained using sklearn.preprocessing. In an effort to remain consistent, data was not retrained through the different model development.
* With the exception of the Oversampling instance, all models used the same training and test sets.
* In the Oversampling case, the original training set was resampled to create an equal-sized case for all four severity levels.
* For the Neural Network case, the target data was modified from 1:4 to 0:3 as the model expects a zero as the lowest classification. Furthermore, target data was encoded due to the multivariable classification.

## Description and explanation of model’s confusion matrix, including final accuracy score
* The DTC model does a satisfactory job predicting Level 2 & 3 traffic accidents (87% and 72% precision/recall). It is worth noting that there is a disproportionately large volume of accidents that fall into that category so there is ample opportunity to get the “bet” right. The model struggles with predicting Level 4 accidents (~50% precision/recall). Strangely, while the model accurately predicts ~50% of the accidents correctly as Level 4, it incorrectly assigns more accidents as Level 2 than it does Level 3. The model performs poorly  predicting Level 1 accidents at 0% for both precision/accuracy. One issue is that of over 600’ accidents in the test set, only 200 were actually Level 1. It is worth noting that ~60% of the accidents are Level 2 and ~30% are Level 3.

## Additionally, the model obviously addresses the question or problem the team is solving.
* The team is pleased with initial performance of the model but understands that it is not production ready. The poor performance in the most severe cases is concerning and at ~50%, it’s still barely better than a coin-flip. Future efforts will focus on improving Level 4 predictions by considering feature elimination, outlier elimination and potentially additional data features.

## Statistical analysis was not included as part of the current analysis.
* In future analyses, we anticipate looking into ways to optimize the performance and speed of modeling. One way we would do that is removing outlier data. Another step would be looking at p-values for data to determine significance.
