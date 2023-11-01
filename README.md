# Bulldozer Price Prediction

This project uses Python-based machine learning and data science libraries to build a machine learning model that can predict the price of bulldozers in the future using a time-series data set.

## Problem

How well can we predict the future sale price of a bulldozer given its characteristics and previous examples of how much similar bulldozers have been sold for?

## Data

The data is downloaded from the Kaggle Bluebook for Bulldozers competition: https://www.kaggle.com/competitions/bluebook-for-bulldozers/data. A data dictionary detailing all of the features of the dataset can be found at the bottom of that page.

There are 3 main data sets:

* Train.csv is the training set, which contains data through the end of 2011.
* Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012.
* Test.csv is the test set, which contains data from May 1, 2012 - November 2012.

## Evaluation

The metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices. We will build a machine learning model and try to minimize the RMSLE.

## Exploratory Data Analysis

We started with some brief EDA to get familiar with the data.

First, we plotted a histogram of the sale prices, seen in **Figure 1** below:<br>
![image](https://github.com/nwferreri/bulldozer-price-prediction/assets/112211174/443c9eb2-6ab3-4709-8359-737b919d64a5)<br>
**Figure 1**: The majority of bulldozer sales fall on the lower end of the price range.

Because we are dealing with time-series data, we next looked at sale price against date, seen in **Figure 2** below:<br>
![image](https://github.com/nwferreri/bulldozer-price-prediction/assets/112211174/d1e911a1-cdb4-4f9c-bd1e-32c71235912d)<br>
**Figure 2**: There seems to be a pretty good spread of data across time, except for gaps around 2005 and 2008.

## Data Preparation

Before running a regression model, the data needed to be prepared.

Many of the features in the data were partially or completely non-numerical, so those were converted into pandas categories using the `pandas.api.types` tools.

Next, we addressed missing values. For numerical features, missing values were filled with the median. Binary columns were also added to track which rows were missing. Categorical variables were turned into numbers and binary columns were again added to track which rows were missing which values.

Finally, data was split into training and test sets.

## Modeling

Because of the large amount of data (over 400,000 entries), we used a subset to train the model and adjust the hyperparameters.

The initial model had a validation set RMSLE of 0.2936.

Hyperparameter tuning was performed using `RandomizedSearchCV` with 5-fold cross-validation. The best parameters were extracted and used to train a model on the full dataset. That model had a validation set RMSLE of 0.2466, showing an improvement over the initial model.

## Predictions
The final model was used to make predictions on the test dataset. First, the data had to be processed to be in the same format as the training and validation sets. Test data predictions were exported to a .csv file.

## Evaluation
To generate some real-world conclusions from the model, we explored the feature importances. They can be seen in **Figure 3** below:<br>
![image](https://github.com/nwferreri/bulldozer-price-prediction/assets/112211174/422f1924-b350-415e-8282-6d7318697b20)<br>
**Figure 3**: Based on the model, the size of a bulldozer and the year it was made are two of the best indicators for predicting it's price.
