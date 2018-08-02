Notes:
Tried to put in construction spending external data - low correlation, little change to model results
plotted same month avg and looked at correlation - extremely difficult to beat as the correlation with same month avg is .94. ADD THIS TO README!

# Construction Equipment Company Demand Forecasting
Mike Irvine - Module 2 Capstone Project

August 3, 2018

Galvanize Data Science Immersive - Denver

## Executive Summary
- Text

<img src="https://github.com/mikeirvine/Capstone-Equipment-Rental/blob/master/imgs/Boom_lift.jpg" width="300" height="300"> <img src="https://github.com/mikeirvine/Capstone-Equipment-Rental/blob/master/imgs/scissor_lift.jpg" width="300" height="300">

## Table of Contents

1. [Context & Key Question](#context_key_question)
2. [Data Source](#data_source)
3. [Exporatory Data Analysis](#eda)
4. [Feature Engineering](#feature_eng)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Future Work](#future_work)
8. [References](#references)

## Context & Key Question: <a name="context_key_question"></a>
### How can a construction equipment rental company in the midwest improve equipment demand forecasting so the right equipment is at the right store location at the right time?

At construction jobsites and industrial plants, companies often rent specialized heavy equipment, rather than own it. The equipment is expensive, specialized, and costly to maintain. Oftentimes, specialized equipment is only needed for a few days or weeks for a particular phase of a construction project, or temporary maintenance shutdown at a plant. So it makes more economic sense to rent the equipment.

What is heavy construction equipment and why is demand forecasting important? There are many categories, but two broad categories are types of equipment that move dirt (backhoes, excavators, etc.), and everything else. The everything else category includes boom lifts, scissor lifts, forklifts, and cranes. This equipment is expensive, so accurate demand planning is critical. Companies that rent the equipment want to make sure they have the right types of equipment available for customers to rent when its needed. If they don't have the piece of equipment available for a customer, they'll either lose out on the deal to a competitor, or be forced to re-rent the equipment from another company, which is costly. Also, companies can't quickly increase or decrease their supply of equipment to meet demand for a couple reasons. 1) If the equipment needs to be transferred between branches, it takes time to move it. 2) If new equipment needs to be purchased, there may be a delay from the dealer or manufacturer as its not as simple as ordering from Amazon.

For my module 2 capstone project, I'm developing a demand forecasting model (predict how many of each equipment type is needed by month) for a heavy construction equipment rental company with 8 stores in the midwest.

## Data Source: <a name="data_source"></a>
### The equipment rental company provided five years of invoice data (one invoice per piece of equipment rented), along with supplemental datasets to enrich the invoice data.

The data provided includes:
- Invoices (one invoice per piece of equipment rented) which includes the rental date, type of equipment, type of rental, and revenue. Invoices were provided for the previous five years.
- Daily equipment data (where each piece of equipment is each day)
- Product type data (more details on the type of equipment including category and class)
- Location data (details on the location of each store)

## Exploratory Data Analysis: <a name="eda"></a>
### After EDA, we agreed to focus on company-wide equipment demand for a subset of the large equipment category.

#### Equipment Type Analysis:
- 20 different categories of equipment - 
- For product category 20, there are ~60K invoices, which can be aggregated by month by product type
- 61 product types for category 20 - further analysis revealed that 17 product types make up 80% of the revenue, and 31 product types make up ~95% of the revenue
<img src="https://github.com/mikeirvine/Capstone-Equipment-Rental/blob/master/imgs/cat_20_rev.png">

#### Revenue Analysis: 
- Revenue and the number of units is consistent over the past five years, but there is significant seasonality
- Since revenue is consistent, will not need to consider an annual growth rate in the model
- Seasonality will need to be considered in the model (i.e., the same month of the prior year will be a good predictor for the same month of the current year)
- Rentals are attributed to 8 different office branches, plus a few sub-branches
<img src="https://github.com/mikeirvine/Capstone-Equipment-Rental/blob/master/imgs/rev_units_time.png">

#### Rental Type Analysis:
- Invoices are categorized as monthly, weekly or daily rentals (i.e., rental type)
- ~60% of rentals are monthly, 20% are weekly, and 20% are daily
- Will need to consider rental type when predicting units rented

#### Approach - *focus on company-wide equipment demand for a subset of the large equipment category*:
- Agreed with the company to focus on category 20, which is all of the large equipment types (scissor lifts, boom lifts, etc.) which make up most the revenue
- Since 30 product types (of 60 total) in category 20 make up ~95% of the revenue, will reduce scope of model to the 30 top product types
- Aggregating invoices by month and a subset of product category 20 leaves only ~3600 records across the company. Data is too limited to also aggregate by branch office, so model will attempt to predict equipment demand across the company.
- Since this is a time series model, the train dataset period (Nov 2014 - April 2017) will be used to predict the test dataset period (May 2017 - April 2018)

Only selecting a subset of product category 20 reduced the dataset to ~3600 records after aggregated by product type and month. Key descriptive statistics for the target variable, 'units_rented' below:

|Stat |    Value 
|-------|----------------|
|count  |  3668 |
|mean |        15.6 |
|std    |      24.6 |
|min     |    1  |
|25%      |     4  |
|50%       |    8  |
|75%      |    17  |
|max      |  300  |


## Feature Engineering: <a name="feature_eng"></a>
### 58 features were created after engineering new features and creating dummy variables for the categorical features.
The invoice dataset had enough information to build the model, but a few new features needed to be engineered to improve the model. Key engineered features include:
- Units rented: each invoice represented a single unit that was rented - created this feature as the target variable
- Prior month features (revenue, units rented, avg price / day): since this is a time series model, prior month features were created to predict the current month
- Rental type: categorized each invoice as daily, weekly, or monthly
- Same month, prior year features (avg units rented, avg days rented): given the seasonality of the business, created new features for the avg units rented and avg days rented for each month (e.g., avg units rented and avg days rented for February). Note: to avoid leakage, I only created these features using the train dataset time period (Nov 2014 - April 2017).
- Categorical features: created categorical features for each month and the 30 product types

The final dataset for modeling had 58 features, mostly due to the categorical features.

## Modeling: <a name="modeling"></a>
### Different types of regression models (linear, lasso, random forest, gradient boosting, and an MLP neural network) were used to try to find the best predictor units rented.

Key Highlights of Modeling Approach:
- Total # of observations for modeling = ~3600 records (after filtering and aggregation)
- Train / test / split approach: split the data based on time. Training dataset was Nov 2014 - April 2017 (~2600 records) and testing dataset was May 2017 - April 2018 (~1000 records). Note: given the small amount of observations and large amount of features, the data is sparse which introduced challenges with dimensionality
- Error metric: root mean squared error
- Types of models:
    - Linear -

From the 500K records, I took a random sample of 100K to run through each regression model. I setup a kfold cross validation with 5 folds, including a standardization of the feature matrix, to ensure an accurate reading of my error metric. For each regression modeling test, I calculated the Root Mean Squared Error. I modeled the data using standard Linear, Ridge and Lasso regression techniques.

After running the results, I identifed the most important features using a recursive feature selection method in sklearn. The method ranks the features in terms of importance (based on the coefficients). It then attemptes to prune the number of features by focusing on the most importants. The most important features groups include:

- Item Class
- Item Number
- Item Perishable
- Month of Year
- Store Cluster
- Store Number
- Store Type
- Week of Year
- Day of Week

According to my models, holidays, oil prices, and promotion status were relatively unimportant compared to the features above.


## Results: <a name="results"></a>
### The standard linear regression model produced the best results on the test dataset, with an R-Squared of .54 and an average Root Mean Squared Error of 8.18. Improvements are needed to better predict sales, but the model is predictive and further refinements will yield better results.

SPARSE DATA!

The results indicate the a linear regression model to predict unit sales is feasible, and with further refinements to the model, accuracy of the model could be improved. The Root Mean Squared Error (RMSE) results for each model are below.

Linear RMSE test results: 8.19

Linear R squared results: 0.54

Lasso RMSE test results: 8.20 with alpha of 0.05

Ridge RMSE test results: 8.20 with alpha of 0.1

I used the regularization models, Lasso and Ridge, to attempt to reduce the magnitude of the coefficients in the case of overfitting. However, I found that the larger the alphas, the worse the scores for the models. This shows that the standard linear regression model is not overfitting.

One important note regarding the outliers. Without removing the outliers, the RMSE was 26+ and R Squared is only ~.25. This indicates that the outliers and leverage points had a major impact on the model's predictions and it was valid to remove them.

Below is a plot of the residuals vs the predicted unit sales for each regression technique. There is an issue with the residuals given the diagonal line, which is likely caused by a set of zero values in the actuals where the model predicts an increasingly larger value. More research is required to uncover the issue. Besides for the diagonal line, the residuals are fairly normally distributed, which is a good sign.

![alt text](https://github.com/mikeirvine/Capstone-Ecuador-Grocery/blob/master/images/residualvpredicted.png)

The next plot shows the actual vs predicted values. This plot shows that there is predicted value in the model as the data points trend to the upper right.

![alt text](https://github.com/mikeirvine/Capstone-Ecuador-Grocery/blob/master/images/predictedvactuals.png)


## Future Work: <a name="future_work"></a>
### There is opportunity to improve the model on the MEATS item family prior to applying it to all item families and time periods.
Additional work is need on the model to yield better results. There are a few questions I would investigate first to improve the model, including:
- What is the issue with the diagonal trend line in the residuals? Are there some zero values I need to adjust?
- Are there any other outliers or leverage points impacting the effectiveness of the model?
- What other features could be engineered to add to the model?
- What other linear regression modeling techniques could be tested?
- Are some features collinear and need to be removed?


## References: <a name="references"></a>
Below is a link to the Kaggle competition with the datasets.
[Kaggle competition link](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

