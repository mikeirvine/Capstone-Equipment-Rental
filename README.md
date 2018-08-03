# Construction Equipment Company Demand Forecasting
Mike Irvine - Module 2 Capstone Project

August 3, 2018

Galvanize Data Science Immersive - Denver

## Executive Summary
- How can a construction equipment rental company in the midwest improve equipment demand forecasting so the right equipment is at the right store location at the right time?
- The equipment rental company provided five years of invoice data (one invoice per piece of equipment rented), along with supplemental datasets to enrich the invoice data.
- After EDA, we agreed to focus on company-wide equipment demand for a subset of the large equipment category.
- 58 features were created after engineering new features and creating dummy variables for the categorical features.
- Different types of regression models (linear, lasso, random forest, gradient boosting, and an MLP neural network) were used to try to find the best predictor units rented.
- The random forest model produced the best results on the test dataset, with an R-Squared of .83 and a root mean squared error of 10.46 units rented. These results slightly beat the same month average rmse of 11.19, which indicates there is some added predictive value of using additional features to predict demand, rather than just a simple same month average.
- There is opportunity to improve the model, but more investigation and analysis is needed.

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
- 20 different categories of equipment (category 20 is the large equipment category, making up most the revenue)
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
- External data: added national construction spending data (as the industry is highly correlated to the economy), but there was little correlation with units rented so the feature was removed

The final dataset for modeling had 58 features, mostly due to the categorical features.

## Modeling: <a name="modeling"></a>
### Different types of regression models (linear, lasso, random forest, gradient boosting, and an MLP neural network) were used to try to find the best predictor units rented.

Key Highlights of Modeling Approach:
- Total # of observations for modeling = ~3600 records (after filtering and aggregation)
- Train / test / split approach: split the data based on time. Training dataset was Nov 2014 - April 2017 (~2600 records) and testing dataset was May 2017 - April 2018 (~1000 records). Note: given the small amount of observations and large amount of features, the data is sparse which introduced challenges with dimensionality
- Error metric: root mean squared error
- Standardization of data: used sklearn's StandardScaler capability to standardize the dataset
- Types of models: linear, lasso, random forest, gradient boosting, neural network MLP

## Results: <a name="results"></a>
### The random forest model produced the best results on the test dataset, with an R-Squared of .83 and a root mean squared error of 10.46 units rented. These results slightly beat the same month average rmse of 11.19, which indicates there is some added predictive value of using additional features to predict demand, rather than just a simple same month average.

Summary results by model on test dataset:

|Model |    RMSE 
|-------|-------------|
|Linear  |  11.17 |
|Lasso |      11.18 |
|Random Forest    |      10.46 |
|Gradient Boosting     |    12.68  |
|MLP      |     12.19  |
|Same Month Avg      |     11.19  |

After talking to the company, I learned that the same month of the prior year is likely the best predictor of current month demand. A correlation matrix (see below) proved this. The same month feature have the highest correlation to the target variable. The lasso model confirms this as well as it zeroed out almost every feature except for the same month average units rented.

An interesting finding though is that daily / weekly rentals, and prior month avg price are negatively correlated to units rented, meaning they may drive demand lower.

|Feature       |          Correlation    |
|--------------|-------------------------|
|units_rented    |                 1|
|same_month_avg_units_rented   |   0.94|
|same_month_avg_days_rented   |    0.89|
|prior_month_units_rented    |     0.82|
|prior_month_total_days_rented  |  0.78|
|rental_type_daily           |    -0.16|
|rental_type_weekly           |   -0.17|
|prior_month_avg_price_per_day |  -0.18|

The random forest feature importances supported the same findings as the correlation matrix. Below is the feature importance rank.

1. same_month_avg_units_rented
2. same_month_avg_days_rented
3. prior_month_total_days_rented
4. prior_month_units_rented
5. product_type_20-220

Given that the same month avg units rented is such a strong predictor of current month demand, the challenge is for my model to beat this single predictor. The graphs below compare the actual units to the predicted units for random forest, linear and the same month average. Random forest has a slightly lower RMSE and visually looks to beat the same month average.

<img src="https://github.com/mikeirvine/Capstone-Equipment-Rental/blob/master/imgs/pred_v_acts_time.png">

Even though the random forest rmse beats the same month average, the error is still high given that the mean units rented is ~15. I inspected the residuals and it appears the model is having difficult predicting product types with a very large number of units rented. Any miss on these product types magnifies the residuals due to the larger numbers. See below for a plot of the random forest residuals.

<img src="https://github.com/mikeirvine/Capstone-Equipment-Rental/blob/master/imgs/random_forest_residuals.png">

## Future Work: <a name="future_work"></a>
### There is opportunity to improve the model, but more investigation and analysis is needed.

Additional work is need on the model to yield better results. There are a few questions I would investigate first to improve the model, including:
- Why is the model having difficulty predicting larger amounts of units rented?
- Is mixing daily, weekly, monthly rental types into one model making the model worse? Can I improve predictions if I create three separate models for each rental type (monthly, weekly, daily)?
- Are there outliers or leverage points impacting the effectiveness of the model?
- What other features could be engineered to add to the model?
- What other linear regression modeling techniques could be tested?
- Some of the features are collinear, so would removing some features improve the model's predictions?
