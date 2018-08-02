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
### 60k invoices from the past five years provide enough information to be the core dataset for the demand forecasting model.

#### Revenue Analysis: 
- Revenue and the number of units is consistent over the past five years, but there is significant seasonality
- Since revenue is consistent, will not need to consider an annual growth rate in the model
- Seasonality will need to be considered in the model (i.e., the same month of the prior year will be a good predictor for the same month of the current year)
<img src="https://github.com/mikeirvine/Capstone-Equipment-Rental/blob/master/imgs/rev_units_time.png">
#### Rental Types:
- 4100 records, which represent 4100 individual items
- Features include: item_nbr, class, and perishable
- All features are categorical
#### Equipment Types:
- 54 records, which represent 54 store locations
- Features include: store_nbr and cluster, both of which are categorical
#### Store Transactions Dataset:
- 100k+ records, which represent the total daily transactions by store by day from Jan 1, 2013 to Aug 15, 2017
- Features include: store_nbr and transactions
#### Holidays Dataset:
- 300+ records, which represent each holiday from Jan 1, 2013 to Aug 15, 2017
- Features include: date, type, locale, locale name and description
- The locale feature identifies if the holiday is local, regional or national
#### Oil Dataset:
- 1200+ records, which represent the daily oil price from Jan 1, 2013 to Aug 15, 2017
- Features include: daily oil price

#### Approach - *focus on a single year, single item family*:
Given the large dataset, I decided to reduce the scope and focus on a single year and item family. I selected the 'MEATS' item family from August 2015 - August 2016 to be my training and test dataset.

The items dataset included 33 item families:

- ['GROCERY I',
 'CLEANING',
 'BREAD/BAKERY',
 'DELI',
 'POULTRY',
 'EGGS',
 'PERSONAL CARE',
 'LINGERIE',
 'BEVERAGES',
 'AUTOMOTIVE',
 'DAIRY',
 'GROCERY II',
 'MEATS',
 'FROZEN FOODS',
 'HOME APPLIANCES',
 'SEAFOOD',
 'PREPARED FOODS',
 'LIQUOR,WINE,BEER',
 'BEAUTY',
 'HARDWARE',
 'LAWN AND GARDEN',
 'PRODUCE',
 'HOME AND KITCHEN II',
 'HOME AND KITCHEN I',
 'MAGAZINES',
 'HOME CARE',
 'PET SUPPLIES',
 'BABY CARE',
 'SCHOOL AND OFFICE SUPPLIES',
 'PLAYERS AND ELECTRONICS',
 'CELEBRATION',
 'LADIESWEAR',
 'BOOKS']

Only selecting the 'MEATS' item family reduced the datasize to ~500k records. Key descriptive statistics for the target variable, 'unit_sales' below:

|Stat |    Value 
|-------|----------------|
|count  |  574203.00 |
|mean |        11.94 |
|std    |      32.96 |
|min     |    -44.26  |
|25%      |     2.52  |
|50%       |    5.30  |
|75%      |    11.61  |
|max      |  5357.83  |

Because of some extreme outliers (12 data points > 1000 unit_sales) that are several standard deviations away from the mean, I decided to remove them from the dataset. Any data point +/- 2 standard deviations from the mean was removed. The table and histogram below shows details for the unit_sales for the MEATS item family after the outliers are removed. The mean is now 9.46 unit sales per day per store, a standard deviation of ~12, and a max value of 77.86.

|Stat |    Value 
|-------|----------------|
|count |   563817.00 |
|mean   |       9.46 |
|std    |      11.97 |
|min    |       0.00 |
|25%    |       2.48 |
|50%    |       5.17 |
|75%    |     11.04 |
|max    |      77.86 |

![alt text](https://github.com/mikeirvine/Capstone-Ecuador-Grocery/blob/master/images/hist_unitsales.png)


## Feature Engineering: <a name="feature_eng"></a>
### 249 features were created after merging datasets to enrich the transaction data and creating dummy variables for all the categorical features.
All of the supplemental datasets could be merged with the transaction dataset using date, item_nbr or store_nbr. This allowed me to enrich the transaction data set with item, store, holiday, and oil price features. Prior to merging, I had to engineer several features across the datasets to ensure the features added value to the model. Key engineered features include:
#### Store Transactions Dataset:
- Created a feature for the daily average transaction count by store to serve as a proxy for store size, an important factor in sales
#### Holidays Dataset:
- Created a holiday eve feature, which is the day before a holiday when sales likely spike as families prepare for holiday meals
- Needed to break apart local, regional and national holiday dates into separate dataframes

After creating these features in the supplemental datasets, I merged all of the supplemental datasets into the master transaction dataset to enrich it with new features.

### Feature Engineering in Merged Dataset
- Created day of week, month of year, and week of year features to replace the date feature
- Identified all categorical variables to create dummies, including store_nbr, item_nbr, item_class, date features, store type, store cluster

The final dataset for modeling had 249 features, mostly due to the dummies. For example, there are 83 dummy features for the MEATS category alone due to the number of different meat related items for sale.

## Modeling: <a name="modeling"></a>
### Different variations of linear regression with cross validation were used to try to find the best fit, including standard linear, lasso and ridge techniques

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

