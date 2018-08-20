import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pandas.tseries.offsets import MonthBegin
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def load_data(filepath, separator=','):
    '''Load invoice data into a pandas dataframe'''
    df = pd.read_csv(filepath, sep=separator)
    return df

def filter_data(df, equipment_list, drop_date, category=20):
    '''
    Summary: Filter invoice data by date and equipment category / type
    - Filter on by category (heavy equipment like scissor and boom lifts)
    - Filter on top equipment types that account for most revenue
    - Filter by date - invoices before Oct 2014 are missing key date information

    Input: dataframe to filter, list of equipment types to keep for analysis, date to filter out any invoices prior to date, equipment category to filter out.

    Output: filtered dataframe
    '''
    # category and equipment filtering
    df = df.loc[df['Client Category'] == category, :]
    df = df.loc[df['Rental Revenue'] > 0, :]
    df = df.loc[df['Client Product Type'].isin(equipment_list), :]

    # date filtering
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.loc[df['Month'] > drop_date, :]
    df['ContractStartDate'] = pd.to_datetime(df['ContractStartDate'])
    df = df.loc[df['ContractStartDate'] >= drop_date, :]

    return df

def drop_reformat_columns(df, cols_to_drop):
    '''
    Drop columns not used in modeling. Reformat remaining columns

    Input:

    Output:
    '''
    df.drop(cols_to_drop, axis=1, inplace=True)
    cols = df.columns.tolist()
    cols = [col.lower() for col in cols]
    cols = [col.replace(' ', '_') for col in cols]
    df.columns = cols
    df.rename(columns={'client_category': 'product_category', 'client_product_type': 'product_type'}, inplace=True)

    return df

def add_features(df):
    '''
    Summary: Create new features, including:
    - Rental type to categorize invoices as daily, weekly or monthly
    - Contract start month, year and year-month (seasonality is a major factor)
    - Units rented, which is the target variable (each invoice is one unit)

    Input: dataframe where features will be added

    Output: dataframe with added features
    '''
    # rental type feature
    conditions = [
        df.loc[:, 'days'] > 0,
        df.loc[:, 'weeks'] > 0,
        df.loc[:, 'months'] > 0]
    choices = ['daily', 'weekly', 'monthly']
    df.loc[:, 'rental_type'] = np.select(conditions, choices, default='NA')
    # remove misc invoices not categorized as daily, weekly, monthly
    df = df.loc[df['rental_type'] != 'NA', :]

    # contract start month, year, and year_month columns
    df.loc[:, 'contract_start_year'] = df['contractstartdate'].dt.year
    df.loc[:, 'contract_start_month'] = df['contractstartdate'].dt.month
    df.loc[:, 'contract_start_year_month'] = pd.to_datetime(df['contractstartdate']).map(lambda dt: dt.replace(day=1))

    # units_rented features (target variable)
    df.loc[:, 'units_rented'] = 1

    return df

def add_days_rented(df):
    '''
    Summary: Create total_days_rented, which represents total days rented for all invoices regardless of invoice type

    Input: dataframe where feature will be added

    Output: dataframe with total_days_rented feature
    '''
    for idx, row in df.iterrows():
        if row['rental_type'] == 'daily':
            df.loc[idx, 'total_days_rented'] = row['days']
        elif row['rental_type'] == 'weekly':
            df.loc[idx, 'total_days_rented'] = row['weeks'] * 6
        elif row['rental_type'] == 'monthly':
            df.loc[idx, 'total_days_rented'] = row['months'] * 28
    return df

def get_month_lag(df, row):
    '''
    Summary: Used in the create_month_lag function to grab the prior month data for each record

    Input: dataframe to grab prior month data

    Output: new lag dataframe that contains only data for the prior month for a single record
    '''
    lag_date = row['contract_start_year_month'] - 1 * MonthBegin()
    mask = (lag_date == df.loc[:, 'contract_start_year_month']) & (row['product_type'] == df.loc[:, 'product_type']) & (row['rental_type'] == df.loc[:, 'rental_type'])
    lag_df = df.loc[mask, :]
    return lag_df

def create_month_lag(df_agg):
    '''
    Summary: Adds prior month data (units rented, days rented, revenue) for every record

    Input: dataframe w/o prior month data

    Output: updated dataframe with prior month data
    '''
    for idx, row in df_agg.iterrows():
        lag_df = get_month_lag(df_agg, row)
        if lag_df.empty:
            df_agg.loc[idx, 'prior_month_rental_revenue'] = 0
            df_agg.loc[idx, 'prior_month_total_days_rented'] = 0
            df_agg.loc[idx, 'prior_month_units_rented'] = 0
            df_agg.loc[idx, 'prior_month_avg_price_per_day'] = 0
        else:
            df_agg.loc[idx, 'prior_month_rental_revenue'] = lag_df['rental_revenue'].values
            df_agg.loc[idx, 'prior_month_total_days_rented'] = lag_df['total_days_rented'].values
            df_agg.loc[idx, 'prior_month_units_rented'] = lag_df['units_rented'].values
            df_agg.loc[idx, 'prior_month_avg_price_per_day'] = lag_df['avg_price_per_day'].values
    return df_agg

def get_month_avg(df, row):
    '''
    Summary: Used in the create_same_month_avg function to grab the average data for the month of the selected record

    Input: dataframe w/o same month average data

    Output: new lag dataframe that contains only data for the same month for a single record
    '''
    month = row['contract_start_month']
    mask = (month == df.loc[:, 'contract_start_month']) & (row['product_type'] == df.loc[:, 'product_type']) & (row['rental_type'] == df.loc[:, 'rental_type'])
    lag_df = df.loc[mask, :]
    return lag_df

def create_same_month_avg(df_agg, cutoff_date):
    '''
    Summary: Adds prior month data (units rented, days rented, revenue) for every record

    Input: cutoff_date (date for same month averages to be considered) and dataframe w/o monthly average data

    Output: updated dataframe with monthly average data. Note: monthly averages only considered before cutoff date (useful to avoid data leakage)
    '''
    df_agg_avg_month = df_agg.loc[df_agg['contract_start_year_month'] < cutoff_date, :]
    df_agg_avg_month = df_agg_avg_month.groupby(['product_type', 'contract_start_month', 'rental_type']).agg({'units_rented': 'mean', 'total_days_rented': 'mean'}).reset_index()
    for idx, row in df_agg.iterrows():
        lag_df = get_month_avg(df_agg_avg_month, row)
        if lag_df.empty:
            df_agg.loc[idx, 'same_month_avg_units_rented'] = 0
            df_agg.loc[idx, 'same_month_avg_days_rented'] = 0
        else:
            df_agg.loc[idx, 'same_month_avg_units_rented'] = lag_df['units_rented'].values
            df_agg.loc[idx, 'same_month_avg_days_rented'] = lag_df['total_days_rented'].values
    return df_agg

def create_dummies(dummy_col, df, col_prefix):
    '''
    Summary: Creates dummy features for categorical features

    Input: dummy_col: categorical feature column to create dummies, df: dataframe to add dummies, col_prefix: string to add as prefix to dummy column name

    Output: updated dataframe with categorical features replaced as dummy features
    '''
    dum_col = df[dummy_col]
    dummies = pd.get_dummies(dum_col, prefix=col_prefix)
    df = df.drop([dummy_col], axis=1)
    df_w_dummies = df.merge(dummies, left_index=True, right_index=True)
    return df_w_dummies

def calc_rmse(true, predicted):
    '''
    Summary: Calculate root mean squared error

    Input: true: target labels. predicted: predicted labels from model

    Output: root mean squared error metric
    '''
    residuals_squared = (predicted - true)**2
    variance = sum(residuals_squared) / len(true)
    rmse = np.sqrt(variance)
    return rmse

if __name__ == '__main__':

    '''LOAD / CLEAN DATA; ENGINEER FEATURES'''

    # load data
    df_rev = load_data('/Users/mwirvine/Galvanize/dsi-immersive/one-source-capstone/Data Feeds/ONE - Revenue Data.txt', separator='|')

    # filter on category 20 (booms, scissors, etc), and top 30 equipment types that account for 95% of revenue (or top 17 equipment types that account for 80% of revenue). Drop data up until Oct 2014 (all of nulls in start dates prior to this)
    top_30_equipment = ['20-873', '20-925', '20-847', '20-220', '20-720', '20-740', '20-240', '20-530', '20-810', '20-840', '20-760', '20-820', '20-890', '20-870', '20-580', '20-270', '20-842', '20-610', '20-210', '20-170', '20-910', '20-940', '20-430', '20-865', '20-860', '20-260', '20-310', '20-565', '20-812', '20-730', '20-350', '20-926']

    top_17_equipment = ['20-873', '20-925', '20-847', '20-220', '20-720', '20-740', '20-240', '20-530', '20-810', '20-840', '20-760', '20-820', '20-890', '20-870', '20-580', '20-270', '20-842']

    df_rev = filter_data(df_rev, top_30_equipment, '2014-10-01', 20)

    ## Drop and reformat columns
    cols_to_drop = ['Client Code', 'Client Region', 'CreditAmount', 'Client District', 'Client Category', 'Country', 'Equipment ID #', 'IsSubstitution', 'Description', 'Transaction ID #', 'IsRPO', 'NationalAcctCode', 'Sales Rep #', 'Sales Rep Name', 'Customer Name', 'Minimums', 'IsNationalAcct', 'IsSpecialPricing', 'IsContract', 'ContractPriceNo', 'CycleBillNo', 'Jobsite ZipCode', 'Branch ZipCode', 'CustomerCorpID', 'Customer #', 'Minimum Revenue', 'Minimum Book Revenue', 'ExcessMileageRevenue', 'UnclassifiedRevenue']
    df_rev = drop_reformat_columns(df_rev, cols_to_drop)

    # fill NaNs with 0s (can do this because all NaNs represent zero (e.g., if monthly rental has zero weekly revenue, can put weekly rental as 0))
    df_rev.fillna(0, inplace=True)

    # add features (rental_type, contract start dates, units rented)
    df_rev = add_features(df_rev)

    # create total days rented column for daily, weekly, monthly rentals
    df_rev = add_days_rented(df_rev)

    # create aggregated dataset and price per day feature
    df_agg = df_rev.groupby(['product_type', 'contract_start_year_month', 'contract_start_month', 'rental_type']).agg({'rental_revenue':'sum', 'total_days_rented': 'sum', 'units_rented':'sum'}).reset_index()
    df_agg['avg_price_per_day'] = df_agg['rental_revenue'] / df_agg['total_days_rented']

    # create month lag columns (prior month units rented, prior month total days rented, prior month rental revenue)
    df_agg.sort_values('contract_start_year_month', inplace=True)
    df_agg.reset_index(drop=True, inplace=True)
    df_agg = create_month_lag(df_agg)

    # create same month avg units rented using only train dates
    train_cutoff_date = '2017-05-01'
    df_agg = create_same_month_avg(df_agg, train_cutoff_date)

    # remove October 2014 data as there is no prior month
    df_agg = df_agg.loc[df_agg['contract_start_year_month'] > '2014-10-01', :]
    df_agg.reset_index(drop=True, inplace=True)

    # fill NaNs for prior month features with zeros as there was no prior month rental revenue
    df_agg.fillna(0, inplace=True)

    # create dummy features for categorical features
    df_agg = create_dummies('product_type', df_agg, 'product_type')
    df_agg = create_dummies('contract_start_month', df_agg, 'contract_start_month')
    df_agg = create_dummies('rental_type', df_agg, 'rental_type')

    # create train / test sets (test will be May 2017 - April 2018, train will be Nov 2014 - April 2017)
    df_agg = df_agg.loc[df_agg['contract_start_year_month'] < '2018-05-01', :]
    df_agg.to_pickle('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-Equipment-Rental-Data/df_agg.pkl')
    df_agg_test = df_agg.loc[df_agg['contract_start_year_month'] >= '2017-05-01', :]
    df_agg_train = df_agg.loc[df_agg['contract_start_year_month'] < '2017-05-01', :]

    # check correlation
    corr_matrix = df_agg_train.drop(['rental_revenue', 'total_days_rented', 'avg_price_per_day'], axis=1).corr()
    corr_matrix['units_rented'].sort_values(ascending=False)



    '''EXECUTE MODELS'''

    # create X,y matrices. Note: need to drop rental_revenue, 'rental_revenue', 'total_days_rented', 'avg_price_per_day' as those are current month data and cannot predict using them
    X_train = df_agg_train.drop(['units_rented', 'contract_start_year_month', 'rental_revenue', 'total_days_rented', 'avg_price_per_day'], axis=1).values
    y_train = df_agg_train['units_rented'].values
    X_test = df_agg_test.drop(['units_rented', 'contract_start_year_month', 'rental_revenue', 'total_days_rented', 'avg_price_per_day'], axis=1).values
    y_test = df_agg_test['units_rented'].values

    # standardize data
    standardizer = StandardScaler()
    standardizer.fit(X_train, y_train)
    X_train_std = standardizer.transform(X_train)
    X_test_std = standardizer.transform(X_test)
    del X_train
    del X_test

    # create model objects
    linear = LinearRegression(n_jobs=-1)
    lasso = Lasso(alpha=.1)
    rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', n_jobs=-1, min_samples_leaf=4, min_samples_split=8)
    gbr = GradientBoostingRegressor(n_estimators=1000, max_depth=5, learning_rate=.1, max_features='sqrt', min_samples_leaf=4, min_samples_split=8)
    models = [(linear, 'Linear'), (lasso, 'Lasso'), (rf, 'Random Forest'), (gbr, 'Gradient Boosting')]

    # run each model, print results, add predictions to test dataframe
    for model in models:
        model[0].fit(X_train_std, y_train)
        y_pred_train = model[0].predict(X_train_std)
        y_pred_test = model[0].predict(X_test_std)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        print('{} RMSE train results: {:.3f}'.format(model[1], train_rmse))
        print('{} RMSE test results: {:.3f}'.format(model[1], test_rmse))
        df_agg_test.loc[:, 'units_rented_pred_' + model[1]] = y_pred_test

    # calculate rmse for same month avg units rented as baseline comparison
    y_pred_train_same_month = df_agg_train.loc[:, 'same_month_avg_units_rented'].values
    y_pred_test_same_month = df_agg_test.loc[:, 'same_month_avg_units_rented'].values
    rmse_same_month_train = calc_rmse(y_train, y_pred_train_same_month)
    rmse_same_month_test = calc_rmse(y_test, y_pred_test_same_month)
    print("Same Month Avg RMSE train results: {:.3f}".format(rmse_same_month_train))
    print("Same Month Avg RMSE test results: {:.3f}".format(rmse_same_month_test))

    # save the final train/test dfs
    df_agg_test.to_pickle('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-Equipment-Rental-Data/df_agg_test.pkl')
    df_agg_train.to_pickle('/Users/mwirvine/Galvanize/dsi-immersive/Capstone-Equipment-Rental-Data/df_agg_train.pkl')


    '''INVESTIGATE DETAILS OF MODELS'''

    # OLS linear regression to look at coefficients
    X_train_std_wconst = sm.add_constant(X_train_std)
    X_test_std_wconst = sm.add_constant(X_test_std)
    ols = sm.OLS(y_train, X_train_std_wconst).fit()
    print(ols.summary())
    #y_pred_test_linear_ols = ols.predict(X_test_std_wconst)
    ols_res = ols.outlier_test()[:,0]
    y_fitted = ols.fittedvalues
    coeffs_ols = ols.params
    features_ols = df_agg_train.drop(['units_rented', 'contract_start_year_month', 'rental_revenue', 'total_days_rented', 'avg_price_per_day'], axis=1).columns
    coeffs_by_feature = sorted(zip(coeffs_ols, features_ols), reverse=True)

    # look at coefficients from lasso model
    features_lasso = df_agg_train.drop(['units_rented', 'contract_start_year_month', 'rental_revenue', 'total_days_rented', 'avg_price_per_day'], axis=1).columns
    coeffs_lasso = lasso.coef_
    coeffs_by_feature_lasso = sorted(zip(coeffs_lasso, features_lasso), reverse=True)





    '''
    Model testing / tuning results:

    1) Full feature matrix - month only - linear rmse - 13.5. RF rmse - 15
    RF looks to be overfitting as train is 5.64 and test is 15. Linear doesn't look
    to be overfitting as train and test are ~13.5

    2) Same as above, but with standardization. RF the same. Linear test rmse is enormous now...what is happening?

    3) Reduced product category from to top 30 pieces of equipment by revenue out of 60 total types. Top 30 pieces make up ~95% of revenue. Also tried 80/20 rule which was 17 pieces of equipment. Similar results. Linear rmse train / test = 15.65 / billions. RF train / test = 6.47 / 17

    4) removed contract start year as a variable...results about the same except linear test is now not off the charts. Linear rmse test = 15.4. Not sure if it makes sense to do this as now what does "prior month" mean if there is no year associated. why did standardization for linear test get better?

    5) feature matrix with all daily, weekly and monthly data. Added 'rental_type' as a categorical variable. Linear rmse train/test = 11.74 / billions. RF rmse train/test = 4.12 / 10.41. Standardized data for linear regression is giving crazy huge test results. Looked at predictions and the last 345 predictions are all the same and enormous. HAVE TO REMOVE year features for linear regression!

    IMPORTANT - predictions with year categoricals are MESSED UP. REMOVE? 2018 categorical has insance predictions - try removing. ASK LAND/FRANK! this looks to have worked by just removing those year categorical variables

    6) same as above but removed contract_start_year features. Linear rmse train / test = 11.75 / 10.68. RF rmse train / test = 4.04 / 10.64

    7) same as above but added natl construction spend data. NO IMPACT. train/test rmses do not change much. Correlation is .02. ADD THIS TO README

    8) tried MLP - cannot get the rmse to be any better than RF or linear

    9) plotted same month avg and looked at correlation - extremely difficult to beat as the correlation with same month avg is .94. ADD THIS TO README!

    '''
