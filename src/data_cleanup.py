import pandas as pd
import numpy as np

## Load data

df_rev = pd.read_csv('/Users/mwirvine/Galvanize/dsi-immersive/one-source-capstone/Data Feeds/ONE - Revenue Data.txt', sep='|')

## Filter on category 20 (booms, scissors, etc), remove credit txns (don't consider invoice credit txns in demand prediction)

df_rev = df_rev[df_rev['Client Category'] == 20]

df_rev = df_rev[df_rev['Rental Revenue'] > 0]


## Drop and reformat columns

cols_to_drop = ['Client Code', 'Client Region', 'CreditAmount', 'Client District', 'Client Category', 'Country', 'Equipment ID #', 'IsSubstitution', 'Description', 'Transaction ID #', 'IsRPO', 'NationalAcctCode', 'Sales Rep #', 'Sales Rep Name', 'Customer Name', 'Minimums', 'IsNationalAcct', 'IsSpecialPricing', 'IsContract', 'ContractPriceNo', 'CycleBillNo', 'Jobsite ZipCode', 'Branch ZipCode', 'CustomerCorpID', 'Customer #', 'Minimum Revenue', 'Minimum Book Revenue']

df_rev.drop(cols_to_drop, axis=1, inplace=True)

cols = df_rev.columns.tolist()
cols = [col.lower() for col in cols]
cols = [col.replace(' ', '_') for col in cols]
df_rev.columns = cols

df_rev.rename(columns={'client_category': 'product_category', 'client_product_type': 'product_type'}, inplace=True)

## Reformat date columns

df_rev['month'] = pd.to_datetime(df_rev['month'])

## Drop 2013/2014 data (all of nulls in contract start date)

df_rev = df_rev[df_rev['month'] > '2014-12-01']

## Reformat ContractStartDate date columns

df_rev['contractstartdate'] = pd.to_datetime(df_rev['contractstartdate'])

## Clean up location codes

df_rev['location_code'].replace(['001'], 100, inplace=True)
df_rev['location_code'].replace(['002'], 200, inplace=True)
df_rev['location_code'].replace(['003'], 300, inplace=True)
df_rev['location_code'].replace(['004'], 400, inplace=True)
df_rev['location_code'].replace(['005'], 500, inplace=True)
df_rev['location_code'].replace(['006'], 600, inplace=True)
df_rev['location_code'].replace(['007'], 700, inplace=True)
df_rev['location_code'].replace(['008'], 800, inplace=True)
df_rev['location_code'].replace(['009'], 900, inplace=True)
df_rev['location_code'].replace(['101'], 101, inplace=True)
df_rev['location_code'].replace(['103'], 103, inplace=True)
df_rev['location_code'].replace(['201'], 201, inplace=True)
df_rev['location_code'].replace(['301'], 301, inplace=True)
df_rev['location_code'].replace([1], 100, inplace=True)
df_rev['location_code'].replace([2], 200, inplace=True)
df_rev['location_code'].replace([3], 300, inplace=True)
df_rev['location_code'].replace([4], 400, inplace=True)
df_rev['location_code'].replace([5], 500, inplace=True)
df_rev['location_code'].replace([6], 600, inplace=True)
df_rev['location_code'].replace([7], 700, inplace=True)
df_rev['location_code'].replace([8], 800, inplace=True)
df_rev['location_code'].replace([9], 900, inplace=True)

# Remove two corporate rental invoices
df_rev = df_rev[df_rev['location_code'] != 'COR']

# Change column to a numeric column
df_rev['location_code'] = pd.to_numeric(df_rev['location_code'])

# Create branch name column (for a dummy later)

conditions = [
    df_rev['location_code'] == 100,
    df_rev['location_code'] == 200,
    df_rev['location_code'] == 300,
    df_rev['location_code'] == 400,
    df_rev['location_code'] == 500,
    df_rev['location_code'] == 600,
    df_rev['location_code'] == 700,
    df_rev['location_code'] == 800,
    df_rev['location_code'] == 900,
    df_rev['location_code'] == 101,
    df_rev['location_code'] == 103,
    df_rev['location_code'] == 201,
    df_rev['location_code'] == 301]

choices = ['LAF', 'KNX', 'DEC', 'GRA', 'MOR', 'TER', 'LOU', 'IND', 'CLA', 'LAF-NUC', 'DEC-STA', 'LAF-FRE', 'LAF-SDI']
df_rev['branch_name'] = np.select(conditions, choices, default='TBD')

# create rental type column (rental was monthly, weekly, daily, hourly)

conditions = [
    df_rev['hours'] > 0,
    df_rev['days'] > 0,
    df_rev['weeks'] > 0,
    df_rev['months'] > 0]

choices = ['hourly', 'daily', 'weekly', 'monthly']
df_rev['rental_type'] = np.select(conditions, choices, default='TBD')

# fill NaNs with 0s (can do this because all NaNs represent zero (e.g., if monthly rental has zero
# weekly revenue, can put weekly rental as 0))

df_rev.fillna(0, inplace=True)

# create contract start month and year columns

df_rev['contract_start_year'] = df_rev['contractstartdate'].dt.year
df_rev['contract_start_month'] = df_rev['contractstartdate'].dt.month

# df_rev.groupby(['branch_name', 'product_type']).agg({'rental_revenue':'count'})

# next steps: remove location_code, contractstartdate
# get dummies on product type, rental_type, branch_name
# make some demand and rev plots by contract start month (across years)
# this should help me figure out how to aggregate (thinking about model for whole company first)
