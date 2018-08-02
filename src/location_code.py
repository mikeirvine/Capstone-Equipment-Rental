'''Code to use location code in equipment rental demand modeling'''

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

# Remove two corporate rental invoices (not needed with reduced product types)

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
