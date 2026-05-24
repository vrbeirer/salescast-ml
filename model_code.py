import pandas as pd
import numpy as np
import seaborn as sns
from fuzzywuzzy import fuzz
from itertools import product
import time

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
%matplotlib notebook

#Functions#
def matching_pairs_within_column_name(df, column_name, scorer):
    from fuzzywuzzy import process, fuzz
    unique_col_values = df[column_name].unique()
    #Create a dataframe from the tuples
    score_sort = [(x,) + i
                 for x in unique_col_values 
                 for i in process.extract(x, unique_col_values,     scorer=scorer)]
    
    similarity_sort = pd.DataFrame(score_sort, columns=[column_name,'match_sort','score_sort'])

    similarity_sort['sorted_' + column_name] = np.minimum(similarity_sort[column_name], similarity_sort['match_sort'])

    high_score_sort = similarity_sort[(similarity_sort['score_sort'] >= 80) &
                    (similarity_sort[column_name] !=  similarity_sort['match_sort']) &
                    (similarity_sort['sorted_' + column_name] != similarity_sort['match_sort'])]

    high_score_sort = high_score_sort.drop('sorted_' + column_name,axis=1).copy()

    return high_score_sort.groupby([column_name,'score_sort']).agg(
                            {'match_sort': ', '.join}).sort_values(
                            ['score_sort'], ascending=False)
import re
def name_correction(x):
    x = x.lower() # all letters lower case
    x = x.partition('[')[0] # partition by square brackets
    x = x.partition('(')[0] # partition by curly brackets
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # remove special characters
    x = x.replace('  ', ' ') # replace double spaces with single spaces
    x = x.strip() # remove leading and trailing white space
    return x

# Define a lag feature function
def lag_feature( df,lags, cols ):
    for col in cols:
        print(col)
        tmp = df[["date_block_num", "shop_id","item_id",col ]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

def runCV(alg, X_train, Y_train, scoring, cv):
    print('Running CV...')
    scores = cross_validate(alg, X_train, Y_train, 
                            scoring=scoring, cv = cv)
    print('Done!')
    return scores

def TimeSeriesIters(X, ts_col, split_block):

    num_blocks = X[ts_col].max()

    train_iters = []
    test_iters = []

    for blocks in range(split_block, num_blocks):
        train_iters.append(X[(X[ts_col] <= blocks)].index)
        test_iters.append(X[(X[ts_col] == blocks+1)].index)
        
    return train_iters, test_iters

sales_train_df = pd.read_csv('data/sales_train.csv')
items_categories_df = pd.read_csv('data/item_categories.csv')
items_df = pd.read_csv('data/items.csv')
sample_submission_df = pd.read_csv('data/sample_submission.csv')
shops_df = pd.read_csv('data/shops.csv')
test_df = pd.read_csv('data/test.csv')

sales_train_df.dtypes

plt.figure(figsize=(10,4))
flierprops = dict(marker='o', markerfacecolor='blue', markersize=6,
                  linestyle='none', markeredgecolor='black')
sns.boxplot(x=sales_train_df.item_cnt_day, flierprops=flierprops)

plt.figure(figsize=(10,4))
sns.boxplot(x=sales_train_df.item_price, flierprops=flierprops)

# outlier removal
sales_train_df = sales_train_df[sales_train_df.item_cnt_day < 900]
sales_train_df = sales_train_df[sales_train_df.item_price < 300000]

print((sales_train_df.item_cnt_day > 0).all())
(sales_train_df.item_price > 0).all()

items_categories_df.nunique()

last_month_train_df = sales_train_df[sales_train_df.date_block_num == 33]
for shop in last_month_train_df.shop_id.unique():
    if last_month_train_df[last_month_train_df.shop_id==shop].item_cnt_day.sum() <= 0:
        print("shop id {} didn't sell in the last month".format(shop))

matching_pairs_within_column_name(shops_df, 'shop_name', fuzz.token_set_ratio)

shops_df["shop_name_split_1"] = shops_df.shop_name.str.split(" ").map( lambda x: x[0] )
shops_df["shop_name_split_2"] = shops_df.shop_name.str.split(" ").map( lambda x: x[1] )

matching_pairs_within_column_name(shops_df, 'shop_name_split_1', fuzz.token_set_ratio)

# adjust Якутск
shops_df.loc[shops_df.shop_name_split_1 == "!Якутск", "shop_name_split_1"] = "Якутск"

# Categorize shop_name_split_1 so that values that only appear once are labeled as other. Helpful for one-hot encoding.
shops_df.shop_name_split_1.value_counts();
cat_thresh = 2
category = []
for cat in shops_df.shop_name_split_1.unique():
    if len(shops_df[shops_df.shop_name_split_1 == cat]) >= cat_thresh:
        category.append(cat)
shops_df.shop_name_split_1 = shops_df.shop_name_split_1.apply( lambda x: x if (x in category) else "other" )

matching_pairs_within_column_name(shops_df, 'shop_name_split_2', fuzz.token_set_ratio)

shops_df.shop_name_split_2.value_counts();
# Categorize shop_name_split_2 so that values that only appear once are labeled as other. Helpful for numerical label encoding.
shops_df.shop_name_split_2.value_counts();
cat_thresh = 2
category = []
for cat in shops_df.shop_name_split_2.unique():
    if len(shops_df[shops_df.shop_name_split_2 == cat]) >= cat_thresh:
        category.append(cat)
shops_df.shop_name_split_2 = shops_df.shop_name_split_2.apply( lambda x: x if (x in category) else "other" )

# label encode the shop data
from sklearn.preprocessing import LabelEncoder
shops_df["shop_name_split_1_encode"] = LabelEncoder().fit_transform( shops_df.shop_name_split_1 )
shops_df["shop_name_split_2_encode"] = LabelEncoder().fit_transform( shops_df.shop_name_split_2 )
shops_df = shops_df[["shop_id", "shop_name_split_1_encode", "shop_name_split_2_encode"]]

# sanity check
shops_df.head()

items_categories_df.head()

items_categories_df.loc[items_categories_df.item_category_name.str.contains("PC"),  'item_category_name']

items_categories_df.loc[items_categories_df.item_category_name == 'PC - Гарнитуры/Наушники', "item_category_name"] = "Гарнитуры/Наушники - PC"
items_categories_df.loc[items_categories_df.item_category_name == 'Игры PC - Дополнительные издания', "item_category_name"] = "Дополнительные издания - Игры PC"
items_categories_df.loc[items_categories_df.item_category_name == 'Игры PC - Коллекционные издания', "item_category_name"] = "Коллекционные издания - Игры PC"
items_categories_df.loc[items_categories_df.item_category_name == 'Игры PC - Стандартные издания', "item_category_name"] = "Стандартные издания - Игры PC"
items_categories_df.loc[items_categories_df.item_category_name == 'Игры PC - Цифра', "item_category_name"] = "Цифра - Игры PC"

# Let's split on '-'
items_categories_df["item_cat_split_1"] = items_categories_df.item_category_name.str.split("-").str[0]
items_categories_df["item_cat_split_2"] = items_categories_df.item_category_name.str.split("-").str[1].fillna("unknown")

# Categorize item_cat_split_1
cat_thresh = 2
category = []
for cat in items_categories_df.item_cat_split_1.unique():
    if len(items_categories_df[items_categories_df.item_cat_split_1 == cat]) >= cat_thresh:
        category.append(cat)
items_categories_df.item_cat_split_1 = items_categories_df.item_cat_split_1.apply( lambda x: x if (x in category) else "other" )

# Categorize item_cat_split_2
cat_thresh = 2
category = []
for cat in items_categories_df.item_cat_split_2.unique():
    if len(items_categories_df[items_categories_df.item_cat_split_2 == cat]) >= cat_thresh:
        category.append(cat)
items_categories_df.item_cat_split_2 = items_categories_df.item_cat_split_2.apply( lambda x: x if (x in category) else "other" )

# label encode the item cat data
items_categories_df["item_cat_split_1_encode"] = LabelEncoder().fit_transform( items_categories_df.item_cat_split_1 )
items_categories_df["item_cat_split_2_encode"] = LabelEncoder().fit_transform( items_categories_df.item_cat_split_2 )
items_categories_df = items_categories_df[["item_category_id", "item_cat_split_1_encode", "item_cat_split_2_encode"]]

# sanity check
items_categories_df.head()

items_df["name1"], items_df["name2"] = items_df.item_name.str.split("[", 1).str
items_df["name1"], items_df["name3"] = items_df.item_name.str.split("(", 1).str

items_df["name2"] = items_df.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items_df["name3"] = items_df.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

items_df = items_df.fillna('0')

items_df["item_name"] = items_df["item_name"].apply(lambda x: name_correction(x))

# drop the last bracket in name2 that was left from the previous str.split
items_df.name2 = items_df.name2.apply( lambda x: x[:-1] if x !="0" else "0")

# The lower the ratio, the 'more categorical' the column
print("name3 ratio: {}".format(items_df.name3.nunique()/items_df.shape[0]))
print("name2 ratio: {}".format(items_df.name2.nunique()/items_df.shape[0]))
print("name1 ratio: {}".format(items_df.name1.nunique()/items_df.shape[0]))

# Let's remove white space before/after
items_df["name2"] = items_df.name2.str.strip()

# the majority of game consoles above is of type xbox (xbox one, xbox 360, xbox360 and x360). 
# we can get most of these by searching over first 8 chars and saving that, then assigning everyting else
# to be just whatever comes first. We don't want to do str.contains() because some rows values have multiple 
# gaming consoles listed.
items_df['type'] = items_df.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0])

items_df['type'].unique()

items_df.loc[items_df['type'] == '', 'type'] = 'pc'

# and the rest..
items_df.loc[(items_df.type == "x360") | (items_df.type == "xbox360") | (items_df.type == "xbox 360") ,"type"] = "xbox 360"
items_df.loc[ (items_df.type == 'pc' )| (items_df.type == 'рс') | (items_df.type == 'pс'), "type" ] = "pc"

# Check again
items_df['type'].unique()

group_item_sum = items_df.groupby(["type"]).agg({"item_id": "count"}).reset_index()

drop_cols = []
for cat in group_item_sum.type.unique():
    if group_item_sum.loc[(group_item_sum.type == cat), "item_id"].values[0] <=20:
        drop_cols.append(cat)
items_df.type = items_df.type.apply( lambda x: "other" if (x in drop_cols) else x )
#items_df = items_df.drop(["name2"], axis = 1)

# value counts looks good - the low-volume values were grouped 55 times as 'other'.
items_df.type.value_counts()

items_df.type = LabelEncoder().fit_transform(items_df.type)
items_df.name3 = LabelEncoder().fit_transform(items_df.name3)
items_df.drop(["item_name", "name1", 'name2'],axis = 1, inplace= True)
items_df.head()

# This nifty piece of code will 'productize' uniquely the month number, shop id and item id into one large matrix,
# and then place it vertically in a dataframe! Finally, we'll set the data types more accurately, and do a sort.
ts = time.time()
matrix = []
cols  = ["date_block_num", "shop_id", "item_id"]
for i in range(34):
    sales = sales_train_df[sales_train_df.date_block_num == i]
    matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )
    
matrix = pd.DataFrame( np.vstack(matrix), columns = cols )
matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
matrix["item_id"] = matrix["item_id"].astype(np.int16)
matrix.sort_values( cols, inplace = True )
time.time()- ts

# add revenue to train df
sales_train_df["revenue"] = sales_train_df["item_cnt_day"] * sales_train_df["item_price"]

ts = time.time()
group = sales_train_df.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
group.columns = ["item_cnt_month"]
group.reset_index( inplace = True)
matrix = pd.merge( matrix, group, on = cols, how = "left" )
matrix["item_cnt_month"] = matrix["item_cnt_month"].fillna(0).astype(np.float16)
time.time() - ts

test_df.head()

test_df["date_block_num"] = 34
test_df["date_block_num"] = test_df["date_block_num"].astype(np.int8)
test_df["shop_id"] = test_df.shop_id.astype(np.int8)
test_df["item_id"] = test_df.item_id.astype(np.int16)

# merge!
ts = time.time()

matrix = pd.concat([matrix, test_df.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
matrix.fillna( 0, inplace = True ) # not all shop/item_id combos will be in the test set.
time.time() - ts

ts = time.time()
matrix = pd.merge( matrix, shops_df, on = ["shop_id"], how = "left" )
matrix = pd.merge(matrix, items_df, on = ["item_id"], how = "left")
matrix = pd.merge( matrix, items_categories_df, on = ["item_category_id"], how = "left" )
matrix["shop_name_split_1_encode"] = matrix["shop_name_split_1_encode"].astype(np.int8)
matrix["shop_name_split_2_encode"] = matrix["shop_name_split_2_encode"].astype(np.int8)
matrix["item_cat_split_1_encode"] = matrix["item_cat_split_1_encode"].astype(np.int8)
matrix["item_cat_split_2_encode"] = matrix["item_cat_split_2_encode"].astype(np.int8)
matrix["item_type"] = matrix["type"].astype(np.int8)
matrix["item_name3"] = matrix["name3"].astype(np.int16)
matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)
time.time() - ts

matrix.drop(["type", "name3"],axis = 1, inplace= True)

matrix.head()

ts = time.time()
matrix = lag_feature( matrix, [1,2,3], ["item_cnt_month"] )
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')

# set the data type
matrix.date_avg_item_cnt = matrix["date_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1], ['date_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_item_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')

# set the data type
matrix.date_item_avg_item_cnt = matrix["date_item_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_item_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_shop_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id'], how='left')

# set the data type
matrix.date_shop_avg_item_cnt = matrix["date_shop_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_shop_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_shop_item_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_id'], how='left')

# set the data type
matrix.date_shop_item_avg_item_cnt = matrix["date_shop_item_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_shop_item_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_shop_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'shop_id', 'item_cat_split_1_encode']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_shop_itemcat1_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_cat_split_1_encode'], how='left')

# set the data type
matrix.date_shop_itemcat1_avg_item_cnt = matrix["date_shop_itemcat1_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_shop_itemcat1_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_shop_itemcat1_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'shop_id', 'item_cat_split_2_encode']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_shop_itemcat2_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_cat_split_2_encode'], how='left')

# set the data type
matrix.date_shop_itemcat2_avg_item_cnt = matrix["date_shop_itemcat2_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_shop_itemcat2_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_shop_itemcat2_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_shop_itemcat_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')

# set the data type
matrix.date_shop_itemcat_avg_item_cnt = matrix["date_shop_itemcat_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_shop_itemcat_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_shop_itemcat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'shop_id', 'item_type']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_shop_itemtype_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_type'], how='left')

# set the data type
matrix.date_shop_itemtype_avg_item_cnt = matrix["date_shop_itemtype_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_shop_itemtype_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_shop_itemtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

ts = time.time()
# create the group (because we are doing an agg function)
group = matrix.groupby(['date_block_num', 'shop_id', 'item_name3']).agg({'item_cnt_month': ['mean']})
group.columns= (['date_shop_itemname3_avg_item_cnt'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_name3'], how='left')

# set the data type
matrix.date_shop_itemname3_avg_item_cnt = matrix["date_shop_itemname3_avg_item_cnt"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_shop_itemname3_avg_item_cnt'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_shop_itemname3_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

group = sales_train_df.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns= (['item_avg_item_price'])
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix["item_avg_item_price"] = matrix.item_avg_item_price.astype(np.float16)

ts = time.time()
# create the group (because we are doing an agg function)
group = sales_train_df.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
group.columns= (['date_avg_item_price'])
group.reset_index(inplace=True)

# then let's merge back to the matrix to add the new column
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')

# set the data type
matrix.date_avg_item_price = matrix["date_avg_item_price"].astype(np.float16)

# then lag the feature we just added
matrix = lag_feature(matrix, [1,2,3], ['date_avg_item_price'])

# drop the newly created feature since it's already been lagged
matrix.drop(['date_avg_item_price'], axis=1, inplace=True)
time.time() - ts

matrix.head().T

import math
import gc
import pickle
from xgboost import XGBRegressor
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, TimeSeriesSplit, GridSearchCV, cross_val_score, cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

data = matrix.copy()
#del matrix
#gc.collect()

data[data["date_block_num"]==34].shape

# if reloading...
data = pd.read_pickle("./data/data.pkl")

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)

data.to_pickle("./data/data.pkl")

del data
gc.collect();

ts = time.time()

model_lr = linear_model.LinearRegression(n_jobs=-1)

model_lr.fit(X_train.fillna(0), Y_train.fillna(0))

# RMSE for Train
preds = model_lr.predict(X_train.fillna(0))
rmse_train = math.sqrt(mean_squared_error(Y_train.fillna(0), preds))
# RMSE for Valid
preds = model_lr.predict(X_valid.fillna(0))
rmse_valid = math.sqrt(mean_squared_error(Y_valid.fillna(0), preds))

print("Train RMSE: {0:.5f}".format(rmse_train))
print("Validation RMSE: {0:.5f}".format(rmse_valid))

time.time() - ts

X = pd.concat([X_train, X_valid])
y = pd.concat([Y_train, Y_valid])

train_iters, test_iters = TimeSeriesIters(X, 'date_block_num', 28)

ts = time.time()

model_lr = linear_model.LinearRegression(n_jobs=-1, normalize=True)
scores = cross_val_score(model_lr, X.fillna(0), y.fillna(0), scoring='neg_root_mean_squared_error', cv=zip(train_iters, test_iters),
               n_jobs=-1)

time.time() - ts

print("RMSE Validation scores: {}".format(scores*-1))
print("mean: {}".format(scores.mean()*-1))

ts = time.time()
model_la= linear_model.Lasso(alpha=2.0, random_state=42)
scores = cross_val_score(model_la, X.fillna(0), y.fillna(0), scoring='neg_root_mean_squared_error', cv=zip(train_iters, test_iters),
               n_jobs=-1)
time.time() - ts

print("RMSE Validation scores: {}".format(scores*-1))
print("mean: {}".format(scores.mean()*-1))

# change file_name
file_name = 'lr_submission_mrj.csv'

Y_test = model_lr.predict(X_test.fillna(0)).clip(0, 20)

submission = pd.DataFrame({
    "ID": test_df.index, 
    "item_cnt_month": Y_test
})
submission.to_csv(file_name, index=False)

! kaggle competitions submit -c competitive-data-science-predict-future-sales -f {file_name} -m "lr"

ts = time.time()

model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
#     tree_method='gpu_hist',
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 20)

time.time() - ts

# change file_name
file_name = 'fe_xgb_submission_mrj.csv'

Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test_df.index, 
    "item_cnt_month": Y_test
})
submission.to_csv(file_name, index=False)

! kaggle competitions submit -c competitive-data-science-predict-future-sales -f {file_name} -m "xgboost with feature engineering"

from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))

ts = time.time()

# Call GridSearch function
parameters = {'eta':[0.1, 0.2, 0.3]}

model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8,
#     tree_method='gpu_hist',
    seed=42)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
reg_xgb_1 = GridSearchCV(model, param_grid=parameters, cv=zip(train_iters, test_iters), verbose=True,
                        scoring='neg_root_mean_squared_error')
reg_xgb_1.fit(X, y)

#reg_xgb_1.best_score_, reg_xgb_1.best_params_

time.time() - ts

