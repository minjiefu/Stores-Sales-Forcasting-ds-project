
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
import patsy
from sklearn.decomposition import PCA

## SVD to process x_train to remove noises
def svd(train, d = 8):
    num_dept = max(train['Dept'])
    new_train = pd.DataFrame()
    for i in range(1, num_dept + 1):
        filtered_train = train[train['Dept'] == i]
        selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]
        train_dept_ts = selected_columns.pivot(index='Store', columns='Date', values='Weekly_Sales').reset_index()
        
        train_dept_ts.fillna(0, inplace=True)
        
        X = train_dept_ts.iloc[:, 2:].values
        store_means = X.mean(axis=1, keepdims=True)
        X_centered = X - store_means
        
        if len(X_centered) == 0:
            continue
            
        rank_X = np.linalg.matrix_rank(X_centered)
        if rank_X > d:
            pca = PCA(n_components=d)
            X_reduced = pca.fit_transform(X_centered)
            X_reduced = pca.inverse_transform(X_reduced) + store_means

        else:
            X_reduced = X
            
        df = pd.DataFrame(X_reduced, columns=train_dept_ts.columns[2:])
        df = pd.concat([train_dept_ts.iloc[:, :2], df], axis=1)
        
        
        tmp_train = df.melt(id_vars='Store', var_name='Date', value_name='Weekly_Sales')
        tmp_train["Dept"] = int(i)
        tmp_train["Dept"] = tmp_train["Dept"].astype('category')
        
        new_train = pd.concat([new_train, tmp_train], ignore_index=True)
        
    return new_train

## Break 'Date' with 'Wk' and 'Yr'
def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks 
    return data

## Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = svd(train, d = 8)

## Use this methods to shorten running time
test_pred = pd.DataFrame()


train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
train_split = preprocess(train_split)
train_split['Yr_squared'] = np.power(train_split['Yr'], 2)
## Add Yr_squared to improve performance
X = patsy.dmatrix('Weekly_Sales + Store + Dept + Yr + Yr_squared + Wk',
                    data = train_split,
                    return_type='dataframe')


train_split = dict(tuple(X.groupby(['Store', 'Dept'])))


test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
test_split = preprocess(test_split)

test_split['Yr_squared'] = np.power(test_split['Yr'], 2)
X = patsy.dmatrix('Store + Dept + Yr + Yr_squared + Wk', 
                    data = test_split, 
                    return_type='dataframe')

X['Date'] = test_split['Date']
test_split = dict(tuple(X.groupby(['Store', 'Dept'])))



keys = list(train_split)

for key in keys:
    X_train = train_split[key]
    X_test = test_split[key]
    

    Y = X_train['Weekly_Sales']
    X_train = X_train.drop(['Weekly_Sales','Store', 'Dept'], axis=1)

    cols_to_drop = X_train.columns[(X_train == 0).all()]
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

## Use this methods for handling missing values
    cols_to_drop = []
    for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
        col_name = X_train.columns[i]
        # Extract the current column and all previous columns
        tmp_Y = X_train.iloc[:, i].values
        tmp_X = X_train.iloc[:, :i].values

        coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
        if np.sum(residuals) < 1e-16:
                cols_to_drop.append(col_name)

    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    

    model = sm.OLS(Y, X_train).fit()
    mycoef = model.params.fillna(0)
    tmp_pred = X_test[['Store', 'Dept', 'Date']]
    X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)

    tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
    
    test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)
        
        

test_pred['Weekly_Pred'].fillna(0, inplace = True)
    
    
test_pred = test.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')
        
test_pred.to_csv('mypred.csv', index=False) 
        





