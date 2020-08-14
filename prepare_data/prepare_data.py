import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: #and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: #and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df


def aggregated_feature(df_tr_2016, df_tr_2017, df_test_2016, df_test_2017, gpfeat, tgfeat):
    
    if type(gpfeat) is list:
        gpfeat_name = '_'.join(gpfeat)
    else:
        gpfeat_name = gpfeat
    
    df_test_2016_group = df_test_2016.groupby(gpfeat).agg({tgfeat:['mean']})
    df_test_2016_group.columns = [f'{gpfeat_name}_{tgfeat}_mean']
    df_test_2016_group.reset_index(inplace = True)

    df_tr_2016 = pd.merge(df_tr_2016, df_test_2016_group, how = 'left', on=gpfeat)
    df_test_2016 = pd.merge(df_test_2016, df_test_2016_group, how = 'left', on=gpfeat)
    
    df_tr_2016[f'{gpfeat_name}_{tgfeat}_dif'] = df_tr_2016[f'{gpfeat_name}_{tgfeat}_mean'] - df_tr_2016[tgfeat]
    df_test_2016[f'{gpfeat_name}_{tgfeat}_dif'] = df_test_2016[f'{gpfeat_name}_{tgfeat}_mean'] - df_test_2016[tgfeat]

    df_tr_2016[f'{gpfeat_name}_{tgfeat}_ratio'] = df_tr_2016[f'{gpfeat_name}_{tgfeat}_dif']/df_tr_2016[f'{gpfeat_name}_{tgfeat}_mean']
    df_test_2016[f'{gpfeat_name}_{tgfeat}_ratio'] = df_test_2016[f'{gpfeat_name}_{tgfeat}_dif']/df_test_2016[f'{gpfeat_name}_{tgfeat}_mean']

    df_test_2017_group = df_test_2017.groupby(gpfeat).agg({tgfeat:['mean']})
    df_test_2017_group.columns = [f'{gpfeat_name}_{tgfeat}_mean']
    df_test_2017_group.reset_index(inplace = True)

    df_tr_2017 = pd.merge(df_tr_2017, df_test_2017_group, how = 'left', on=gpfeat)
    df_test_2017 = pd.merge(df_test_2017, df_test_2017_group, how = 'left', on=gpfeat)
    
    df_tr_2017[f'{gpfeat_name}_{tgfeat}_dif'] = df_tr_2017[f'{gpfeat_name}_{tgfeat}_mean'] - df_tr_2017[tgfeat]
    df_test_2017[f'{gpfeat_name}_{tgfeat}_dif'] = df_test_2017[f'{gpfeat_name}_{tgfeat}_mean'] - df_test_2017[tgfeat]

    df_tr_2017[f'{gpfeat_name}_{tgfeat}_ratio'] = df_tr_2017[f'{gpfeat_name}_{tgfeat}_dif']/df_tr_2017[f'{gpfeat_name}_{tgfeat}_mean']
    df_test_2017[f'{gpfeat_name}_{tgfeat}_ratio'] = df_test_2017[f'{gpfeat_name}_{tgfeat}_dif']/df_test_2017[f'{gpfeat_name}_{tgfeat}_mean']
    
    return df_tr_2016, df_tr_2017, df_test_2016, df_test_2017


# Use KFold target encoding to try avoiding data leakage
def K_Fold_target_encoding(df_train, df_test_1, df_test_2, GCol, target, folds):
    
    if type(GCol) is list:
        GCol_use = '_'.join(GCol)
        GCol_use = '_'.join(GCol)
        df_train[GCol_use] = ''
        df_test_1[GCol_use] = ''
        df_test_2[GCol_use] = ''
        for col in GCol:
            df_train[GCol_use] = df_train[GCol_use] + df_train[col].astype(str) + '_'
            df_test_1[GCol_use] = df_test_1[GCol_use] + df_test_1[col].astype(str) + '_'
            df_test_2[GCol_use] = df_test_2[GCol_use] + df_test_2[col].astype(str) + '_'
        df_train[GCol_use]=df_train[GCol_use].str.strip('_')
        df_test_1[GCol_use]=df_test_1[GCol_use].str.strip('_')
        df_test_2[GCol_use]=df_test_2[GCol_use].str.strip('_')
        
    else:
        GCol_use = GCol
        
    colName = f'{GCol_use}_{target}_enc'
    mean_of_target = df_train[target].mean()
    
    df_train[colName] = np.nan
    
    for tr_ind, val_ind in folds.split(df_train):
        X_tr, X_val = df_train.iloc[tr_ind], df_train.iloc[val_ind]
        df_train.loc[df_train.index[val_ind], colName] = X_val[GCol_use].map(X_tr.groupby(GCol_use)[target].mean())
    
    group_mean = df_train.groupby(GCol_use).agg({target: ['mean']})
    group_mean.columns = [colName]
    group_mean.reset_index(inplace=True)
    
    df_test_1 = pd.merge(df_test_1, group_mean, how = 'left', on = GCol_use)
    df_test_2 = pd.merge(df_test_2, group_mean, how = 'left', on = GCol_use)
    
    del group_mean
    gc.collect()
    
    df_test_1[colName].fillna(mean_of_target, inplace = True)
    df_test_2[colName].fillna(mean_of_target, inplace = True)
    
    if type(GCol) is list:
        df_train.drop(GCol_use, axis=1, inplace=True)
        df_test_1.drop(GCol_use, axis=1, inplace=True)
        df_test_2.drop(GCol_use, axis=1, inplace=True)
        
    return df_train, df_test_1, df_test_2


# Prepare data sets for train and test. Will combine 2016 and 2017 data together for training purpose and keep test data for 2016 and 2017 separate
def prepare_data(path):
    
    properties_2016 = pd.read_csv(f'{path}/properties_2016.csv')
    properties_2017 = pd.read_csv(f'{path}/properties_2017.csv')
    train_2016 = pd.read_csv(f'{path}/train_2016_v2.csv')
    train_2017 = pd.read_csv(f'{path}/train_2017.csv')
    test_2016 = pd.read_csv(f'{path}/sample_submission.csv')
    test_2017 = test_2016.copy()
    
    properties_2016 = properties_2016.sort_values(by=['parcelid'])
    properties_2017 = properties_2017.sort_values(by=['parcelid'])
    
    # Merge properties_2016 and properties_2017 together
    all_features = properties_2016.columns.values
    properties_2017.columns = [str(feature)+'_2017' for feature in properties_2016.columns]
    properties_2017.rename({'parcelid_2017':'parcelid'},axis = 1,inplace=True)
    properties_2016 = pd.merge(properties_2016, properties_2017, how='left', on='parcelid')
    
    features_exclude = ['parcelid','unitcnt_2017','structuretaxvaluedollarcnt_2017','taxvaluedollarcnt_2017','landtaxvaluedollarcnt_2017']
    cat_features = [feature for feature in properties_2017.columns if ('id' in feature)|('cnt' in feature)|('nbr' in feature)]
    cat_features = [feature.replace('_2017','') for feature in cat_features if feature not in features_exclude]
    
    for feature in cat_features:
        if feature == 'parcelid':
            continue
        properties_2016.loc[(properties_2016[feature].isnull())&(properties_2016[f'{feature}_2017'].notnull()),feature]=        properties_2016.loc[(properties_2016[feature].isnull())&(properties_2016[f'{feature}_2017'].notnull()),f'{feature}_2017']

        properties_2016.loc[(properties_2016[feature].notnull())&(properties_2016[f'{feature}_2017'].isnull()),f'{feature}_2017']=        properties_2016.loc[(properties_2016[feature].notnull())&(properties_2016[f'{feature}_2017'].isnull()),feature]

        na_value = min(properties_2016[feature].min(),properties_2016[f'{feature}_2017'].min()) - 1
        properties_2016.fillna({feature: na_value,
                      f'{feature}_2017': na_value},
                              inplace = True)
        
    features = ['regionidcity','regionidzip','regionidneighborhood','propertycountylandusecode']
    for feature in features:

        group = properties_2016.groupby(feature).agg({feature:'count'})
        group.columns = ['count']
        group.reset_index(inplace=True)

        threshould = 100
        lst = group.loc[group['count']<threshould,feature].unique()
        if len(lst) == 0:
            continue
        value_replace = lst.min()
        properties_2016.loc[properties_2016[feature].isin(lst),feature] = value_replace
        properties_2016.loc[properties_2016[f'{feature}_2017'].isin(lst),f'{feature}_2017'] = value_replace
        
    train_2016.rename(columns={'parcelid': 'ParcelId'}, inplace=True)
    train_2017.rename(columns={'parcelid': 'ParcelId'}, inplace=True)
    properties_2016.rename(columns={'parcelid': 'ParcelId'}, inplace=True)
    
    features_2016 = properties_2016.columns.values[:58]
    features_2017 = np.append(properties_2016.columns.values[58:],['ParcelId'])
    
    train_2016 = pd.merge(train_2016, properties_2016[features_2016], how = 'left', on=['ParcelId'])
    train_2017 = pd.merge(train_2017, properties_2016[features_2017], how = 'left', on=['ParcelId'])
    
    test_2016 = pd.merge(test_2016, properties_2016[features_2016], how='left', on=['ParcelId'])
    test_2017 = pd.merge(test_2017, properties_2016[features_2017], how='left', on=['ParcelId'])
    
    train_2016['date'] = pd.to_datetime(train_2016['transactiondate'])
    train_2016['year'] = train_2016['date'].apply(lambda x: x.year)
    train_2016['month'] = train_2016['date'].apply(lambda x: x.month)
    train_2016['day'] = train_2016['date'].apply(lambda x: x.day)
    train_2016['month_block'] = (train_2016['year'] - 2016)*12 + train_2016['month']

    train_2016.drop(['transactiondate', 'date', 
                     'day'],axis=1, inplace = True)
    
    train_2017['date'] = pd.to_datetime(train_2017['transactiondate'])
    train_2017['year'] = train_2017['date'].apply(lambda x: x.year)
    train_2017['month'] = train_2017['date'].apply(lambda x: x.month)
    train_2017['day'] = train_2017['date'].apply(lambda x: x.day)
    train_2017['month_block'] = (train_2017['year'] - 2016)*12 + train_2017['month']

    train_2017.drop(['transactiondate', 'date', 
                     'day'],axis=1, inplace = True)
    
    train_2017.columns = [feature.replace('_2017','') if '_2017' in feature else feature for feature in train_2017.columns]
    test_2017.columns = [feature.replace('_2017','') if '_2017' in feature else feature for feature in test_2017.columns]
    
    features = ['bathroomcnt','calculatedbathnbr']
    for feature in features:
        train_2016[feature] = train_2016[feature]*10
        train_2017[feature] = train_2017[feature]*10
        test_2016[feature] = test_2016[feature]*10
        test_2017[feature] = test_2017[feature]*10
        
    for column in test_2016.columns:
        if test_2016[column].dtypes == 'object':
            
            test_2016[column] = test_2016[column].astype(str)
            test_2017[column] = test_2017[column].astype(str)
            train_2016[column] = train_2016[column].astype(str)
            train_2017[column] = train_2017[column].astype(str)

            lb = LabelEncoder()

            lb = lb.fit(list(set(test_2017[column].unique()).union(set(test_2016[column]))))
            test_2017[column] = lb.transform(test_2017[column])
            test_2016[column] = lb.transform(test_2016[column])
            train_2016[column] = lb.transform(train_2016[column])
            train_2017[column] = lb.transform(train_2017[column])
            
    # Use log to process some highly skewed features
    columns = ['calculatedfinishedsquarefeet', 'finishedsquarefeet50', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
              'taxamount', 'landtaxvaluedollarcnt', 'lotsizesquarefeet', 'finishedsquarefeet6', 'garagetotalsqft',
              'finishedsquarefeet12', 'unitcnt', 'finishedsquarefeet15', 'finishedfloor1squarefeet', 'longitude']
    for column in columns:
        if column not in train_2016.columns:
            continue
        # +0.1 to avoid log0
        train_2016[f'{column}_log'] = np.log(train_2016[column]+0.1)
        train_2017[f'{column}_log'] = np.log(train_2017[column]+0.1)
        test_2016[f'{column}_log'] = np.log(test_2016[column]+0.1)
        test_2017[f'{column}_log'] = np.log(test_2017[column]+0.1)

    train_2016.drop(columns, axis=1, inplace = True)
    train_2017.drop(columns, axis=1, inplace = True)
    test_2016.drop(columns, axis=1, inplace = True)
    test_2017.drop(columns, axis=1, inplace = True)
    
    del properties_2016
    del properties_2017
    
    train_2016 = reduce_mem_usage(train_2016)
    train_2017 = reduce_mem_usage(train_2017)
    test_2016 = reduce_mem_usage(test_2016)
    test_2017 = reduce_mem_usage(test_2017)

    gc.collect()
    
    # Aggregated features
    cat_columns_1 = ['regionidcounty', 'regionidcity', 'regionidneighborhood', 'regionidzip']
    cat_columns_2 = ['propertycountylandusecode','propertyzoningdesc']
    num_columns = ['taxvaluedollarcnt_log', 'lotsizesquarefeet_log', 'finishedsquarefeet12_log', 'landtaxvaluedollarcnt_log',
                  'yearbuilt', 'taxamount_log', 'structuretaxvaluedollarcnt_log', 'calculatedfinishedsquarefeet_log']
    
    for col_cat in cat_columns_1:
        for col_num in num_columns:
            train_2016, train_2017, test_2016, test_2017 =             aggregated_feature(train_2016, train_2017, test_2016, test_2017, col_cat, col_num)
            
    for col_cat_1 in cat_columns_1:
        for col_cat_2 in cat_columns_2: 
            for col_num in num_columns:
                train_2016, train_2017, test_2016, test_2017 =                 aggregated_feature(train_2016, train_2017, test_2016, test_2017, [col_cat_1, col_cat_2], col_num)
                
    train_2016['num_of_nan'] = train_2016.isnull().sum(axis=1)
    train_2017['num_of_nan'] = train_2017.isnull().sum(axis=1)

    test_2016['num_of_nan'] = test_2016.isnull().sum(axis=1)
    test_2017['num_of_nan'] = test_2017.isnull().sum(axis=1)
    
    # Combine 2016 and 2017 data together
    train = train_2016.append(train_2017, ignore_index=True)
    del train_2016, train_2017
    gc.collect()
    
    # 5 fold target encoding
    n_splits = 5
    folds = KFold(n_splits)
    
    train, test_2016, test_2017 = K_Fold_target_encoding(train, test_2016, test_2017, ['regionidcity','propertycountylandusecode'], 'logerror', folds)
    train, test_2016, test_2017 = K_Fold_target_encoding(train, test_2016, test_2017, ['regionidzip','propertycountylandusecode'], 'logerror', folds)
    train, test_2016, test_2017 = K_Fold_target_encoding(train, test_2016, test_2017, ['regionidneighborhood','propertycountylandusecode'], 'logerror', folds)

    train, test_2016, test_2017 = K_Fold_target_encoding(train, test_2016, test_2017, ['regionidcity','propertyzoningdesc'], 'logerror', folds)
    train, test_2016, test_2017 = K_Fold_target_encoding(train, test_2016, test_2017, ['regionidzip','propertyzoningdesc'], 'logerror', folds)
    train, test_2016, test_2017 = K_Fold_target_encoding(train, test_2016, test_2017, ['regionidneighborhood','propertyzoningdesc'], 'logerror', folds)

    # lag features
    group_user = train.groupby('ParcelId').agg({'logerror':'count'})
    group_user.columns = ['user_history']
    group_user.reset_index(inplace = True)
    
    repeat_sale = group_user.loc[group_user['user_history']>1,'ParcelId'].unique()
    train_repeat = train.loc[train['ParcelId'].isin(repeat_sale)]
    
    sale_history = pd.DataFrame()
    sale_history['ParcelId'] = train_repeat['ParcelId'].unique()

    sale_first = train_repeat[['ParcelId','logerror']].groupby('ParcelId').first()
    sale_first.reset_index(inplace = True)
    sale_history = pd.merge(sale_history, sale_first, on = 'ParcelId', how= 'left')
    sale_history.rename(columns = {'logerror':'logerror_first'},inplace=True)

    sale_second = train_repeat[['ParcelId','logerror']].groupby('ParcelId').nth(1)
    sale_second.reset_index(inplace = True)
    sale_history = pd.merge(sale_history, sale_second, on = 'ParcelId', how= 'left')
    sale_history.rename(columns = {'logerror':'logerror_second'},inplace=True)
    
    train['year_copy'] = train['year']
    train['month_copy'] = train['month']
    train['month_block_copy'] = train['month_block']
    train.drop(['year','month','month_block'], axis=1, inplace=True)

    train.rename(columns= {'year_copy':'year',
                          'month_copy':'month',
                          'month_block_copy':'month_block'},inplace=True)
    
    months = [10, 11, 12]
    for month in months:
        tmp=train.loc[train['month_block']<month]

        sale_last = tmp[['ParcelId',
                         'month_block',
                         'logerror']].groupby('ParcelId').nth(-1)
        sale_last.rename(columns={'logerror':f'prev_logerror_{month}',
                                 'month_block':f'prev_month_{month}'},inplace=True)
        sale_last.reset_index(inplace = True)

        test_2016 = pd.merge(test_2016, sale_last, on = 'ParcelId', how='left')
        test_2016[f'month_interval_{month}'] = month - test_2016[f'prev_month_{month}']
        test_2016[f'has_history_{month}'] = 0
        test_2016.loc[test_2016[f'prev_logerror_{month}'].notnull(),f'has_history_{month}'] = 1

    months = [10, 11, 12]
    for month in months:
        tmp=train.loc[train['month_block']<(month+12)]

        sale_last = tmp[['ParcelId',
                         'month_block',
                         'logerror']].groupby('ParcelId').nth(-1)
        sale_last.rename(columns={'logerror':f'prev_logerror_{month}',
                                 'month_block':f'prev_month_{month}'},inplace=True)
        sale_last.reset_index(inplace = True)

        test_2017 = pd.merge(test_2017, sale_last, on = 'ParcelId', how='left')
        test_2017[f'month_interval_{month}'] = (month + 12) - test_2017[f'prev_month_{month}']
        test_2017[f'has_history_{month}'] = 0
        test_2017.loc[test_2017[f'prev_logerror_{month}'].notnull(),f'has_history_{month}'] = 1
        
    train['prev_month'] = train.groupby('ParcelId')['month_block'].shift()
    train['prev_logerror'] = train.groupby('ParcelId')['logerror'].shift()
    train['month_interval'] = train['month_block'] - train['prev_month']
    train['has_history'] = 0
    train.loc[train['prev_logerror'].notnull(),'has_history'] = 1
    
    return train, test_2016, test_2017


if __name__ == "__main__":
    
    path = '../input'
    
    train, test_2016, test_2017 = prepare_data(path)
    
    with open('../processed/train.pickle', 'wb') as handle:
        pickle.dump(train, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    with open('../processed/test_2016.pickle', 'wb') as handle:
        pickle.dump(test_2016, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    with open('../processed/test_2017.pickle', 'wb') as handle:
        pickle.dump(test_2017, handle, protocol = pickle.HIGHEST_PROTOCOL)