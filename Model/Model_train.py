import numpy as np
import pandas as pd
import pickle, gc, shap, math, random, time
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import NuSVR, SVR


def train_model_regression_Zillow(X, y, params, groups, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame
    :params: X_test - test data, can be pd.DataFrame
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    models = []
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error},
                    'rmse': {'lgb_metric_name': 'rmse',
                        'catboost_metric_name': 'RMSE',
                        'sklearn_scoring_function': root_mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # list of scores on folds
    scores = []
    train_loss = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    if groups is None:
        splits = folds.split(X)
        print('yes')
    else:
        splits = folds.split(X, groups = groups)
        print('no')
        
    for fold_n, (train_index, valid_index) in enumerate(splits):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred_train = model.predict(X_train)
            models.append(model)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred_train = model.predict(xgb.DMatrix(X_train, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            models.append(model)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred_train = model.predict(X_train)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            train_loss.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_train, y_pred_train))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    print('Train loss mean: {0:.6f}, std: {1:.6f}.'.format(np.mean(train_loss), np.std(train_loss)))
    print('CV mean score: {0:.6f}, std: {1:.6f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['scores'] = scores
    result_dict['models'] = models
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict


def root_mean_squared_error(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def test_predict(df, models, features, year):
    
    months = [10, 11, 12]
    for month in months:
        df['year'] = year
        df['year'] = df['year'].astype('int16')
        df['month'] = month
        df['month'] = df['month'].astype('int8')
        df['month_block'] = month + (year-2016)*12
        df['month_block'] = df['month_block'].astype('int8')
        
        n_models = len(models)
        y_pred =  np.zeros(len(df))
        
        features_old = {feature.replace(f'_{month}',''):feature for feature in df.columns if (f'_{month}' in feature) }
        features_new = {value:key for key, value in features_old.items()}
        df.rename(columns=features_new, inplace = True)
    
        for model in models:
            y_pred += model.predict(df[features])
            gc.collect()
            
        y_pred /= n_models
        df.rename(columns=features_old, inplace = True)

        test_2017[str(year) + str(month)] = y_pred


class train_config:
    
    def __init__(self, n_splits, features, model_type, model_params, eval_metric, early_stopping_rounds, n_estimators,
                seed):
        
        self.n_splits = n_splits
        self.features = features
        self.model_type = model_type
        self.model_params = model_params
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.seed = seed


def model_train(df_train, train_config):
    
    n_splits = train_config.n_splits
    seed = train_config.seed
    folds = KFold(train_config.n_splits, shuffle = True, random_state = train_config.seed)
    
    X_train = train.loc[(train['month_block']>3),train_config.features]
    y_train = train.loc[(train['month_block']>3),'logerror'].clip(-1, 1) 
    
    result_dict = train_model_regression_Zillow(
                         X=X_train, 
                         y=y_train, 
                         params=train_config.params, 
                         groups = None, 
                         folds=folds, model_type=train_config.model_type, eval_metric=train_config.eval_metirc, 
                         plot_feature_importance=True, verbose=500, early_stopping_rounds=train_config.early_stopping_rounds, 
                         n_estimators=train_config.n_estimators)
    
    return result_dict


if __name__ == "__main__":
    
    with open('../processed/train.pickle', 'rb') as handle:
        train = pickle.load(handle)

    with open('../processed/test_2016.pickle', 'rb') as handle:
        test_2016 = pickle.load(handle)
        
    with open('../processed/test_2017.pickle', 'rb') as handle:
        test_2017 = pickle.load(handle
                           
    config_path = sys.argv[1]
    with open(config_path) as json_file:
        config = json.load(json_file)
        
    t_config = train_config(
        n_splits = config['n_splits'], 
        features = config['features'],
        model_type = config['model_type'],
        model_params = config['model_params'], 
        eval_metric = config['eval_metric'],
        early_stopping_rounds = config['early_stopping_rounds'],
        n_estimators = config['n_estimators'],
        seed = config['seed'],
    )
    
    result_dict = model_train(train, t_config)
    
    models = result_dict['models']
                                
    test_predict(test_2016, models, features_0, 2016)
    test_predict(test_2017, models, features_0, 2017)
    
    sample_submission = pd.read_csv('../input/sample_submission.csv')
    columns_submission = test_2017.columns[:7]
    submission = test_2017[columns_submission]
    submission = submission.round(4)
    submission.to_csv(f'./submissions/submission_{t_config.model_type}_{t_config.seed}.csv', index=False)

