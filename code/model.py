import numpy as np
import pandas as pd 

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model

def ridge_model(x_train, y_train, x_test):
    model = linear_model.Ridge()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    preds[preds < 0] = 0
    return preds


def xgboost_model(x_train, y_train, x_test):
    train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.2)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(valid_x, label=valid_y)
    param = {
        # 'learning_rate': 0.05,
        # 'n_estimator': 1000,
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'scale_pos_weight': 1,
        'alpha': 1,
        'lambda': 2,
        'eta': 0.05,
        'silent': 1,
        'objective': 'reg:linear',
    }
    num_round = 1000
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    model = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=10, verbose_eval=250)
    x_test = xgb.DMatrix(x_test)
    preds = model.predict(x_test)
    # preds[preds < 0] = 0
    return preds


def layer1_xgb(train_x, test_x, train_y, test_y, test):
    param = {
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 1,
        'subsample': 1,
        'colsample_bytree': 0.6,
        'scale_pos_weight': 1,
        'max_delta_step': 0,
        'lambda': 2,
        'eta': 0.01,
        'silent': 1,
        'objective': 'reg:linear'
    }
    num_round = 1000
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    model = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=10, verbose_eval=250)
    pred = model.predict(dtest)
    test = xgb.DMatrix(test)
    pred_test = model.predict(test)
    pred[pred < 0] = 0
    pred_test[pred_test < 0] = 0
    return pred, pred_test


def five_fold_stacking(x_train, y_train, x_test, PREDICT):
    n = x_train.shape[0] // 5
    preds = np.array([])
    layer1_average_pred_test = np.zeros(x_test.shape[0])
    for i in range(5):
        x1 = x_train[i*n: (i + 1) *n]
        y1 = y_train[i*n: (i + 1) *n]
        if not PREDICT:
            x = x_train.drop(index=list(np.arange(i * n + 1, (i + 1) * n + 1)))
        else:
            x = x_train.drop(index=list(np.arange(i * n, (i + 1) * n)))
        y = y_train.drop(index=np.arange(i*n, (i + 1) *n))
        pred, pred_test = layer1_xgb(train_x=x, test_x=x1, train_y=y, test_y=y1, test=x_test)
        preds = np.append(preds, pred)
        layer1_average_pred_test += pred_test
    layer1_average_pred_test /= 5
    return preds, layer1_average_pred_test