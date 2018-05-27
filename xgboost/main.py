# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
# import lightgbm as lgb
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import gc 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_regression
from math import radians, cos, sin, asin, sqrt

if os.path.exists("./data/PINGAN-2018-train_demo.csv"):
    path_train = "./data/PINGAN-2018-train_demo.csv"
    path_test = path_train
    path_test_out = "./data/out.csv"
    PREDICT = False
else:
    PREDICT = True
    path_train = "/data/dm/train.csv"  # 训练文件
    path_test = "/data/dm/test.csv"  # 测试文件
    path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。
    path_test_out = os.path.join(path_test_out, "test.csv")


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # print("*****************read_csv*******************")
    # for filename in os.listdir(path_train):
    train = pd.read_csv(path_train)
    # train = train[train['SPEED'] > 5]
    # train = train[(True ^ train['Y'].isin([0]))]
    nrow_train = train.shape[0]
    # test = pd.DataFrame()
    # test = pd.read_csv(path_train)
    # if PREDICT:
    test = pd.read_csv(path_test)
    # test = test[test['SPEED'] > 5]
    train = pd.concat([train, test], 0)
    # print(train)
    # train.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        # "CALLSTATE", "Y"]
    test = train[nrow_train:]
    train = train[:nrow_train]
    # print("****************read_csv done***************")
    return train, test


def get_speed_feature(trainset):
    """
    速度处理：最大最小平均
    :param trainset:
    :return:
    """
    # print("*************get speed feature**************")
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)

    maxspeed = groupby_userid['SPEED'].max()
    maxspeed.columns = ["TERMINALNO", "SPEED_max"]

    meanspeed = groupby_userid['SPEED'].mean()
    meanspeed.columns = ["TERMINALNO", "SPEED_mean"]

    speed_feature = pd.merge(maxspeed, meanspeed, on='TERMINALNO')
    # print("*************get speed feature done**********")
    return speed_feature


def get_direction_feature(trainset):
    """
    方向处理：方差
    :param trainset:
    :return:
    """
    # print("**************get direction feature**********")
    groupby_userid_tripid = trainset.groupby(['TERMINALNO', 'TRIP_ID'], as_index=False)
    max_var = groupby_userid_tripid['DIRECTION'].var().fillna(0).groupby('TERMINALNO', as_index=False)['DIRECTION'].max()
    max_var.columns = ['TERMINALNO', 'DIRECTION_var_max']
    mean_var = groupby_userid_tripid['DIRECTION'].var().fillna(0).groupby('TERMINALNO', as_index=False)['DIRECTION'].mean()
    mean_var.columns = ['TERMINALNO', 'DIRECTION_var_mean']
    direction_feature = pd.merge(max_var, mean_var, on='TERMINALNO')
    # print("**************get direction feature done**********")
    return direction_feature


def get_height_feature(trainset):
    """
    高度处理
    :param trainset:
    :return:
    """
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)

    max_height = groupby_userid['HEIGHT'].max()
    max_height.columns = ["TERMINALNO", "HEIGHT_max"]
    min_height = groupby_userid['HEIGHT'].min()
    min_height.columns = ['TERMINALNO', 'HEIGHT_min']
    mean_height = groupby_userid['HEIGHT'].mean()
    mean_height.columns = ["TERMINALNO", "HEIGHT_mean"]

    groupby_userid_tripid = trainset.groupby(['TERMINALNO', 'TRIP_ID'], as_index=False)
    max_var = groupby_userid_tripid['HEIGHT'].var().fillna(0).groupby('TERMINALNO', as_index=False)[
        'HEIGHT'].max()
    max_var.columns = ['TERMINALNO', 'HEIGHT_var_max']
    mean_var = groupby_userid_tripid['HEIGHT'].var().fillna(0).groupby('TERMINALNO', as_index=False)[
        'HEIGHT'].mean()
    mean_var.columns = ['TERMINALNO', 'HEIGHT_var_mean']

    height_feature = mean_height
    # height_feature = pd.merge(height_feature, max_height, on='TERMINALNO')
    # height_feature = pd.merge(height_feature, max_var, on='TERMINALNO')
    # height_feature = pd.merge(height_feature, min_height, on='TERMINALNO')
    height_feature = pd.merge(height_feature, mean_var, on='TERMINALNO')
    # print(height_feature.head())
    return height_feature


def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


def get_gps_feature(trainset):
    trainset['hdis'] = trainset.apply(lambda x: haversine1(x['LONGITUDE'], x['LATITUDE'], 113.9177317,22.54334333), axis=1)    
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)
    max_height = groupby_userid['HEIGHT'].max()
    min_height = groupby_userid['HEIGHT'].min()
    max_longitude = groupby_userid['LONGITUDE'].max()
    min_longitude = groupby_userid['LONGITUDE'].min()
    max_latitude = groupby_userid['LATITUDE'].max()
    min_latitude = groupby_userid['LATITUDE'].min()
    mean_longitude = groupby_userid['LONGITUDE'].mean()
    mean_latitude = groupby_userid['LATITUDE'].mean()
    dis_mean = groupby_userid['hdis'].mean()
    dis_min = groupby_userid['hdis'].min()
    dis_max = groupby_userid['hdis'].max()
    gps_feature = pd.merge(max_height, min_height, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, max_longitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, min_longitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, max_latitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, min_latitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, mean_latitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, mean_longitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, dis_mean, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, dis_min, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, dis_max, on='TERMINALNO')
    return gps_feature


def get_trips_geature(trainset):
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)
    trip_num = groupby_userid['TRIP_ID'].max()
    return trip_num


def get_call_state_feature(trainset):
    """
    电话状态特征
    :param trainset:
    :return:
    """
    groupby_userid_tripid = trainset.groupby(['TERMINALNO'], as_index=False)
    count = groupby_userid_tripid['CALLSTATE'].agg({
        'count0': lambda x: list(x).count(0) / len(x),
        'count1': lambda x: list(x).count(1) / (len(x)),
        'count2': lambda x: list(x).count(2) / (len(x)),
        'count3': lambda x: list(x).count(3) / (len(x)),
        'count4': lambda x: list(x).count(4) / len(x)
    })
    return count


def get_time_feature(trainset):
    trainset['TIME1'] = trainset['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)
    time_feature = groupby_userid['TIME1'].agg({
        'hour0':lambda x: list(x).count(0) / len(x),
        'hour1':lambda x: list(x).count(1) / len(x),
        'hour2':lambda x: list(x).count(2) / len(x),
        'hour3':lambda x: list(x).count(3) / len(x),
        'hour4':lambda x: list(x).count(4) / len(x),
        'hour5':lambda x: list(x).count(5) / len(x),
        'hour6':lambda x: list(x).count(6) / len(x),
        'hour7':lambda x: list(x).count(7) / len(x),
        'hour8':lambda x: list(x).count(8) / len(x),
        'hour9':lambda x: list(x).count(9) / len(x),
        'hour10':lambda x: list(x).count(10) / len(x),
        'hour11':lambda x: list(x).count(11) / len(x),
        'hour12':lambda x: list(x).count(12) / len(x),
        'hour13':lambda x: list(x).count(13) / len(x),
        'hour14':lambda x: list(x).count(14) / len(x),
        'hour15':lambda x: list(x).count(15) / len(x),
        'hour16':lambda x: list(x).count(16) / len(x),
        'hour17':lambda x: list(x).count(17) / len(x),
        'hour18':lambda x: list(x).count(18) / len(x),
        'hour19':lambda x: list(x).count(19) / len(x),
        'hour20':lambda x: list(x).count(20) / len(x),
        'hour21':lambda x: list(x).count(21) / len(x),
        'hour22':lambda x: list(x).count(22) / len(x),
        'hour23':lambda x: list(x).count(23) / len(x),
    })
    return time_feature


def get_weekday_feature(trainset):
    trainset['TIME'] = trainset['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
    groupby_userid_tripid = trainset.groupby(['TERMINALNO', 'TRIP_ID'], as_index=False)
    groupby_userid = groupby_userid_tripid['TIME'].max().groupby(['TERMINALNO'], as_index=False)
    week_feature = groupby_userid['TIME'].agg({
        'mon':lambda x: list(x).count(0) / len(x),
        'tue':lambda x: list(x).count(1) / len(x),
        'wed':lambda x: list(x).count(2) / len(x),
        'thu':lambda x: list(x).count(3) / len(x),
        'fri':lambda x: list(x).count(4) / len(x),
        'sat':lambda x: list(x).count(5) / len(x),
        'sun':lambda x: list(x).count(6) / len(x),
    })
    return week_feature


def get_Y(trainset):
    """
    提取Y值
    :param trainset:
    :return:
    """
    Y = trainset.groupby('TERMINALNO', as_index=False)['Y'].max()
    Y.columns = ['TERMINALNO', 'Y']
    return Y


def make_train_set(trainset):
    # print("**************make set*******************")
    speed = get_speed_feature(trainset)
    direction = get_direction_feature(trainset)
    call = get_call_state_feature(trainset)
    height = get_height_feature(trainset)
    time_feat = get_time_feature(trainset)
    gps_feat = get_gps_feature(trainset)
    trip_feat = get_trips_geature(trainset)
    weekday_feat = get_weekday_feature(trainset)
    y = get_Y(trainset)
    x = speed
    x = pd.merge(x, direction, on='TERMINALNO')
    x = pd.merge(x, call, on='TERMINALNO')
    x = pd.merge(x, height, on='TERMINALNO')
    x = pd.merge(x, time_feat, on='TERMINALNO')
    x = pd.merge(x, gps_feat, on='TERMINALNO')
    x = pd.merge(x, trip_feat, on='TERMINALNO')
    x = pd.merge(x, weekday_feat, on='TERMINALNO')
    x.set_index('TERMINALNO', inplace=True)
    # print("**************make set done**************")
    return x, y


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
        'subsample': 1,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,
        'alpha': 1,
        'lambda': 2,
        'eta': 0.1,
        'silent': 1,
        'objective': 'reg:linear',
        "seed":1123,        
    }
    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    model = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=10, verbose_eval=10)
    x_test = xgb.DMatrix(x_test)
    preds = model.predict(x_test)
    preds[preds < 0] = 0
    return preds


def xgb_sklearn_model(x_train, y_train, x_test):
    model = xgb.XGBRegressor()
    params_grid = {
        "n_estimators": [50, 200, 400, 600, 800], 
        "learning_rate": [0.01], 
        "max_depth": [3],
        "min_child_weight": [1, 5, 10, 20],
        "subsample": [1.0],
        "colsample_bytree": [0.9],
        "scale_pos_weight": [1, 0.9, 1.1],
    }
    grid_search = GridSearchCV(estimator=model, 
        param_grid=params_grid, 
        cv=3, 
        scoring="explained_variance",
        n_jobs=2)
    grid_search.fit(X=x_train, y=y_train)
    print(grid_search.best_params_)


def xgb_sklearn_submission_model(x_train, y_train, x_test):
    model = xgb.XGBRegressor(max_depth=3, learning_rate=0.01, n_estimators=200,
        min_child_weight=5, subsample=1, colsample_bytree=0.9, scale_pos_weight=2.0)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return preds


def test_xgb_sklearn_model():
    train, test = read_csv()
    x_train, y_train = make_train_set(train)
    y0_size = y_train[y_train['Y'] == 0].shape[0]
    y1_size = y_train[y_train['Y'] > 0].shape[0]
    print("{0: f} \t {1: f}".format(y0_size, y1_size))
    global SCALE_POS_WEIGHT
    SCALE_POS_WEIGHT = y0_size / y1_size
    x_test, y_test = make_train_set(test)
    y_train = y_train['Y']
    # feature selection
    sel = SelectPercentile(f_regression, 50)
    x_train = sel.fit_transform(x_train, y_train)
    x_test = sel.transform(x_test)
    xgb_sklearn_model(x_train, y_train, x_test)


def layer1_xgb(train_x, test_x, train_y, test_y, test):
    param = {
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 1,
        'subsample': 1,
        'colsample_bytree': 1,
        'scale_pos_weight': 0.9,
        'max_delta_step': 0,
        'lambda': 1,
        'eta': 0.05,
        'silent': 1,
        'objective': 'reg:linear',
        "seed":1123,
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


def five_fold_stacking(x_train, y_train, x_test):
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


SCALE_POS_WEIGHT = 0.0


def make_submissin():
    train, test = read_csv()
    x_train, y_train = make_train_set(train)
    y0_size = y_train[y_train['Y'] == 0].shape[0]
    y1_size = y_train[y_train['Y'] > 0].shape[0]
    print("{0: f} \t {1: f}".format(y0_size, y1_size))
    global SCALE_POS_WEIGHT
    SCALE_POS_WEIGHT = y0_size / y1_size
    x_test, y_test = make_train_set(test)
    y_train = y_train['Y']
    # feature selection
    sel = SelectPercentile(f_regression, 40)
    x_train = sel.fit_transform(x_train, y_train)
    x_test = sel.transform(x_test)
    preds = xgb_sklearn_submission_model(x_train, y_train, x_test)
    
    if not PREDICT:
        tmp = pd.DataFrame()
        tmp['rawY'] = y_train
    # preds1 = xgboost_model(x_train, y_train, x_train)
    # if not PREDICT:
    #     tmp['preds1'] = preds1
    # # y_train = (y_train + preds1) / 2
    # preds2 = xgboost_model(x_train, y_train, x_train)
    # if not PREDICT:
    #     tmp['preds2'] = preds2
    # y_train = (preds1 + preds2) / 2
    
    # tmp['newY'] = y_train
    # SCALE_POS_WEIGHT = 1
    # preds, layer1_preds = five_fold_stacking(x_train, y_train, x_test)
    # y_train = 0.3 * preds + 0.7 * y_train
    # if not PREDICT:
        # tmp['layer1_combine'] = preds
        # tmp['newY'] = y_train
    # preds = xgboost_model(x_train, y_train, x_test)
    # if not PREDICT:
        # tmp['layer1_preds'] = layer1_preds
    # preds = 0.7*preds + 0.3*layer1_preds
    # preds = layer1_preds
    if not PREDICT:
        print("out tmp ")
        tmp['pred'] = preds
        tmp.to_csv('./data/tmp.csv')
    y_test['Y'] = preds
    print(y_test['Y'].var())
    y_test.columns = ['TERMINALNO', 'Pred']
    y_test.set_index('TERMINALNO', inplace=True)
    # x_test = pd.merge(x_test, y_test, left_index=True, right_index=True)
    # x_test.set_index('TERMINALNO', inplace=True)
    # print(x_test.head())
    y_test.to_csv(path_test_out, columns=['Pred'], index=True, index_label=['Id'])


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    make_submissin()
    # test_time_feature()
    # test_weekday_feature()
    # test_xgb_sklearn_model()
