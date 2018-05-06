# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
# import lightgbm as lgb
from sklearn import linear_model
import gc 
from sklearn.model_selection import train_test_split, cross_val_score

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
    # train = train[(True ^ train['Y'].isin([0]))]
    nrow_train = train.shape[0]
    test = pd.DataFrame()
    # test = pd.read_csv(path_train)
    # if PREDICT:
    test = pd.read_csv(path_test)
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
    height_feature = pd.merge(height_feature, max_height, on='TERMINALNO')
    # height_feature = pd.merge(height_feature, max_var, on='TERMINALNO')
    # height_feature = pd.merge(height_feature, min_height, on='TERMINALNO')
    # height_feature = pd.merge(height_feature, mean_var, on='TERMINALNO')
    # print(height_feature.head())
    return height_feature


def get_call_state_feature(trainset):
    """
    电话状态特征
    :param trainset:
    :return:
    """
    groupby_userid_tripid = trainset.groupby(['TERMINALNO'], as_index=False)
    count = groupby_userid_tripid['CALLSTATE'].agg({
        # 'count0':lambda x: list(x).count(0) / len(x),
        'count1': lambda x: list(x).count(1) / (len(x) - list(x).count(0) + 1),
        'count2': lambda x: list(x).count(2) / (len(x) - list(x).count(0) + 1),
        'count3': lambda x: list(x).count(3) / (len(x) - list(x).count(0) + 1),
        # 'count4':lambda x: list(x).count(4) / len(x)
    })
    return count


def get_Y(trainset):
    """
    提取Y值
    :param trainset:
    :return:
    """
    # print("******************get Y*******************")
    Y = trainset.groupby('TERMINALNO', as_index=False)['Y'].max()
    Y.columns = ['TERMINALNO', 'Y']
    # Y['Y'] = Y['Y'] ** 2
    # Y['Y'] = np.expm1(Y['Y'])
    # Y['Y'] = np.log1p(Y['Y'])
    # Y['Y'] = sigmoid(Y['Y'])
    # print("***************get Y done*****************")
    return Y


def make_train_set(trainset):
    # print("**************make set*******************")
    speed = get_speed_feature(trainset)
    direction = get_direction_feature(trainset)
    call = get_call_state_feature(trainset)
    height = get_height_feature(trainset)
    y = get_Y(trainset)
    x = speed
    x = pd.merge(x, direction, on='TERMINALNO')
    x = pd.merge(x, call, on='TERMINALNO')
    x = pd.merge(x, height, on='TERMINALNO')
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
    train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(valid_x, label=valid_y)
    param = {
        'learning_rate': 0.05,
        # 'n_estimator': 1000,
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'scale_pos_weight': SCALE_POS_WEIGHT,
        'eta': 0.05,
        'silent': 1,
        'objective': 'reg:linear'
    }
    num_round = 10000
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    model = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=10, verbose_eval=False)
    x_test = xgb.DMatrix(x_test)
    preds = model.predict(x_test)
    preds[preds < 0] = 0.1
    return preds


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
    tmp = pd.DataFrame()
    tmp['rawY'] = y_train
    preds = xgboost_model(x_train, y_train, x_train)
    y_train = preds
    tmp['newY'] = y_train
    # SCALE_POS_WEIGHT = 1
    preds = xgboost_model(x_train, y_train, x_test)
    if not PREDICT:
        print("out tmp ")
        tmp['pred'] = preds
        tmp.to_csv('./data/tmp.csv')
    y_test['Y'] = preds
    print(y_test['Y'].var())
    y_test.columns = ['TERMINALNO', 'Pred']
    y_test.set_index('TERMINALNO', inplace=True)
    x_test = pd.merge(x_test, y_test, left_index=True, right_index=True)
    # x_test.set_index('TERMINALNO', inplace=True)
    # print(x_test.head())
    x_test.to_csv(path_test_out, columns=['Pred'], index=True, index_label=['Id'])


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    make_submissin()
