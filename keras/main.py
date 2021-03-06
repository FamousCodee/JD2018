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


def get_acceleration_feature(trainset):
    groupby_userid_tripid = trainset.groupby(['TERMINALNO', 'TRIP_ID'])
    trainset['SPEED_diff'] = groupby_userid_tripid['SPEED'].diff().fillna(0)
    groupby_userid_tripid = trainset.groupby('TERMINALNO', as_index=False)
    max_acc = groupby_userid_tripid['SPEED_diff'].max()
    max_acc.columns = ['TERMINALNO', 'max_acc']
    min_acc = groupby_userid_tripid['SPEED_diff'].min()
    min_acc.columns = ['TERMINALNO', 'min_acc']    
    mean_acc = groupby_userid_tripid['SPEED_diff'].mean()
    mean_acc.columns = ['TERMINALNO', 'mean_acc']
    var_acc = groupby_userid_tripid['SPEED_diff'].var().fillna(0)
    var_acc.columns = ['TERMINALNO', 'var_acc']    
    acc_feature = pd.merge(max_acc, min_acc, on='TERMINALNO')
    acc_feature = pd.merge(acc_feature, mean_acc, on='TERMINALNO')
    acc_feature = pd.merge(acc_feature, var_acc, on='TERMINALNO')
    # print(acc_feature)
    return acc_feature

def test_get_acceleration_feature():
    train, test = read_csv()
    get_acceleration_feature(train)


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
    height_feature = pd.merge(height_feature, max_var, on='TERMINALNO')
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


def test_gps_feature():
    train, test = read_csv()
    gps = get_gps_feature(train)
    print(gps)


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
    acc_feat = get_acceleration_feature(trainset)
    y = get_Y(trainset)
    x = speed
    x = pd.merge(x, direction, on='TERMINALNO')
    x = pd.merge(x, call, on='TERMINALNO')
    x = pd.merge(x, height, on='TERMINALNO')
    x = pd.merge(x, time_feat, on='TERMINALNO')
    x = pd.merge(x, gps_feat, on='TERMINALNO')
    x = pd.merge(x, trip_feat, on='TERMINALNO')
    x = pd.merge(x, weekday_feat, on='TERMINALNO')
    x = pd.merge(x, acc_feat, on='TERMINALNO')
    x.set_index('TERMINALNO', inplace=True)
    # print("**************make set done**************")
    return x, y


def keras_model(x_train, y_train, x_test):
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential([
        Dense(100, input_shape=(x_train.shape[1],)),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(1),
        Activation('linear'),
    ])
    model.compile(optimizer='rmsprop',
        loss='mse')
    model.fit(x_train, y_train, batch_size=32, epochs=10000)
    preds = model.predict(x_test, batch_size=32)
    # print(preds)
    return preds


def make_submissin():
    train, test = read_csv()
    x_train, y_train = make_train_set(train)
    y0_size = y_train[y_train['Y'] == 0].shape[0]
    y1_size = y_train[y_train['Y'] > 0].shape[0]
    print("{0: f} \t {1: f}".format(y0_size, y1_size))
    global SCALE_POS_WEIGHT
    SCALE_POS_WEIGHT = y0_size / y1_size
    x_test, y_test = make_train_set(test)
    y_train = 10 * y_train['Y']
    # feature selection
    sel = SelectPercentile(f_regression, 70)
    x_train = sel.fit_transform(x_train, y_train)
    x_test = sel.transform(x_test)

    preds = keras_model(x_train, y_train, x_test)
    print(preds)
    y_test['Y'] = preds
    print(y_test['Y'].var())
    y_test.columns = ['TERMINALNO', 'Pred']
    y_test.set_index('TERMINALNO', inplace=True)
    y_test.to_csv(path_test_out, columns=['Pred'], index=True, index_label=['Id'])


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    make_submissin()