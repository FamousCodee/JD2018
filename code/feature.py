import os
import pandas as pd
import numpy as np
import datetime
import gc 


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


def get_gps_feature(trainset):
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)
    max_height = groupby_userid['HEIGHT'].max()
    min_height = groupby_userid['HEIGHT'].min()
    max_longitude = groupby_userid['LONGITUDE'].max()
    min_longitude = groupby_userid['LONGITUDE'].min()
    max_latitude = groupby_userid['LATITUDE'].max()
    min_latitude = groupby_userid['LATITUDE'].min()
    gps_feature = pd.merge(max_height, min_height, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, max_longitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, min_longitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, max_latitude, on='TERMINALNO')
    gps_feature = pd.merge(gps_feature, min_latitude, on='TERMINALNO')
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
    trainset['TIME'] = trainset['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)
    time_feature = groupby_userid['TIME'].agg({
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


def test_time_feature():
    train, test = read_csv()
    tmp = get_time_feature(train)
    print(tmp.head())


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
    y = get_Y(trainset)
    x = speed
    x = pd.merge(x, direction, on='TERMINALNO')
    x = pd.merge(x, call, on='TERMINALNO')
    x = pd.merge(x, height, on='TERMINALNO')
    x = pd.merge(x, time_feat, on='TERMINALNO')
    x = pd.merge(x, gps_feat, on='TERMINALNO')
    x = pd.merge(x, trip_feat, on='TERMINALNO')
    x.set_index('TERMINALNO', inplace=True)
    # print("**************make set done**************")
    return x, y
    