# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np 
import lightgbm as lgb
from sklearn import linear_model
import gc 
from sklearn.model_selection import train_test_split, cross_val_score
import datetime

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


def test_time_feature():
    train, test = read_csv()
    tmp = get_time_feature(train)
    print(tmp.head())
    print(type(tmp))


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


def test_weekday_feature():
    train,test = read_csv()
    tmp = get_weekday_feature(train)
    print(tmp.head())
    print(type(tmp))


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



def lightgbm_make_submission():
    train, test = read_csv()
    # x_train, x_test, y_train, y_test = make_train_set()
    x_train, y_train = make_train_set(train)
    # df = pd.merge(x_train, y_train, on='TERMINALNO')
    # if PREDICT == False:
    #     df.corr().to_csv("./data/corr.csv")
    # else:
    #     print(df.corr())
    # del df
    # gc.collect()
    x_test, y_test = make_train_set(test)
    y_train = y_train['Y']
    # print("**********************x_train*******************")
    # print(x_train)
    # print("**********************x_train end***************")
    # print("**********************x_test********************")
    # print(x_test.head())
    # print("**********************x_test end****************")
    train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    params = {
        'boosting': 'dart',
        'learning_rate': 0.005,
        'application': 'regression',
        'max_depth': -1,
        'num_leaves': 10,
        'verbosity': -1,
        'feature_fraction': 0.5,
        'feature_fraction_seed': 9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'bagging_seed': 9,
        'min_data_in_leaf': 6,
        'min_sum_hessian_in_leaf': 11,
        # 'metric': 'poisson',
        # 'min_data': 1,
        # 'min_data_in_bin': 1,
        # 'poisson_max_delta_step': 7,
        # 'reg_sqrt': True,
        'metric': 'rmse',
        # 'metric': ['rmse', 'poisson'],
    }
    d_train = lgb.Dataset(train_x, label=train_y)
    d_valid = lgb.Dataset(valid_x, label=valid_y)
    watchlist = [d_train, d_valid]
    model = lgb.train(params, train_set=d_train, num_boost_round=10000, valid_sets=watchlist,
                      early_stopping_rounds=50, verbose_eval=100)
    # if PREDICT:
    print("*******************************start predict***************************")
    preds = model.predict(x_test)
    preds[preds < 0] = 0
    # preds = np.sqrt(preds)
    # preds = np.log1p(preds)
    # preds = np.expm1(preds)
    # preds = unsigmoid(preds)
    # print(preds)
    y_test['Y'] = preds
    print(y_test['Y'].var())
    y_test.columns = ['TERMINALNO', 'Pred']
    y_test.set_index('TERMINALNO', inplace=True)
    x_test = pd.merge(x_test, y_test, left_index=True, right_index=True)
    # x_test.set_index('TERMINALNO', inplace=True)
    # print(x_test.head())
    x_test.to_csv(path_test_out, columns=['Pred'], index=True, index_label=['Id'])


def ridge_make_submission():
    train, test = read_csv()
    x_train, y_train = make_train_set(train)
    x_test, y_test = make_train_set(test)
    y_train = y_train['Y']
    model = linear_model.Ridge()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    preds[preds < 0] = 0
    y_test['Y'] = preds
    print(y_test['Y'].var())
    y_test.columns = ['TERMINALNO', 'Pred']
    y_test.set_index('TERMINALNO', inplace=True)
    x_test = pd.merge(x_test, y_test, left_index=True, right_index=True)
    # x_test.set_index('TERMINALNO', inplace=True)
    # print(x_test.head())
    x_test.to_csv(path_test_out, columns=['Pred'], index=True, index_label=['Id'])


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    import numpy as np

    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], np.random.rand()]) # 随机值
                
                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    # process()
    # test_speed_feature()
    # test_direction_feature()
    # test_Y()
    # test_make_train_set()
    lightgbm_make_submission()
    # test_call_state_feature()
    # ridge_make_submission()