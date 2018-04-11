# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np 
import lightgbm as lgb 
import gc 
from sklearn.model_selection import train_test_split, cross_val_score

if os.path.exists("./data/PINGAN-2018-train_demo.csv"):
    path_train = "./data/PINGAN-2018-train_demo.csv"
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
    # for filename in os.listdir(path_train):
    train = pd.read_csv(path_train)
    nrow_train = train.shape[0]
    test = pd.DataFrame()
    if PREDICT:
        test = pd.read_csv(path_test)
    train = pd.concat([train, test], 0)
    train.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    test = train[nrow_train:]
    train = train[:nrow_train]
    return train, test


def get_speed_feature(trainset):
    """
    速度处理：最大最小平均
    :param trainset:
    :return:
    """
    groupby_userid = trainset.groupby('TERMINALNO', as_index=False)

    maxspeed = groupby_userid['SPEED'].max()
    maxspeed.columns = ["TERMINALNO", "SPEED_max"]

    meanspeed = groupby_userid['SPEED'].mean()
    meanspeed.columns = ["TERMINALNO", "SPEED_mean"]

    speed_feature = pd.merge(maxspeed, meanspeed, on='TERMINALNO')
    return speed_feature


def test_speed_feature():
    train, test = read_csv()
    print(get_speed_feature(train).head())


def get_direction_feature(trainset):
    """
    方向处理：方差
    :param trainset:
    :return:
    """
    groupby_userid_tripid = trainset.groupby(['TERMINALNO', 'TRIP_ID'], as_index=False)
    max_var = groupby_userid_tripid['DIRECTION'].var().fillna(0).groupby('TERMINALNO', as_index=False)['DIRECTION'].max()
    max_var.columns = ['TERMINALNO', 'DIRECTION_var_max']
    mean_var = groupby_userid_tripid['DIRECTION'].var().fillna(0).groupby('TERMINALNO', as_index=False)['DIRECTION'].mean()
    mean_var.columns = ['TERMINALNO', 'DIRECTION_var_mean']
    direction_feature = pd.merge(max_var, mean_var, on='TERMINALNO')
    return direction_feature


def test_direction_feature():
    train, nrow_train = read_csv()
    print(get_direction_feature(train).head())


def get_Y(trainset):
    """
    提取Y值
    :param trainset:
    :return:
    """
    Y = trainset.groupby('TERMINALNO', as_index=False)['Y'].max()
    Y.columns = ['TERMINALNO', 'Y']
    # Y = Y['Y']
    return Y


def test_Y():
    train, nrow_train = read_csv()
    print(get_Y(train).head())


def make_train_set(trainset):
    speed = get_speed_feature(trainset)
    direction = get_direction_feature(trainset)
    y = get_Y(trainset)
    x = pd.merge(speed, direction, on='TERMINALNO')

    return x, y


def test_make_train_set():
    x1, x2, y1, y2 = make_train_set()
    print(x1.head())
    print(x2.head())


def lightgbm_make_submission():
    train, test = read_csv()
    # x_train, x_test, y_train, y_test = make_train_set()
    x_train, y_train = make_train_set(train)
    x_test, y_test = make_train_set(test)
    y_train = y_train['Y']
    # print(x_train.head())
    # print(x_test.head())
    train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
    params = {
        'learning_rate': 0.05,
        'application': 'regression',
        'max_depth': -1,
        'num_leaves': 200,
        'verbosity': -1,
        'metric': 'RMSE',
    }
    d_train = lgb.Dataset(train_x, label=train_y)
    d_valid = lgb.Dataset(valid_x, label=valid_y)
    watchlist = [d_train, d_valid]
    model = lgb.train(params, train_set=d_train, num_boost_round=2200, valid_sets=watchlist,
                      early_stopping_rounds=50, verbose_eval=100)
    if PREDICT:
        print("*******************************start predict***************************")
        preds = model.predict(x_test)
        # print(preds)
        y_test['Y'] = preds
        y_test.columns = ['TERMINALNO', 'Pred']
        x_test = pd.merge(x_test, y_test, on='TERMINALNO')
        # x_test.set_index('TERMINALNO', inplace=True)
        print(x_test.head())
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