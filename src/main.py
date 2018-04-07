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


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train)
    test = pd.DataFrame()
    if PREDICT:
        test = pd.read_csv(path_test)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    return tempdata, test


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
    train, test = read_csv()
    nrow_train = train.shape[0]
    train = pd.concat([train, test], 0)
    # print(df.head())
    y_train = np.log1p(train["Y"])
    X = train.drop('Y',axis=1, inplace=False)
    # del train
    # gc.collect()
    x_train = X[:nrow_train]
    x_test = X[nrow_train:]

    train_X, valid_X, train_Y, valid_Y = train_test_split(x_train, y_train, test_size=0.1)
    d_train = lgb.Dataset(train_X, label=train_Y)
    d_valid = lgb.Dataset(valid_X, label=valid_Y)
    watchlist = [d_train, d_valid]
    params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }

    model = lgb.train(params, 
    train_set=d_train, 
    num_boost_round=2200,
    valid_sets=watchlist,
    early_stopping_rounds=50,
    verbose_eval=100)

if PREDICT:
    preds = model.predict(x_test)
    