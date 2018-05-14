import os
from feature import make_train_set
from model import five_fold_stacking, xgboost_model
import pandas as pd 
import numpy as np 


SCALE_POS_WEIGHT = 0.0


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
    if not PREDICT:
        tmp = pd.DataFrame()
        tmp['rawY'] = y_train
    # preds1 = xgboost_model(x_train, y_train, x_train)
    # tmp['preds1'] = preds1
    # y_train = (y_train + preds1) / 2
    # preds2 = xgboost_model(x_train, y_train, x_train)
    # tmp['preds2'] = preds2
    # y_train = (preds1 + preds2) / 2
    # tmp['newY'] = y_train
    # SCALE_POS_WEIGHT = 1
    preds, layer1_preds = five_fold_stacking(x_train, y_train, x_test, PREDICT)
    y_train = 0.3 * preds + 0.7 * y_train
    if not PREDICT:
        tmp['layer1_combine'] = preds
        tmp['newY'] = y_train
    preds = xgboost_model(x_train, y_train, x_test)
    if not PREDICT:
        tmp['layer1_preds'] = layer1_preds
    preds = (preds + layer1_preds) / 2
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
    # test_time_feature()