import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

import sklearn.linear_model as skline
import sklearn.ensemble as ske
import sklearn.model_selection as skms
import sklearn.metrics as skmc
import sklearn.pipeline as skp

import feature

class RTLR():
    def __init__(self, X_train, y_train):
        X_train, X_train_lr, y_train, y_train_lr = \
            skms.train_test_split(X_train, y_train, test_size=0.5)
        rt = ske.RandomTreesEmbedding(n_estimators=50)
        lm = skline.LogisticRegression()
        self.model = skp.make_pipeline(rt, lm)
        self.model.fit(X_train, y_train)

    def predict(self, X_test, prob=True):
        if prob:
            return self.model.predict_proba(X_test)[:, 1]
        else:
            return self.model.predict(X_test)

class GBT():
    def __init__(self, X_train, y_train):
        self.model = ske.GradientBoostingClassifier(n_estimators=50)
        self.model.fit(X_train, y_train)

    def predict(self, X_test, prob=True):
        if prob:
            return self.model.predict_proba(X_test)[:, 1]
        else:
            return self.model.predict(X_test)

class RF():
    def __init__(self, X_train, y_train):
        self.model = ske.RandomForestClassifier(max_depth=3, n_estimators=50)
        self.model.fit(X_train, y_train)

    def predict(self, X_test, prob=True):
        if prob:
            return self.model.predict_proba(X_test)[:, 1]
        else:
            return self.model.predict(X_test)

class LGB():
    def __init__(self, X_train, y_train):
        _train = X_train.copy()
        _train['is_trade'] = y_train
        train, valid = skms.train_test_split(_train, test_size=0.5)
        train_data = lgb.Dataset(train.iloc[:, :-1], label=train.iloc[:, -1])
        valid_data = lgb.Dataset(valid.iloc[:, :-1], label=valid.iloc[:, -1])
        _train_data = lgb.Dataset(_train.iloc[:, :-1], label=_train.iloc[:, -1])
        param = {'num_leaves':31, 'num_trees':100, 'objective':'binary', 'train_metric':False}
        offline = lgb.train(
            param, 
            train_data, 
            valid_sets=[valid_data], 
            early_stopping_rounds=10)

        self.model = lgb.train(
            param, 
            _train_data, 
            num_boost_round=offline.best_iteration)

    def predict(self, X_test, prob=True):
        if prob:
            return self.model.predict(X_test)
        else:
            return self.model.predict(X_test)

def load_data(data, dropna=False, drop_features=False):
    '''
    加载数据 过滤数据 定义特征类型
    '''
    if dropna:
        data = data.dropna()
    if drop_features:
        data = data.drop(labels=drop_features, axis=1)
        
    for col in data.columns:
        if 'id' in col:
            data[col] = data[col].astype('category')
        # elif 'hit' in col or 'score' in col:
        #     data[col] = data[col].astype(np.float32)
        else:
            # data[col] = data[col].astype(np.int32)
            data[col] = data[col].astype(np.float32)

    return data

def split(data, by='day'):
    '''
    划分离线数据集 用于超参
    '''
    test_day = data[by].max()
    train, test = data[data[by] < test_day], data[data[by] == test_day]
    return train, test    


if __name__ == '__main__':

    train = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep=' ').dropna()
    feabox = feature.featureBox(train)
    data = feabox.feature_program()
    data = load_data(data, dropna=True, drop_features=['item_property_ids'])
    _train, _test = split(data, by='day') 
    X_train, y_train = _train.iloc[:, :-1], _train.iloc[:, -1]
    X_test, y_test = _test.iloc[:, :-1], _test.iloc[:, -1]

    clfs = [
        RTLR(X_train, y_train),
        GBT(X_train, y_train),
        RF(X_train, y_train),
        LGB(X_train, y_train)
    ]

    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_train[:, j] = clf.predict(X_train)
        dataset_blend_test[:, j] = clf.predict(X_test)
        print(skmc.log_loss(y_test, dataset_blend_test[:, j]))

    for j, clf_name in enumerate(['RTLR', 'GBT', 'RF']):
        X_train[clf_name] = dataset_blend_train[:, j]
        X_test[clf_name] = dataset_blend_test[:, j]
    mylgb = LGB(X_train, y_train)
    y_pre = mylgb.predict(X_test)
    print(skmc.log_loss(y_test, y_pre))
    print(sorted(list(zip(
        mylgb.model.feature_importance(),mylgb.model.feature_name())), 
        key=lambda x: x[0], reverse=True))
