import time
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import feature


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

    test = pd.read_csv('../data/round1_ijcai_18_test_b_20180418.txt', sep=' ')
    instance_id = test['instance_id'].values    
    test['is_trade'] = 2

    # =========================================================
    feabox = feature.featureBox(pd.concat([train, test]))
    # feabox = feature.featureBox(train)
    # =========================================================
    
    data = feabox.feature_program()
    data = load_data(data, dropna=True, drop_features=['item_property_ids'])

    # ==========================================================
    _train, _test = split(data, by='is_trade')
    # _train, _test = split(data, by='day') 
    # ==========================================================    
    # train, valid = split(_train, by='day')   
    train, valid = train_test_split(_train, test_size=0.5)
    # ==========================================================


    train_data = lgb.Dataset(train.iloc[:, :-1], label=train.iloc[:, -1])
    _train_data = lgb.Dataset(_train.iloc[:, :-1], label=_train.iloc[:, -1])
    valid_data = lgb.Dataset(valid.iloc[:, :-1], label=valid.iloc[:, -1])

    param = {'num_leaves':31, 'num_trees':100, 'objective':'binary', 'train_metric':'false'}
    offline = lgb.train(
        param, 
        train_data, 
        valid_sets=[valid_data], 
        early_stopping_rounds=10)

    online = lgb.train(
        param, 
        _train_data, 
        num_boost_round=offline.best_iteration)
        
    y_pre = online.predict(_test.iloc[:, :-1])

#    y_true = _test.iloc[:, -1].values
#    print(log_loss(y_true, y_pre))

    result = pd.DataFrame([instance_id, y_pre]).T
    result.columns = ['instance_id', 'predicted_score']
    result['instance_id'] = result['instance_id'].astype(np.int64)
    
    t = time.ctime()
    ts = t[4:16].replace(' ', '-').replace(':', '-')
    result.to_csv('../data/result%s.txt' % ts, sep=" ", index=False)



