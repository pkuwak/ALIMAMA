#coding:utf-8
import pandas as pd
import time

from sklearn.metrics import log_loss
# instance_id 样本编号
# item_id 广告商品编号
# item_category_list 广告商品的的类目列表 分割; item_property_list_0 item_property_list_1 item_property_list_2
# item_property_list 广告商品的属性列表 分割 1 2 3
# item_brand_id 广告商品的品牌编号
# item_city_id 广告商品的城市编号
# item_price_level 广告商品的价格等级
# item_sales_level 广告商品的销量等级
# item_collected_level 广告商品被收藏次数的等级
# item_pv_level 广告商品被展示次数的等级
# user_id 用户的编号
# 'user_gender_id', 用户的预测性别编号
# 'user_age_level', 用户的预测年龄等级
# 'user_occupation_id', 用户的预测职业编号
# 'user_star_level' 用户的星级编号
# context_id 上下文信息的编号
#  context_timestamp 广告商品的展示时间
# context_page_id 广告商品的展示页面编号
# predict_category_property
def time2cov(time_):
    '''
    时间是根据天数推移，所以日期为脱敏，但是时间本身不脱敏
    :param time_: 
    :return: 
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))

def pre_process(data):
    '''
    :param data: 
    :return: 
    '''

    print('预处理')
    print('item_category_list_ing')
    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else " "
        )
    del data['item_category_list']

    print('item_property_list_ing')
    for i in range(3):
        data['property_%d'%(i)] = data['item_property_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else " "
        )
    del data['item_property_list']

    print('context_timestamp_ing')
    data['context_timestamp'] = data['context_timestamp'].apply(time2cov)

    print('time')
    data['context_timestamp_tmp'] = pd.to_datetime(data['context_timestamp'])
    data['week'] = data['context_timestamp_tmp'].dt.weekday
    data['hour'] = data['context_timestamp_tmp'].dt.hour
    del data['context_timestamp_tmp']


    print('predict_category_property_ing_0')
    for i in range(3):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(
            lambda x:str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
        )

    # print('predict_category_property_ing_1')
    # for i in range(3):
    #     data['predict_property_%d'%(i)] = data['predict_category_property'].apply(
    #         lambda x:str(x.split(";")[i]).split(":")[1] if len(x.split(";")) > i else " "
    #     )
    #
    #     for j in range(3):
    #         data['predict_property_%d_%d' % (i,j)] = data['predict_property_%d'%(i)].apply(
    #             lambda x: x.split(",")[j] if len(x.split(",")) > j else -1
    #         )

    del data['predict_category_property']
    # del data['predict_property_1']
    # del data['predict_property_2']

    return data

print('train')
train = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt',sep=" ")

train = pre_process(train)

all_data = train.copy()


print('all_shape',train.shape)
print(train['context_timestamp'].max())
val = train[train['context_timestamp']>'2018-09-22 23:59:59']

# 空去 22 号

train = train[train['context_timestamp']<='2018-09-21 23:59:59']

train = train[train['context_timestamp']>'2018-09-19 23:59:59']
print(train.shape)
print(val.shape)

print('test')
test_a = pd.read_csv('../data/round1_ijcai_18_test_a_20180301.txt',sep=" ")
print(test_a.shape)
test_a = pre_process(test_a)

# 这里是增加的内容
import datetime
def get_count_feat(all_data,data,long=3):
    end_time = data['context_timestamp'].min()
    begin_time = pd.to_datetime(end_time) - datetime.timedelta(days=long)
    all_data['context_timestamp'] = pd.to_datetime(all_data['context_timestamp'])
    all_data = all_data[
        (all_data['context_timestamp']<end_time)&(all_data['context_timestamp']>=begin_time)
                    ]
    print(end_time)
    print(begin_time)
    print(all_data['context_timestamp'].max()-all_data['context_timestamp'].min())
    item_count = all_data.groupby(['item_id'],as_index=False).size().reset_index()
    item_count.rename(columns={0:'item_count'},inplace=True)

    user_count = all_data.groupby(['user_id'], as_index=False).size().reset_index()
    user_count.rename(columns={0: 'user_count'}, inplace=True)
    return user_count,item_count

train_user_count,train_item_count = get_count_feat(all_data,train,2)

test_user_count,test_item_count = get_count_feat(all_data,test_a,2)

val_user_count,val_item_count = get_count_feat(all_data,val,2)

train = pd.merge(train,train_user_count,on=['user_id'],how='left')
train = pd.merge(train,train_item_count,on=['item_id'],how='left')
train = train.fillna(-1)
val = pd.merge(val,val_user_count,on=['user_id'],how='left')
val = pd.merge(val,val_item_count,on=['item_id'],how='left')
val = val.fillna(-1)
test_a = pd.merge(test_a,test_user_count,on=['user_id'],how='left')
test_a = pd.merge(test_a,test_item_count,on=['item_id'],how='left')
test_a = test_a.fillna(-1)
y_train = train.pop('is_trade')
train_index = train.pop('instance_id')

y_val = val.pop('is_trade')
val_index = val.pop('instance_id')
test_index = test_a.pop('instance_id')

print(test_a.shape)
del train['context_timestamp']
del val['context_timestamp']
del test_a['context_timestamp']
del all_data

print('baseline ing ... ...')
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
from sklearn.linear_model import LogisticRegression
print(test_a.columns)

enc = OneHotEncoder()
lb = LabelEncoder()
feat_set = list(test_a.columns)
for i,feat in enumerate(feat_set):
    tmp = lb.fit_transform((list(train[feat])+list(val[feat])+list(test_a[feat])))
    print(tmp)
    enc.fit(tmp.reshape(-1,1))
    x_train = enc.transform(lb.transform(train[feat]).reshape(-1, 1))
    x_test = enc.transform(lb.transform(test_a[feat]).reshape(-1, 1))
    x_val = enc.transform(lb.transform(val[feat]).reshape(-1, 1))
    if i == 0:
        X_train, X_test,X_val = x_train, x_test,x_val
    else:
        X_train, X_test,X_val = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test)),sparse.hstack((X_val, x_val))


lr = LogisticRegression()


lr.fit(X_train, y_train)
proba_val = lr.predict_proba(X_val)[:,1]
proba_sub = lr.predict_proba(X_val)[:,1]

import lightgbm as lgb

gbm = lgb.LGBMRegressor(objective='binary',
                        num_leaves=64,
                        learning_rate=0.01,
                        n_estimators=2000,
                        colsample_bytree = 0.65,
                        subsample = 0.75,
                        # reg_alpha = 0.4



                        )
gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        early_stopping_rounds=150)

print('Start predicting...')
# predict
y_pred_1 = gbm.predict(X_val, num_iteration=gbm.best_iteration)
y_sub_1 = gbm.predict(X_test, num_iteration=gbm.best_iteration)

print(log_loss(y_val,proba_val))
print(log_loss(y_val,y_pred_1))

print(log_loss(y_val,(y_pred_1 + proba_val)/2))


xx_analyse = pd.DataFrame()
xx_analyse['ture'] = list(y_val)
xx_analyse['pre'] = list(proba_val)
xx_analyse['pre_1'] = list(y_pred_1)
xx_analyse.to_csv('../tmp/xx.csv',index=False)


sub = pd.DataFrame()
sub['instance_id'] = list(test_index)

sub['instance_id'] = list(test_index)
sub['predicted_score'] = list(y_sub_1)
sub.to_csv('../result/20180322.txt',sep=" ",index=False)