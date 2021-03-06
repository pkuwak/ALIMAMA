# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:53:58 2018
@author : HaiyanJiang
@email  : jianghaiyan.cn@gmail.com

what does the doc do?
    some ideas of improving the accuracy of imbalanced data classification.
data characteristics:
    imbalanced data.
the models:
    model_baseline : lgb
    model_baseline2 : another lgb
    model_baseline3 : bagging

Other Notes:
除了基本特征外，还包括了'用户'在当前小时内和当天的点击量统计特征，以及当前所在的小时。
'context_day', 'context_hour',
'user_query_day', 'user_query_hour', 'user_query_day_hour',
non_feat = [
        'instance_id', 'user_id', 'context_id', 'item_category_list',
        'item_property_list', 'predict_category_property',
        'context_timestamp', 'TagTime', 'context_day'
        ]

"""

import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
from scipy import interp

from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier


def read_bigcsv(filename, **kw):
    with open(filename) as rf:
        reader = pd.read_csv(rf, **kw, iterator=True)
        chunkSize = 100000
        chunks = []
        while True:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                print("Iteration is stopped.")
                break
        df = pd.concat(chunks, axis=0, join='outer', ignore_index=True)
    return df


def timestamp2datetime(value):
    value = time.localtime(value)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', value)
    return dt


'''
from matplotlib import pyplot as plt
tt = data['context_timestamp']
plt.plot(tt)
# 可以看出时间是没有排好的,有一定的错位。如果做成online的模型，一定要将时间排好。
# aa = data[data['user_id']==24779788309075]
aa = data_train[data_train.duplicated(subset=None, keep='first')]
bb = data_train[data_train.duplicated(subset=None, keep='last')]
cc = data_train[data_train.duplicated(subset=None, keep=False)]

a2 = pd.DataFrame(train_id)[pd.DataFrame(train_id).duplicated(keep=False)]
b2 = train_id[train_id.duplicated(keep='last')]
c2 = train_id[train_id.duplicated(keep=False)]

c2 = data_train[data_train.duplicated(subset=None, keep=False)]

经验证, 'instance_id'有重复
a3 = Xdata[Xdata['instance_id']==1037061371711078396]
'''


def convert_timestamp(data):
    '''
    1. convert timestamp to datetime.
    2. no sort, no reindex.
    data.duplicated(subset=None, keep='first')
    TagTime from-to is ('2018-09-18 00:00:01', '2018-09-24 23:59:47')
    'user_query_day', 'user_query_day_hour', 'hour',
    np.corrcoef(data['user_query_day'], data['user_query_hour'])
    np.corrcoef(data['user_query_hour'], data['user_query_day_hour'])
    np.corrcoef(data['user_query_day'], data['user_query_day_hour'])
    '''
    data['TagTime'] = data['context_timestamp'].apply(timestamp2datetime)
    # data['TagTime'][0], data['TagTime'][len(data) - 1]
    # x = data['TagTime'][len(data) - 1]
    data['context_day'] = data['TagTime'].apply(lambda x: int(x[8:10]))
    data['context_hour'] = data['TagTime'].apply(lambda x: int(x[11:13]))
    query_day = data.groupby(['user_id', 'context_day']).size(
            ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, query_day, 'left', on=['user_id', 'context_day'])
    query_hour = data.groupby(['user_id', 'context_hour']).size(
            ).reset_index().rename(columns={0: 'user_query_hour'})
    data = pd.merge(data, query_hour, 'left', on=['user_id', 'context_hour'])
    query_day_hour = data.groupby(
            by=['user_id', 'context_day', 'context_hour']).size(
                    ).reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, query_day_hour, 'left',
                    on=['user_id', 'context_day', 'context_hour'])
    return data


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def data_baseline():
    filename = '../data/round1_ijcai_18_train_20180301.txt'
    data = read_bigcsv(filename, sep=' ')
    # data = pd.read_csv(filename, sep=' ')
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)  # very important
    data = convert_timestamp(data)
    train = data.loc[data['context_day'] < 24]  # 18,19,20,21,22,23,24
    test = data.loc[data['context_day'] == 24]  # 暂时先使用第24天作为验证集
    features = [
            'item_id', 'item_brand_id', 'item_city_id', 'item_price_level',
            'item_sales_level', 'item_collected_level', 'item_pv_level',
            'user_gender_id', 'user_age_level', 'user_occupation_id',
            'user_star_level', 'context_page_id', 'shop_id',
            'shop_review_num_level', 'shop_review_positive_rate',
            'shop_star_level', 'shop_score_service',
            'shop_score_delivery', 'shop_score_description',
            'user_query_day', 'user_query_day_hour', 'context_hour',
            ]
    x_train = train[features]
    x_test = test[features]
    y_train = train['is_trade']
    y_test = test['is_trade']
    return x_train, x_test, y_train, y_test
# x_train, x_test, y_train, y_test = data_baseline()


def model_baseline(x_train, y_train, x_test, y_test):
    cat_names = [
        'item_price_level',
        'item_sales_level',
        'item_collected_level',
        'item_pv_level',
        'user_gender_id',
        'user_age_level',
        'user_occupation_id',
        'user_star_level',
        'context_page_id',
        'shop_review_num_level',
        'shop_star_level',
        ]
    print("begin train...")
    kw_lgb = dict(num_leaves=63, max_depth=7, n_estimators=80, random_state=6,)
    clf = lgb.LGBMClassifier(**kw_lgb)
    clf.fit(x_train, y_train, categorical_feature=cat_names,)
    prob = clf.predict_proba(x_test,)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    # print(loss_val)  # 0.0848226750637
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    fig = plt.figure('fig1')
    ax = fig.add_subplot(1, 1, 1)
    name = 'base_lgb'
    plt.plot(mean_fpr, mean_tpr, linestyle='--',
             label='{} (area = %0.2f, logloss = %0.4f)'.format(name) %
             (x_auc, loss_val), lw=2)
    y_pred = clf.predict(x_test)
    cm1 = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1], title='Confusion matrix base1')
    # add weighted according to the labels
    clf = lgb.LGBMClassifier(**kw_lgb)
    clf.fit(x_train, y_train,
            sample_weight=[1 if y == 1 else 0.02 for y in y_train],
            categorical_feature=cat_names)
    prob = clf.predict_proba(x_test,)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    name = 'base_lgb_weighted'
    plt.figure('fig1')  # 选择图
    plt.plot(
            mean_fpr, mean_tpr, linestyle='--',
            label='{} (area = %0.2f, logloss = %0.4f)'.format(name) %
            (x_auc, loss_val), lw=2)
    y_pred = clf.predict(x_test)
    cm2 = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1],
                          title='Confusion matrix basemodle')
    plt.figure('fig1')  # 选择图
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return cm1, cm2, fig


def model_baseline3(x_train, y_train, x_test, y_test):
    bagging = BaggingClassifier(random_state=0)
    balanced_bagging = BalancedBaggingClassifier(random_state=0)
    bagging.fit(x_train, y_train)
    balanced_bagging.fit(x_train, y_train)
    prob = bagging.predict_proba(x_test)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    y_pred = [1 if x > 0.5 else 0 for x in predict_score]
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    fig = plt.figure('Bagging')
    ax = fig.add_subplot(1, 1, 1)
    name = 'base_Bagging'
    plt.plot(mean_fpr, mean_tpr, linestyle='--',
             label='{} (area = %0.2f, logloss = %0.2f)'.format(name) %
             (x_auc, loss_val), lw=2)
    y_pred_bagging = bagging.predict(x_test)
    cm_bagging = confusion_matrix(y_test, y_pred_bagging)
    cm1 = plt.figure()
    plot_confusion_matrix(cm_bagging,
                          classes=[0, 1],
                          title='Confusion matrix of BaggingClassifier')
    # balanced_bagging
    prob = balanced_bagging.predict_proba(x_test)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    plt.figure('Bagging')  # 选择图
    name = 'base_Balanced_Bagging'
    plt.plot(
            mean_fpr, mean_tpr, linestyle='--',
            label='{} (area = %0.2f, logloss = %0.2f)'.format(name) %
            (x_auc, loss_val), lw=2)
    y_pred_balanced_bagging = balanced_bagging.predict(x_test)
    cm_balanced_bagging = confusion_matrix(y_test, y_pred_balanced_bagging)
    cm2 = plt.figure()
    plot_confusion_matrix(cm_balanced_bagging,
                          classes=[0, 1],
                          title='Confusion matrix of BalancedBagging')
    plt.figure('Bagging')  # 选择图
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return cm1, cm2, fig


def model_baseline2(x_train, y_train, x_test, y_test):
    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 2,
            'verbose': 0,
            'metric': 'logloss',
            'max_bin': 255,
            'max_depth': 7,
            'learning_rate': 0.3,
            'nthread': 4,
            'n_estimators': 85,
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'num_boost_round': 160,
            }
    lgb_train = lgb.Dataset(x_train, label=y_train)
    lgb_eval = lgb.Dataset(x_test, label=y_test, reference=lgb_train)
    print("begin train...")
    bst = lgb.train(params, lgb_train, valid_sets=lgb_eval)
    prob = bst.predict(x_test)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    y_pred = [1 if x > 0.5 else 0 for x in predict_score]
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    x_auc = auc(fpr, tpr)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    fig = plt.figure('weighted')
    ax = fig.add_subplot(1, 1, 1)
    name = 'base_lgb'
    plt.plot(mean_fpr, mean_tpr, linestyle='--',
             label='{} (area = %0.2f, logloss = %0.2f)'.format(name) %
             (x_auc, loss_val), lw=2)
    cm1 = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1],
                          title='Confusion matrix basemodle')
    # add weighted according to the labels
    lgb_train = lgb.Dataset(
            x_train, label=y_train,
            weight=[1 if y == 1 else 0.02 for y in y_train])
    lgb_eval = lgb.Dataset(
            x_test, label=y_test, reference=lgb_train,
            weight=[1 if y == 1 else 0.02 for y in y_test])
    bst = lgb.train(params, lgb_train, valid_sets=lgb_eval)
    prob = bst.predict(x_test)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    y_pred = [1 if x > 0.5 else 0 for x in predict_score]
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    plt.figure('weighted')  # 选择图
    name = 'base_lgb_weighted'
    plt.plot(
            mean_fpr, mean_tpr, linestyle='--',
            label='{} (area = %0.2f, logloss = %0.2f)'.format(name) %
            (x_auc, loss_val), lw=2)
    cm2 = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1],
                          title='Confusion matrix basemodle')
    plt.figure('weighted')  # 选择图
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return cm1, cm2, fig


'''
1. logloss VS AUC
虽然 baseline 的 logloss= 0.0819, 确实很小，但是从 Confusion matrix 看出，
模型倾向于将所有的数据都分成多的那个，加了weight 之后稍好一点？
Though the logloss is 0.0819, which is a very small value.
Confusion matrix shows y_pred all 0, which feavors the majority classes.

AUC 只有 0.64~0.67.
AUC如此小，按理来说不应该啊，但是为什么呢？
因为数据的label 极度不平衡，1 的比例大概只有 2%. 50:1.
AUC 对不平衡数据的分类性能测试更友好，用AUC去选特征，可能结果更好哦。
这里只提供一个大概的思考改进点。
2. handling with imbalanced data:
    1. resampling, over- or under-,
    over- is increasing # of minority, under- is decreasing # of majority.
    2. revalue the loss function by giving large loss of misclassifying the
    minority labels.
'''


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_baseline()
    cm11, cm12, fig1 = model_baseline(x_train, y_train, x_test, y_test)
    cm21, cm22, fig2 = model_baseline2(x_train, y_train, x_test, y_test)
    cm31, cm32, fig3 = model_baseline3(x_train, y_train, x_test, y_test)

    fig1.savefig('./base_lgb_weighted.jpg', format='jpg')
    cm11.savefig('./Confusion matrix1.jpg', format='jpg')
    cm12.savefig('./Confusion matrix2.jpg', format='jpg')
