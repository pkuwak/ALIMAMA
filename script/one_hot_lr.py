import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, log_loss
from sklearn.pipeline import make_pipeline

# =============================================================================
# X, y = make_classification(n_samples=80000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# =============================================================================

data = pd.read_csv('../data/data.csv', index_col=0).dropna()
data = data.drop(labels=['item_property_ids'], axis=1)

for col in data.columns:
    if 'id' in col:
        data[col] = data[col].astype(str)
    elif 'hit' in col:
        data[col] = data[col].astype(np.float32)
    elif 'score' in col:
        data[col] = data[col].astype(np.float32)
    else:
        data[col] = data[col].astype(np.int32)

test_day = data['day'].max()
train, test = data[data['day'] < test_day], data[data['day'] == test_day]

X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

n_estimator = 100

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
    random_state=0)

rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)


# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)


# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()

# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()

first_layer = [grd, rf, pipeline]
dataset_blend_train = []
dataset_blend_test = []
for clf in first_layer:
    # pre = clf.predict_proba(X_train)[:, 1]
    pre = clf.predict(X_train)    
    dataset_blend_train.append(pre)
    # pre = clf.predict_proba(X_test)[:, 1]
    pre = clf.predict(X_test)
    dataset_blend_test.append(pre)
    print(log_loss(y_test, pre))
dataset_blend_train = np.array(dataset_blend_train).T
dataset_blend_test = np.array(dataset_blend_test).T
clf = LogisticRegression()
clf.fit(dataset_blend_train, y_train)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
print(log_loss(y_test, y_submission))