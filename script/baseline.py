import pandas as pd

test = pd.read_csv('../data/round1_ijcai_18_test_a_20180301.txt',sep=' ',usecols=[0,6])
train = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt',sep=' ',usecols=[6,26])
train = train.groupby('item_price_level',as_index=False).mean()
predict = pd.merge(train, test, on='item_price_level', how='right')
predict = predict.drop('item_price_level',axis=1)
predict = predict.reindex(columns=['instance_id', 'is_trade'])
predict.to_csv('../data/baseline.csv',index=False,sep=' ',header=['instance_id','predicted_score'])