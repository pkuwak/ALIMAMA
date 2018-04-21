'''
特征工具箱
'''

import time
import itertools
import sklearn.preprocessing as skp
import numpy as np
import pandas as pd

class featureBox(object):
    def __init__(self, data):
        '''
        input:pd.DataFrame
        instance_id item_id item_category_list item_property_list item_brand_id item_city_id item_price_level item_sales_level item_collected_level item_pv_level user_id user_gender_id user_age_level user_occupation_id user_star_level context_id context_timestamp context_page_id predict_category_property shop_id shop_review_num_level shop_review_positive_rate shop_star_level shop_score_service shop_score_delivery shop_score_description
        '''
        self.raw = data
        self.raw = self.raw.drop_duplicates(subset='context_id')
        self.data = pd.DataFrame()

        

    def feature_program(self):
        self.item_bias()
        self.user_bias()
        self.context_bias()
        self.similar_features()
        self.shop_bias()
        self.data['is_trade'] = self.raw['is_trade']

        return self.data

    def item_bias(self):
        '''
        item 偏置项
        '''
        # self.data['item_id'] = self.raw['item_id']
        self.data['item_brand_id'] = self.raw['item_brand_id'].map(str)
        self.data['item_city_id'] = self.raw['item_city_id'].map(str)
        self.data['item_price_level'] = self.raw['item_price_level']
        self.data['item_sales_level'] = self.raw['item_sales_level']
        self.data['item_collected_level'] = self.raw['item_collected_level']
        self.data['item_pv_level'] = self.raw['item_pv_level']
        self.data['item_categoryi_id'], self.data['item_if_j'], self.data['item_categoryj_id'] = self.category_id()
        self.data['item_property_ids'] = self.property_list()
        return

    def category_id(self):
        '''
        item bias 组件
        '''
        cl = self.raw['item_category_list'].map(lambda x: x.split(';'))
        ci = cl.map(lambda x: x[1])
        if_cj, cj = zip(*cl.map(lambda x: (1, x[2]) if (len(x) > 2) else (0, 0)))
        return ci, if_cj, cj        

    def property_list(self):
        '''
        item bias 组件
        '''
        pl = self.raw['item_property_list'].map(lambda x: x.split(';'))
        return pl

    def shop_bias(self):
        '''
        shop 偏置项
        '''
        # self.data['shop_id'] = self.raw['shop_id']
        self.data['shop_review_num_level'] = self.raw['shop_review_num_level']
        self.data['shop_review_positive_rate'] = self.raw['shop_review_positive_rate']
        self.data['shop_star_level'] = self.raw['shop_star_level']
        self.data['shop_score_service'] = self.raw['shop_score_service']
        self.data['shop_score_delivery'] = self.raw['shop_score_delivery']
        self.data['shop_score_description'] = self.raw['shop_score_description']
        return

    def user_bias(self):
        '''
        user 偏置项
        '''
        # self.data['user_id'] = self.raw['user_id']
        self.data['user_gender_id'] = self.raw['user_gender_id']
        self.data['user_if_personas'], self.data['user_age_level'] = self.user_age_level()
        self.data['user_occupation_id'] = self.raw['user_occupation_id']
        self.data['user_star_level'] = self.user_star_level()

    def user_age_level(self):
        '''
        user bias 组件
        '''
        age = self.raw['user_age_level']
        mean = age[age.values != -1].mean() - 1000
        if_age_age = age.map(lambda x: (1, (x-1000)) if x != -1 else (0, mean))
        if_age, age = zip(*if_age_age)
        return if_age, age

    def user_star_level(self):
        '''
        user bias 组件
        '''
        star = self.raw['user_star_level']
        mean = star[star.values != -1].mean() - 3000
        star = star.map(lambda x: mean if x == -1 else x - 3000)
        return star

    def context_bias(self):
        '''
        转化时间特征
        '''
        self.data['day'], self.data['context_wday'], self.data['context_h'], self.data['context_m'] = \
            np.array(list(self.raw['context_timestamp'].apply(self.timestamp_analysis))).T
        self.data['context_page_id'] = self.raw['context_page_id'] - 4000
        return


    @staticmethod
    def timestamp_analysis(stamp):
        '''
        context bias 组件
        时间戳字符串转化为 天/365 星期 小时 分钟
        '''
        struct_time = time.localtime(stamp)
        year_day = struct_time.tm_yday
        week_day = struct_time.tm_wday
        hour = struct_time.tm_hour
        minute = struct_time.tm_min
        return year_day, week_day, hour, minute

    def similar_features(self):
        '''
        计算query和item是否相似
        '''
        self.data[['categoryi_hit', 'categoryj_hit', 'property_hit']] =\
            self.query_item_similar()

    
    def query_item_similar(self, sim_func=None):
        '''
        similar_features 组件
        计算query和item的category是否命中
        计算query和item的property是否相似
        '''
        if not sim_func:
            sim_func = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
        result = list()
        self.data['predict_category_property'] = self.raw['predict_category_property']
        for i, row in self.data.loc[:, [
            'item_categoryi_id', 
            'item_categoryj_id', 
            'item_property_ids', 
            'predict_category_property']].iterrows():
            ici, icj, ips, qcps = row
            if qcps == '-1':
                result.append([0, 0, 0])
                continue
            # query 特征 =============================
            qcps = qcps.split(';')
            qc_set = set()
            qp_set = set()
            for qcp in qcps:
                qc, qps = qcp.split(':')
                qps = set(qps.split(','))
                qc_set.add(qc)
                qp_set |= qps
            qp_set -= set(['-1'])
            # ========================================

            ihit = sim_func([icj], qc_set)
            jhit = sim_func([icj], qc_set)
            phit= sim_func(ips, qp_set)
            result.append([ihit, jhit, phit])
        self.data.drop(labels='predict_category_property', axis=1, inplace=True)

        return pd.DataFrame(result)
    

if __name__ == '__main__':
    data = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep=' ')
    feabox = featureBox(data)
    # features = feabox.feature_program()
    feabox.feature_program()
    features = feabox.data
    features.to_csv('../data/data.csv', index=False)


