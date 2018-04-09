'''
特征工具箱
'''

import time
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
        self.data = pd.DataFrame()

    def feature_program(self):
        self.category_match()
        self.time_feature()
        self.data['is_trade'] = self.raw['is_trade']

        return self.data

    def category_match(self, sim_func=None):
        if not sim_func:
            sim_func = lambda x, y: len(x & y)
        return_list = list()
        for i, row in self.raw.loc[:, [
            'item_category_list', 'item_property_list', 'predict_category_property']].iterrows():
            ics, ips, qcps = row
            # item特征
            ics = ics.split(';')
            c1 = ics[1]
            c2 = 0 if len(ics) <= 2 else ics[2]
            ip_set = set(ips.split(';'))
            # query特征
            qcps = qcps.split(';')
            qc_set = set()
            qp_set = set()
            for qcp in qcps:
                qc, qps = qcp.split(':')
                qps = set(qps.split(','))
                qc_set.add(qc)
                qp_set |= qps
            qp_set -= set(['-1'])
            # 匹配度计算
            hit1 = c1 in qc_set
            hit2 = c2 in qc_set
            property_similarity = sim_func(ip_set, qp_set)
            return_list.append([hit1, hit2, property_similarity])
        self.data['hit1'], self.data['hit2'], self.data['prosim'] =\
            np.array(return_list).T
        return
    
    @staticmethod
    def timestamp_day_hour(stamp):
        struct_time = time.localtime(stamp)
        week_day = struct_time.tm_wday
        hour = struct_time.tm_hour
        minute = struct_time.tm_min
        return week_day, hour, minute

    def time_feature(self):
        self.data['week_day'], self.data['hour'], self.data['minute'] = \
            np.array(list(self.raw['context_timestamp'].apply(self.timestamp_day_hour))).T
        return


if __name__ == '__main__':
    data = pd.read_csv('../data/round1_ijcai_18_train_20180301.txt', sep=' ').iloc[:10000]
    feabox = featureBox(data)
    features = feabox.feature_program()
