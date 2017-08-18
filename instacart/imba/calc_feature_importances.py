# -*- coding: utf-8 -*-
__author__ = 'bataev.evgeny@gmail.com'
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit

categories = [
    'product_id',
    'aisle_id', 'department_id'
]

features = [
               'product_id',
               'aisle_id', 'department_id',
               # 'reordered_dow_ration', 'reordered_dow', 'reordered_dow_size',
               # 'reordered_prev', 'add_to_cart_order_prev', 'order_dow_prev', 'order_hour_of_day_prev',
               'user_product_reordered_ratio', 'reordered_sum',
               'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
               'reorder_prob',
               'last', 'prev1', 'prev2', 'median', 'mean',
               'dep_reordered_ratio', 'aisle_reordered_ratio',
               'aisle_products',
               'aisle_reordered',
               'dep_products',
               'dep_reordered',
               'prod_users_unq', 'prod_users_unq_reordered',
               'order_number', 'prod_add_to_card_mean',
               'days_since_prior_order',
               'order_dow', 'order_hour_of_day',
               'reorder_ration',
               'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
               # 'user_median_days_since_prior',
               'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
               'prod_orders', 'prod_reorders',
               'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
               'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
               # 'up_median_cart_position',
               'days_since_prior_order_mean',
               # 'days_since_prior_order_median',
               'order_dow_mean',
               # 'order_dow_median',
               'order_hour_of_day_mean',
               # 'order_hour_of_day_median'
           ] + [
               'user_total_orders',
               'user_total_items',
               'user_total_distinct_items',
               'user_average_days_between_orders',
               'user_max_days_between_orders',
               'user_min_days_between_orders',
               'user_std_days_between_orders',
               'user_reorders',
               'user_reorders_max',
               'user_reorders_min',
               'user_reorders_mean',
               'user_reorders_std',
               'user_reorder_rate',
               'user_period',
               'dow',
               'days_since_ratio',
               'product_orders',
               'product_users',
               'product_order_freq',
               'product_reorders',
               'product_reorders_max',
               'product_reorders_min',
               'product_reorders_mean',
               'product_reorders_std',
               'product_reorder_rate',
               'product_add_to_cart_order_mean',
               'product_add_to_cart_order_std',
               'aisle_orders',
               'aisle_users',
               'aisle_order_freq',
               'aisle_reorders',
               'aisle_reorder_rate',
               'aisle_add_to_cart_order_mean',
               'aisle_add_to_cart_order_std',
               'department_orders',
               'department_users',
               'department_order_freq',
               'department_reorders',
               'department_reorder_rate',
               'department_add_to_cart_order_mean',
               'department_add_to_cart_order_std',
               'UP_orders',
               'UP_orders_ratio',
               'UP_average_pos_in_cart',
               'UP_sum_add_to_cart_order',
               'UP_min_add_to_cart_order',
               'UP_mean_add_to_cart_order',
               'UP_max_add_to_cart_order',
               'UP_std_add_to_cart_order',
               'UP_sum_reordered',
               'UP_mean_reordered',
               'UP_std_reordered',
               'UP_reorders_rate',
               'UP_last_order_number',
               'UP_first_order_number',
               'UP_last_order_number_prc',
               'UP_first_order_number_prc',
               'UP_orders_since_last',
               'UP_orders_rate_since_first_order',
               'UP_weeks_sinse_last',
               'UP_days_sinse_last',
               'UP_delta_hour_vs_last',
               'UA_orders',
               'UA_orders_ratio',
               'UA_average_pos_in_cart',
               'UA_sum_add_to_cart_order',
               'UA_min_add_to_cart_order',
               'UA_mean_add_to_cart_order',
               'UA_max_add_to_cart_order',
               'UA_std_add_to_cart_order',
               'UA_sum_reordered',
               'UA_mean_reordered',
               'UA_std_reordered',
               'UA_reorders_rate',
               'UA_last_order_number',
               'UA_first_order_number',
               'UA_last_order_number_prc',
               'UA_first_order_number_prc',
               'UA_orders_since_last',
               'UA_orders_rate_since_first_order',
               'UA_weeks_sinse_last',
               'UA_days_sinse_last',
               'UA_delta_hour_vs_last',
               'UD_orders',
               'UD_orders_ratio',
               'UD_average_pos_in_cart',
               'UD_sum_add_to_cart_order',
               'UD_min_add_to_cart_order',
               'UD_mean_add_to_cart_order',
               'UD_max_add_to_cart_order',
               'UD_std_add_to_cart_order',
               'UD_sum_reordered',
               'UD_mean_reordered',
               'UD_std_reordered',
               'UD_reorders_rate',
               'UD_last_order_number',
               'UD_first_order_number',
               'UD_last_order_number_prc',
               'UD_first_order_number_prc',
               'UD_orders_since_last',
               'UD_orders_rate_since_first_order',
               'UD_weeks_sinse_last',
               'UD_days_sinse_last',
               'UD_delta_hour_vs_last'
           ] + list(range(32))


def features_select(df):
    return df.rename(columns={
        'aisle_id_x': 'aisle_id',
        'department_id_x': 'department_id',
        'order_hour_of_day_x': 'order_hour_of_day',
        'days_since_prior_order_x': 'days_since_prior_order',
        'user_average_basket_x': 'user_average_basket'
    })[features]


def feature_importances2(dftrain, dfval, ytrain, yval, feature_name):
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 0
    }

    print('Start training...')

    lgb_train = lgb.Dataset(dftrain, ytrain, feature_name=feature_name, categorical_feature=categories)
    lgb_train.raw_data = None

    lgb_val = lgb.Dataset(dfval, yval, feature_name=feature_name, categorical_feature=categories)
    lgb_val.raw_data = None

    # train
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    params['verbose'] = 0
    global_score = gbm.best_score["valid_0"]["auc"]

    feature_importances_values = pd.read_csv(
        "feature_importances.csv",
        names=["name", 'gain', 'global_score', 'local_score', 'score_diff', 'iter']
    )
    existing_feats = list(feature_importances_values.name.values)

    features = gbm.feature_name()
    imp = pd.DataFrame({"name": features, "imp": gbm.feature_importance("gain")})
    with open("feature_importances.csv", "a") as f:
        for row in imp.sort_values("imp", ascending=False).itertuples():
            feature = row.name
            if feature in existing_feats:
                print("skip %s" % feature)
                continue

            if feature in [str(x) for x in range(32)]:
                continue

            gain = row.imp
            feature_name_sm = [x for x in feature_name if x != feature]
            categories_sm = [x for x in categories if x != feature]

            lgb_train = lgb.Dataset(dftrain.drop(feature, axis=1), ytrain, feature_name=feature_name_sm,
                                    categorical_feature=categories_sm)
            lgb_train.raw_data = None
            lgb_val = lgb.Dataset(dfval.drop(feature, axis=1), yval, feature_name=feature_name_sm,
                                  categorical_feature=categories_sm)
            lgb_val.raw_data = None

            # train
            gbm = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=1000,
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            local_score = gbm.best_score["valid_0"]["auc"]
            line = ",".join([
                feature, str(gain), str(global_score), str(local_score),
                str(global_score - local_score), str(gbm.best_iteration),
            ])
            f.write(line + "\n")
            f.flush()
            print(line)
            del lgb_train, lgb_val
            gc.collect()


def feature_importances(lgb_train, lgb_val):
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        'use_two_round_loading': True,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 0
    }

    print('Start training...')

    # train
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    global_score = gbm.best_score["valid_0"]["auc"]

    features = gbm.feature_name()
    imp = pd.DataFrame({"name": features, "imp": gbm.feature_importance("gain")})
    with open("feature_importances.tsv", "a") as f:
        for row in imp.sort_values("imp", ascending=False).itertuples():
            feature = row.name
            gain = row.imp
            feature_name_sm = [x for x in feature_name if x != feature]
            categories_sm = [x for x in categories if x != feature]

            lgb_train = lgb.Dataset("data/train/%s.csv" % feature, feature_name=feature_name_sm,
                                    categorical_feature=categories_sm)
            lgb_val = lgb.Dataset("data/val/%s.csv" % feature, feature_name=feature_name_sm,
                                  categorical_feature=categories_sm)

            # specify your configurations as a dict
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': {'binary_logloss', 'auc'},
                'num_leaves': 256,
                'min_sum_hessian_in_leaf': 20,
                'max_depth': 12,
                'learning_rate': 0.05,
                'feature_fraction': 0.6,
                # 'bagging_fraction': 0.9,
                # 'bagging_freq': 3,
                'verbose': 0,
                'verbosity': 0,
                'use_two_round_loading': True
            }

            # train
            gbm = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=1000,
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            local_score = gbm.best_score["valid_0"]["auc"]
            line = "\t".join([
                feature, str(gain), str(global_score), str(local_score),
                str(global_score - local_score), str(gbm.best_iteration),
            ]) + "\n"
            f.write(line)
            print(line)
            gc.collect()


if __name__ == "__main__":
    # feature_name = pickle.load(open("data/feature_name.pkl", "rb"))[1:]
    # ytrain = pickle.load(open("data/ytrain.pkl", "rb"))
    # yval = pickle.load(open("data/yval.pkl", "rb"))
    # print(feature_name)
    # lgb_train = lgb.Dataset("data/train.csv", feature_name=feature_name, categorical_feature=categories)
    # lgb_val = lgb.Dataset("data/val.csv", feature_name=feature_name, categorical_feature=categories)
    # feature_importances(lgb_train, lgb_val)

    raw_data = pd.read_pickle("data/dataset2.pkl")
    # 30% of data
    gkf = GroupShuffleSplit(n_splits=1, test_size=0.2)
    for train_idx, test_idx in gkf.split(raw_data.index, groups=raw_data.user_id_x):
        raw_data = raw_data.loc[test_idx]
        break

    print("dataset loaded")

    user_id = raw_data.user_id_x
    labels = raw_data[['reordered']].values.astype(np.float32).flatten()

    data = features_select(raw_data)
    del raw_data
    gc.collect()

    feature_name = [str(x) for x in data.columns]
    lgb_train = None
    lgb_val = None
    gkf = GroupShuffleSplit(n_splits=1, test_size=0.1)

    for train_idx, test_idx in gkf.split(data.index, groups=user_id):
        dftrain = data.iloc[train_idx]
        dfval = data.iloc[test_idx]
        ytrain = labels[train_idx]
        yval = labels[test_idx]

        print(dftrain.shape, dfval.shape)

        feature_importances2(dftrain, dfval, ytrain, yval, feature_name)
        break
