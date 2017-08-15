import gc
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

features_to_drop = [
    'user_id',
    'user_id_x',
    'reordered_dow_ration', 'reordered_dow', 'reordered_dow_size',
    'user_median_days_since_prior', 'up_median_cart_position',
    'days_since_prior_order_median', 'order_dow_median',
    'order_hour_of_day_median',
    "reordered", 'product_name',
    'aisle_id_y',
    'department_id_y',
    'user_id_y',
    'eval_set',
    'order_hour_of_day_y',
    'days_since_prior_order_y',
    'user_average_basket_y'
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


def tocsv(df, name="train"):
    df.to_csv("data/%s.csv" % name, index=False, header=True)


if __name__ == "__main__":
    raw_data = pd.read_pickle("data/dataset.pkl")

    # 30% of data
    gkf = GroupShuffleSplit(n_splits=1, test_size=0.3)
    for train_idx, test_idx in gkf.split(raw_data.index, groups=raw_data.user_id_x):
        raw_data = raw_data.loc[test_idx]
        break
    print("dataset loaded")

    user_id = raw_data.user_id_x
    labels = raw_data[['reordered']].values.astype(np.float32).flatten()

    data = features_select(raw_data)

    data.insert(0, 'reordered', labels)

    del raw_data
    gc.collect()

    categories = [
        'product_id',
        'aisle_id', 'department_id'
    ]
    # categories = map(list(data.columns).index, categories)
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

        # for fname in list(data.columns)[1:]:
        #     print(fname)
        #     dftrain.drop(fname, axis=1).to_csv("data/train/%s.csv" % fname, index=False, header=False)
        #     dfval.drop(fname, axis=1).to_csv("data/val/%s.csv" % fname, index=False, header=False)

        dftrain.to_csv("data/train.csv", index=False, header=False)
        dfval.to_csv("data/val.csv", index=False, header=False)

        pickle.dump(feature_name, open("data/feature_name.pkl", mode="wb"))

        break
