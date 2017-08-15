import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_expectations(P, pNone=None):
    expectations = []
    P = np.sort(P)[::-1]

    n = np.array(P).shape[0]
    DP_C = np.zeros((n + 2, n + 1))
    if pNone is None:
        pNone = (1.0 - P).prod()

    DP_C[0][0] = 1.0
    for j in range(1, n):
        DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

    for i in range(1, n + 1):
        DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
        for j in range(i + 1, n + 1):
            DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

    DP_S = np.zeros((2 * n + 1,))
    DP_SNone = np.zeros((2 * n + 1,))
    for i in range(1, 2 * n + 1):
        DP_S[i] = 1. / (1. * i)
        DP_SNone[i] = 1. / (1. * i + 1)
    for k in range(n + 1)[::-1]:
        f1 = 0
        f1None = 0
        for k1 in range(n + 1):
            f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
            f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
        for i in range(1, 2 * k - 1):
            DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
            DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
        expectations.append([f1None + 2 * pNone / (2 + k), f1])

    return np.array(expectations[::-1]).T


def maximize_expectation(P, pNone=None):
    expectations = get_expectations(P, pNone)

    ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
    max_f1 = expectations[ix_max]

    predNone = True if ix_max[0] == 0 else False
    best_k = ix_max[1]

    return best_k, predNone, max_f1


def print_best_prediction(P, pNone=None):
    print("Maximize F1-Expectation")
    print("=" * 23)
    P = np.sort(P)[::-1]
    n = P.shape[0]
    L = ['L{}'.format(i + 1) for i in range(n)]

    if pNone is None:
        print("Estimate p(None|x) as (1-p_1)*(1-p_2)*...*(1-p_n)")
        pNone = (1.0 - P).prod()

    PL = ['p({}|x)={}'.format(l, p) for l, p in zip(L, P)]
    print("Posteriors: {} (n={})".format(PL, n))
    print("p(None|x)={}".format(pNone))

    opt = maximize_expectation(P, pNone)
    best_prediction = ['None'] if opt[1] else []
    best_prediction += (L[:opt[0]])
    f1_max = opt[2]

    print("Prediction {} yields best E[F1] of {}\n".format(best_prediction, f1_max))


def final_predict(df_test):
    d = dict()

    current_order_id = None
    current_order_count = 0
    current_order_basket_size = 0
    for row in tqdm(df_test.sort_values(
        by=["order_id", "pred"],
        ascending=[False, False]
    ).itertuples(), total=len(df_test)):
        order_id = row.order_id
        pred_value = row.pred

        # if pred_value < 0.01:
        #     continue

        if order_id != current_order_id:
            current_order_id = order_id
            current_order_count = 0
            P = df_test[df_test.order_id == order_id].pred.values
            best_k, predNone, max_f1 = maximize_expectation(P)
            current_order_basket_size = best_k
            if predNone:
                d[order_id] = 'None'

        if current_order_count >= current_order_basket_size:
            continue

        current_order_count += 1
        try:
            d[order_id] += ' ' + str(row.product_id)
        except KeyError:
            d[order_id] = str(row.product_id)

    for order_id in df_test.order_id:
        if order_id not in d:
            d[order_id] = 'None'

    sub = pd.DataFrame.from_dict(d, orient='index')
    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']
    return sub


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
                   ] + [
                       # 'product_id',
                       # 'aisle_id',
                       # 'department_id',
                       'UP_days_sinse_last',
                       'median',
                       'prev2',
                       'UP_weeks_sinse_last',
                       'days_since_prior_order',
                       'aisle_reordered_ratio',
                       'user_reorder_ratio',
                       'prod_users_unq',
                       'up_last_order',
                       'up_order_rate',
                       'product_reorder_rate',
                       'UA_days_sinse_last',
                       'UD_mean_reordered',
                       'user_mean_days_since_prior',
                       'user_reorders_mean',
                       'UP_delta_hour_vs_last',
                       'UD_days_sinse_last',
                       'product_users',
                       'UA_mean_reordered',
                       'user_product_reordered_ratio',
                       'prod_users_unq_reordered',
                       'UD_reorders_rate',
                       'UA_delta_hour_vs_last',
                       'up_order_rate_since_first_order',
                       'UD_std_add_to_cart_order',
                       'UD_delta_hour_vs_last',
                       'UA_orders_rate_since_first_order',
                       'UD_std_reordered',
                       'UA_std_add_to_cart_order',
                       'reorder_ration',
                       'UA_last_order_number_prc',
                       'UD_average_pos_in_cart',
                       'product_orders',
                       'UP_first_order_number',
                       'user_total_products',
                       'UA_first_order_number_prc',
                       'UD_orders_rate_since_first_order',
                       'UA_std_reordered',
                       'UA_reorders_rate',
                       'add_to_cart_order_relative_mean',
                       'UA_sum_add_to_cart_order',
                       'UA_orders_ratio',
                       'UP_std_reordered',
                       'UD_sum_add_to_cart_order',
                       'UD_orders_ratio',
                       'UD_first_order_number_prc',
                       'user_total_items',
                       'UA_average_pos_in_cart',
                       'prod_reorders',
                       'user_orders',
                       'UA_orders_since_last',
                   ]

features = [
               # 'product_id',
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
           ] + list(range(32)) + \
           [
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
           ]

categories = list({
                      # 'product_id',
                      'aisle_id',
                      'department_id'
                  } - set(features_to_drop))

print(categories)


# ['UP_orders_rate_since_first_order',
#  'UP_orders_ratio',
#  'UP_orders_since_last',
#  'UP_last_order_number_prc',
#  'product_reorders_mean',
#  'user_average_days_between_orders',
#  'user_reorders_std',
#  'UP_first_order_number_prc',
#  'user_reorder_rate',
#  'user_std_days_between_orders',
#  'user_min_days_between_orders',
#  'user_average_basket',
#  'UP_std_add_to_cart_order',
#  'UD_max_add_to_cart_order',
#  'user_total_distinct_items',
#  'user_reorders',
#  'product_reorders_std',
#  'UP_sum_add_to_cart_order',
#  'user_max_days_between_orders',
#  'UD_last_order_number_prc',
#  'UP_std_reordered',
#  'UP_average_pos_in_cart']

def features_select(df):
    df = df.rename(columns={
        'aisle_id_x': 'aisle_id',
        'department_id_x': 'department_id',
        'order_hour_of_day_x': 'order_hour_of_day',
        'days_since_prior_order_x': 'days_since_prior_order',
        'user_average_basket_x': 'user_average_basket'
    })[features]
    return df
    # return .drop(features_to_drop, axis=1, errors="ignore")


def tocsv(df, lbl, name="train"):
    df["reordered"] = lbl
    df.to_csv("data/%s.csv" % name, index=False, header=True)


if __name__ == "__main__":
    raw_data = pd.read_pickle("data/dataset2.pkl")
    print("dataset loaded")

    # user_id = raw_data.user_id_x
    labels = raw_data[['reordered']].values.astype(np.float32).flatten()

    data = features_select(raw_data)
    del raw_data
    gc.collect()

    # categories = map(list(data.columns).index, categories)
    feature_name = [str(x) for x in data.columns]
    print(feature_name)

    lgb_train = lgb.Dataset(data, labels, feature_name=feature_name, categorical_feature=categories)
    lgb_train.raw_data = None
    gc.collect()

    # lgb_train = None
    # lgb_val = None
    # gkf = GroupShuffleSplit(n_splits=1, test_size=0.1)
    #
    # for train_idx, test_idx in gkf.split(data.index, groups=user_id):
    #     dftrain = data.iloc[train_idx]
    #     dfval = data.iloc[test_idx]
    #     ytrain = labels[train_idx]
    #     yval = labels[test_idx]
    #
    #     print(dftrain.shape, dfval.shape)
    #
    #     tocsv(dftrain, ytrain, "train")
    #     tocsv(dfval, yval, "val")
    #     pickle.dump(feature_name, open("data/feature_name.pkl", mode="w"))
    #     exit(1)
    #
    #     lgb_train = lgb.Dataset(dftrain, ytrain, feature_name=feature_name, categorical_feature=categories)
    #     lgb_train.raw_data = None
    #
    #     lgb_val = lgb.Dataset(dfval, yval, feature_name=feature_name, categorical_feature=categories)
    #     lgb_val.raw_data = None
    #
    #     del data, labels
    #     del dftrain, dfval
    #     gc.collect()
    #     break

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 512,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 15,
        'learning_rate': 0.03,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }

    print('Start training...')

    # train
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=420,
        # valid_sets=[lgb_val],
        # early_stopping_rounds=100,
        # verbose_eval=100,
    )

    gbm.save_model("data/model.pkl")

    print('Predicting...')
    print('Read data_val')
    data_val = pd.read_pickle("data/dataset_val.pkl")
    print('Predict')
    prediction = gbm.predict(features_select(data_val))

    orders = data_val.order_id.values
    products = data_val.product_id.values

    result = pd.DataFrame({'product_id': products, 'order_id': orders, 'pred': prediction})
    print('Construct Submission')
    subdf = final_predict(result)
    subdf.to_csv('sub.csv', index=False)
