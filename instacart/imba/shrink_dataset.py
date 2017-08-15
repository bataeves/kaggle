# -*- coding: utf-8 -*-
__author__ = 'bataev.evgeny@gmail.com'

import pandas as pd

data = pd.read_pickle("data/dataset.pkl")
data = data.drop(["product_name", "eval_set"], axis=1)
data.to_pickle("data/dataset2.pkl")

