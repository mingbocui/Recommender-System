#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from data_process import load_data, split_data
dat_dir = '../data/'

ratings = load_data(dat_dir + "data_train.csv")
print(np.shape(ratings))

_, train, test = split_data(ratings, p_test=0.1)

from SGD_helpers import init_MF, matrix_factorization_SGD

# Grid Search:
grid = np.zeros((4, 4, 4))
gamma = 0.025 # best gamma we found above
num_epochs = 20
lambdas_user = np.logspace(-3,0,4)[::-1] #From max to min
lambdas_item = np.logspace(-3,0,4)[::-1]
num_features = np.array([10, 20, 50 ,100])
min_loss = 99999
best_user_features = []
best_item_features = []

for x,K in enumerate(num_features):
    user_init, item_init = init_MF(train, int(K))
    for y,lambda_u in enumerate(lambdas_user):
        for z,lambda_i in enumerate(lambdas_item):
            print("K = {}, lambda_u = {}, lambda_i = {}".format(int(K), lambda_u, lambda_i))
            item_feats, user_feats, rmse = matrix_factorization_SGD(ratings, test, gamma, int(K), lambda_u, lambda_i, num_epochs, user_init, item_init)
            if rmse < min_loss:
                min_loss = rmse
                user_init = user_feats
                item_init = item_feats
                best_user_features = np.copy(user_feats)
                best_item_features = np.copy(item_feats)
            grid[x, y, z] = rmse
        np.save('rmse_lambda_K.npy', grid)