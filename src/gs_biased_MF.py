#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from data_process import load_data, split_data
dat_dir = '../data/'

ratings = load_data(dat_dir + "data_train.csv")
print(np.shape(ratings))

_, train, test = split_data(ratings, p_test=0.1)

from SGD_helpers import init_MF, matrix_factorization_SGD

from MF_helpers import get_bias_train, get_bias_test

bias_train, overal_bias, bias_u_train, bias_i_train = get_bias_train(train) #ratings for final submissions
bias_test = get_bias_test(test, overal_bias, bias_u_train, bias_i_train)

# Grid Search:
grid = np.zeros((3, 4, 4))
gamma = 0.025
num_features = np.array([20, 50, 100])
lambda_user = np.logspace(-3,0,4)[::-1]
lambda_item = np.logspace(-3,0,4)[::-1]
num_epochs = 20

best_user_features = []
best_item_features = []

tempt_dir = '../submit/'

loss_least = 99999
for i,K in enumerate(num_features):
    user_features, item_features = init_MF(train, int(K))
    for y,lambda_u in enumerate(lambda_user):
        for z,lambda_i in enumerate(lambda_item):
            print("K = {}, lambda_u = {}, lambda_i = {}".format(int(K), lambda_u, lambda_i))
            item_feats, user_feats, rmse = matrix_factorization_SGD(bias_train, bias_test, gamma, int(K), lambda_u, lambda_i, num_epochs, user_features, item_features)
            if rmse < min_loss:
                min_loss = rmse
#               user_features = user_feats
#               item_features = item_feats
                best_user_features = np.copy(user_feats)
                best_item_features = np.copy(item_feats)
            grid[x, y, z] = rmse
            
            np.save(tempt_dir + 'user_feature_'+ str(x) +'_'+ str(y) + '_'+ str(z) +'.npy', user_feats)
            np.save(tempt_dir +'item_feature_'+ str(x) +'_'+ str(y) + '_'+ str(z) +'.npy', item_feats)
            
            np.save('grid.npy', grid)
            np.save('best_user_features_bias.npy', best_user_features)
            np.save('best_item_features_bias.npy', best_item_features)