#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)/num_features
    item_features = np.random.rand(num_features, num_item)/num_features

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))

def matrix_factorization_SGD(train, test, gamma, num_features, lambda_user, lambda_item, num_epochs, user_features, item_features):
    """matrix factorization by SGD."""
    # set seed
    np.random.seed(988)
     
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
        
    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)
            
        if(it % 5 == 0 or it == num_epochs - 1):
            rmse = compute_error(train, user_features, item_features, nz_train)
            print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
    # evaluate the test error
    if len(nz_test) > 0:
        rmse = compute_error(test, user_features, item_features, nz_test)
        print("RMSE on test data: {}.".format(rmse))

    return item_features, user_features, rmse