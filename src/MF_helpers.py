import numpy as np

"""Check README for more details"""

def get_bias_train(ratings):
    """Compute bias matrix where 
    bias[:] = r_ui - mu - bias_u - bias_i"""
    nb_item, nb_user = ratings.shape
    # find the non-zero ratings indices 
    non_zero = ratings.nonzero()
    non_zero = list(zip(non_zero[0], non_zero[1]))
    
    # overall_avg is overall mean
    overall_avg = np.sum(ratings)/len(non_zero)

    ratings_per_user = np.zeros((nb_user,), dtype=int)
    ratings_per_item = np.zeros((nb_item,), dtype=int)
    
    # Compute number of non zero ratings by user and item
    for item, user in non_zero:
        ratings_per_user[user] += 1
        ratings_per_item[item] += 1
    
    # user_avg is mean by user
    # item_avg is mean by item
    user_avg = np.array(np.sum(ratings, axis = 0)).squeeze()
    item_avg = np.array(np.sum(ratings, axis = 1)).squeeze()
    if ratings_per_user.min() == 0:
        print('some ratings_per_user = 0')
    if ratings_per_item.min() == 0:
        print('some ratings_per_user = 0')
    user_avg /= ratings_per_user
    item_avg /= ratings_per_item
    
    bias_user = user_avg - np.ones(nb_user) * overall_avg
    bias_item = item_avg - np.ones(nb_item) * overall_avg
    
    # compute ratings matrix considering bias
    for item, user in non_zero:
        ratings[item, user] -= (overall_avg + bias_user[user] + bias_item[item])
    return ratings, overall_avg, bias_user, bias_item

def get_bias_test(ratings, overall_avg, bias_user, bias_item):
    """Compute bias matrix use thr same bias from training set
    where 
    bias[:] = r_ui - mu - bias_u - bias_i"""
    nb_item, nb_user = ratings.shape
    # find the non-zero ratings indices 
    non_zero = ratings.nonzero()
    non_zero = list(zip(non_zero[0], non_zero[1]))
    
    # compute ratings matrix considering bias
    for item, user in non_zero:
        ratings[item, user] -= (overall_avg + bias_user[user] + bias_item[item])
    return ratings

def predict_no_bias(item_feature, user_feature):
    '''Compute predictions matrix using the formula: 
    pred_u_i = item_features.T @ user_features'''
    return np.dot(item_feature.T, user_feature)

def predict_with_bias(item_feature, user_feature, overal_avg, bias_user, bias_item):
    '''Compute predictions matrix using the formula: 
    pred_u_i = mean + bias_u + bias_i + item_features.T @ user_features'''
    
    # compute the prediction without bias
    prediction_no_Bias = predict_no_bias(item_feature, user_feature)
    nb_item, nb_user = prediction_no_Bias.shape

    # bias_User: corresponding to the user bias
    bias_User = np.tile(bias_user, (nb_item,1))

    # bias_Item: corresponding to the movie bias
    bias_Item = np.tile(bias_item, (nb_user,1)).T

    # overal_Bias: corresponding to the overal bias
    overal_Bias = np.ones((nb_item, nb_user)) * overal_avg
    
    # add the bias
    prediction_with_Bias = overal_Bias + bias_User + bias_Item + prediction_no_Bias
    
    return prediction_with_Bias