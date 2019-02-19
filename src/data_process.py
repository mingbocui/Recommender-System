#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy.sparse as sp

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]
    
    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_col, max_row))

    # build rating matrix.
    ratings = sp.lil_matrix((max_col, max_row))
    
    for row, col, rating in data:
        ratings[ col- 1, row - 1] = rating

    return ratings

def split_data(ratings, p_test=0.1):
    """
    Split the ratings to training data and test data
    Args: ratings: the matrix contains the data
    """
    # set seed
    np.random.seed(988)
    
    # select all the users and items
    valid_ratings = ratings
    
    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format(
        (num_rows, num_cols)))

    nz_items, nz_users = valid_ratings.nonzero()

    # split the data
    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = valid_ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

def create_submission(prediction, dat_dir ='../data/', sub_dir ='../submit/'):
    """
    Creates an output file in csv format for submission
    Arguments: prediction: the final matrix with the prediction ratings
    """
    # get the user id and movie id from the sample submission 
    submission = pd.read_csv(dat_dir + "sample_submission.csv")
    
    submission['row'], submission['column'] = submission['Id'].str.split('_', 1).str
    submission['row'] = submission['row'].apply(lambda x: x.replace("r",""))
    submission['column'] = submission['column'].apply(lambda x: x.replace("c",""))
    
    row = submission['row'].tolist()
    column = submission['column'].tolist()
    
    prediction_list = []
    
    # Set predictions above 5 (below 1) to 5 (1)
    prediction[ np.where(prediction > 5.0)] = 5.0
    prediction[ np.where(prediction < 1.0)] = 1.0
    
    # get the prediction value for each (user, movie) pair from the prediction matrix
    for i in range(submission['Id'].count()):
        prediction_list.append(np.round(prediction[int(column[i]) - 1, int(row[i]) - 1]))
        
    prediction_series = pd.Series((j for j in prediction_list) )
    submission['Prediction'] = prediction_series
    
    # create the csv file    
    submission = submission.drop(['row','column'], axis=1)
    submission.to_csv(sub_dir + 'submission.csv', index=False)