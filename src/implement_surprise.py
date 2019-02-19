import surprise
import pandas as pd
import numpy as np
import re
from surprise import SVD
from surprise.model_selection import GridSearchCV

dat_dir = '../data/'

file_path = dat_dir + 'data_train.csv'
ratings = pd.read_csv(file_path)
ratings.head()

# preprocess the data since surprise needs a dataframe with ['User', 'Item', 'Prediction']
r_c = np.array(list(map(lambda x:re.split("[r_c]", x), ratings.Id)))

ratings['User'] = r_c[:,1]
ratings['Item'] = r_c[:,3]

# grid search with cross validation
reader = surprise.Reader(rating_scale=(1, 5))
data = surprise.Dataset.load_from_df(ratings[['User', 'Item', 'Prediction']], reader)

param_grid = {'n_epochs': [30], 'n_factors':[20, 45, 100, 150], 'lr_all': [0.005],
              'reg_pu': [1.0, 0.1, 0.01, 0.001], 'reg_qi': [1.0, 0.1, 0.01, 0.001]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', ], cv=5)
gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Best parameters provided by surprise: {'n_epochs': 30, 'n_factors': 10, 'lr_all': 0.005, 'reg_pu': 0.1, 'reg_qi': 0.01}