#following https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
import numpy as np
import pandas as pd
import random
import math
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

data = pd.read_csv("data/train.csv")

#test remove some high prices
#data[data['SalePrice'] < 600000]
 ### Let's remove unwanted values
mostWantedIds = [980,
1091,
1027,
672,
852,
439,
1255,
444,
964,
395]


def toduplicate_row(row):
    if row.Id in mostWantedIds:
        return True
    else :
        return False

data2 = data[data.apply(toduplicate_row, axis=1)]
data.append([data2]*5,ignore_index=True)

#y_array = np.log1p(data['SalePrice'].values)
y_array = data['SalePrice'].values
X_df = data.drop("SalePrice",1)
# Feature extraction
import feature_extractor
fe = feature_extractor.FeatureExtractor()
X_df = fe.transform(X_df)

print "OK"
'''
param_test1 = {'n_estimators':range(10,210,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(
    learning_rate=0.1, 
    min_samples_split=500,
    min_samples_leaf=50,
    max_depth=8,
    max_features='sqrt',
    subsample=0.8,
    random_state=10), 
    param_grid = param_test1)
gsearch1.fit(X_df,y_array)
print gsearch1.grid_scores_
print gsearch1.best_params_
print gsearch1.best_score_
'''
#OK WE CHOOSE n_estimators = 200
"""
param_test2 = {'max_depth':range(20,50,3), 'min_samples_split':range(100,300,20)}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(
    learning_rate=0.1,
     n_estimators=200,
      max_features='sqrt',
       subsample=0.8,
        random_state=10), 
    param_grid = param_test2)
gsearch2.fit(X_df,y_array)
print gsearch2.grid_scores_
print gsearch2.best_params_
print gsearch2.best_score_
"""
#OK we get max_depth = 38 & min_samples_split = 160
"""
param_test3 = {'min_samples_split':range(130,180,10), 'min_samples_leaf':range(10,30,3)}
gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(
    learning_rate=0.1,
     n_estimators=200,
     max_depth=38,
     max_features='sqrt',
      subsample=0.8,
       random_state=10), 
param_grid = param_test3)
gsearch3.fit(X_df,y_array)
print gsearch3.grid_scores_
print gsearch3.best_params_
print gsearch3.best_score_
"""

#OK we get min_samples_split = 160 & min_samples_leaf = 16

"""
param_test4 = {'max_features':range(20,50,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingRegressor(
    learning_rate=0.1, 
    n_estimators=200,
    max_depth=38, 
    min_samples_split=160,
     min_samples_leaf=16,
      subsample=0.8,
       random_state=10),
param_grid = param_test4)
gsearch4.fit(X_df,y_array)
print gsearch4.grid_scores_
print gsearch4.best_params_
print gsearch4.best_score_
"""
#OK WE GET max_features = 28

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1]}
gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(
    learning_rate=0.1,
     n_estimators=200,
     max_depth=38,
     min_samples_split=160,
      min_samples_leaf=16,
       random_state=10,
       max_features=28),
param_grid = param_test5)
gsearch5.fit(X_df,y_array)
print gsearch5.grid_scores_
print gsearch5.best_params_
print gsearch5.best_score_

#OK WE GET subsample = 0.9