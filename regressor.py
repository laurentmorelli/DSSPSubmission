from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression

from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
 

class Regressor(BaseEstimator):
    def __init__(self):
        
        #GRADIENT BOOOST
        self.clf = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
                                             max_depth=40, min_samples_split=30,
                                             loss='ls')
        #-> 32982.0783185
        
        #RANDOM FOREST
        #self.clf = RandomForestRegressor(n_estimators=1000, max_depth=40, max_features=600)
        #-> 32700.0678043

        #NEURAL NETWORK
        #self.clf = MLPRegressor(hidden_layer_sizes=100)
        #-> 33067.3718781
        
        #LINEAR REGRESSOR
        #self.clf = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)
        #-> 3853067.31481

        #LOGISTIC REGRESSOR
        self.clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', 
            max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        #-> 100091.541589

        #BAYESIAN RIDGE
        self.clf = BayesianRidge(n_iter=400, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, 
            lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, 
            copy_X=True, verbose=False)
        #-> 45178.6580026

        #Bayesian ARD regression
        self.clf = ARDRegression(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, 
            lambda_1=1e-06, lambda_2=1e-06, compute_score=False, 
            threshold_lambda=10000.0, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
        #-> 81561.3385945

    def newRandomRandomforest(self, n_esti, max_de, max_fe, min_spi):
        #self.clf = LinearRegression()

        self.clf = RandomForestRegressor(
            n_estimators=n_esti, max_depth=max_de, max_features=max_fe)
        #self.clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    def newRandomGradientBoostingRegressor(self,n_esti,learn_r,max_d,min_sam):
        self.clf = GradientBoostingRegressor(n_estimators=n_esti, learning_rate=learn_r,
                                             max_depth=max_d, min_samples_split=min_sam,
                                             loss='ls')
        #n_estimators=500, learning_rate=0.1,max_depth=5, min_samples_split = 30, loss= 'ls'
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
