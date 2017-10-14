import numpy as np
import pandas as pd
import random
import math
from sklearn.cross_validation import ShuffleSplit


print '\n'
print "let s go"

def train_submission(module_path, X_df, y_array, train_is):
    # Preparing the training set
    X_train_df = X_df.iloc[train_is]
    y_train_array = y_array[train_is]

    # Feature extraction
    import feature_extractor
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train_df, y_train_array)
    X_train_array = fe.transform(X_train_df)

    # Regressor
    import regressor
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg


def test_submission(trained_model, X_df, test_is):

    # Preparing the test (or valid) set
    X_test_df = X_df.iloc[test_is]

    fe, reg = trained_model

    # Feature extraction
    X_test_array = fe.transform(X_test_df)

    # Regression
    y_pred = reg.predict(X_test_array)
    return y_pred

#np.expm1(price)
#np.log1p

data = pd.read_csv("data/train.csv")

#test remove some high prices
#data[data['SalePrice'] < 600000]

#y_array = np.log1p(data['SalePrice'].values)
y_array = data['SalePrice'].values
X_df = data.drop("SalePrice",1)

skf = ShuffleSplit(y_array.shape[0], n_iter=2, test_size=0.2, random_state=61)
skf_is = list(skf)[0]
train_is, test_is = skf_is

nbtest =1
matrixres = np.zeros(shape=(nbtest,5))
for i in range(nbtest):
    trained_model = train_submission('.', X_df, y_array, train_is)
    print "ok"
    #y_pred_array = np.expm1(test_submission(trained_model, X_df, test_is))
    #ground_truth_array = np.expm1(y_array[test_is])
    y_pred_array = test_submission(trained_model, X_df, test_is)
    ground_truth_array = y_array[test_is]

    score = np.sqrt(np.mean(np.square(ground_truth_array - y_pred_array)))
    score_log = np.sqrt(np.square(np.log(y_pred_array + 1) - np.log(ground_truth_array + 1)).mean())
    print 'RMSE =', score
    print 'RMSLE =', score_log
    
    #Let's play it on the whole dataset
    data = pd.read_csv("data/train.csv")
    X_df = data
    
    fe, reg = trained_model
    # Feature extraction
    X_test_array = fe.transform(X_df)

    # Regression
    y_pred = reg.predict(X_test_array)

    results_train = pd.DataFrame({
        "Id" : data["Id"],
        "Original Saleprice" : data["SalePrice"],
        "Calculated SalePrice" : y_pred,
    })

    # print results["SalePrice"]
    results_train.to_csv("CompareSaleprice.csv", index=False)

    #FOR THE FINAL STAGE
    #now we export the data
    data = pd.read_csv("data/test.csv")
    X_df = data
    
    fe, reg = trained_model
    # Feature extraction
    X_test_array = fe.transform(X_df)

    # Regression
    y_pred = reg.predict(X_test_array)

    results = pd.DataFrame({
        "Id" : data["Id"],
        "SalePrice" : y_pred,
    })

    # print results["SalePrice"]
    results.to_csv("results.csv", index=False)

print "thx fot watching"
print '\n'