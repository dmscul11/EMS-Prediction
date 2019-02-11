
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def read_data(file):

    # read in input file
    limbs_data = pd.read_csv(file)
    limbs_data.fillna(0, inplace=True)

    # test data
    print(limbs_data.head)
    print(limbs_data.describe())
    print(limbs_data.dtypes)
    print(limbs_data.shape)

    return limbs_data


def preprocess_data(limbs_data, rand_numb):

    # separate features from classes
    feature_space = limbs_data.iloc[:, limbs_data.columns != 'Event']
    feature_class = limbs_data.iloc[:, limbs_data.columns == 'Event']

    # split data randomly
    training_set, test_set, class_set, test_class_set = train_test_split(feature_space, \
        feature_class, test_size=0.20, random_state=rand_numb)
    class_set = class_set.values.ravel()
    test_class_set = test_class_set.values.ravel()

    return training_set, test_set, class_set, test_class_set


def find_best_params(rand_numb, training_set, class_set, fit_rf):

    # set up random forest params
    start = time.time()
    param_dist = {'max_depth': [2, 3, 4], 'bootstrap': [True, False], \
        'max_features': ['auto', 'sqrt', 'log2', None], 'criterion': ['gini', 'entropy']}
    cv_rf = GridSearchCV(fit_rf, cv=10, param_grid=param_dist, n_jobs=3)
    cv_rf.fit(training_set, class_set)

    # run model to find best params
    print('Best Parameters using grid search: \n', cv_rf.best_params_)
    end = time.time()
    print('Time taken in grid search: {0: .2f}'.format(end - start))
    # {'bootstrap': False, 'criterion': 'gini', 'max_depth': 4, 'max_features': 'auto'}


def find_best_estimators(training_set, class_set, fit_rf):

    # run model to find optimal # of estimators
    fit_rf.set_params(warm_start=True, oob_score=True)
    min_estimators = 15
    max_estimators = 1000
    error_rate = {}
    for i in range(min_estimators, max_estimators + 1):
        fit_rf.set_params(n_estimators=i)
        fit_rf.fit(training_set, class_set)
        oob_error = 1 - fit_rf.oob_score_
        error_rate[i] = oob_error

    # plot model to decide optimal # of estimators
    oob_series = pd.Series(error_rate)
    fig, ax = plt.subplots(figsize=(10, 10))
    oob_series.plot(kind='line', color='red')
    plt.xlabel('n_estimators')
    plt.ylabel('OOB Error Rate')
    plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')
    plt.show()
    # n_estimators = 200


def cross_validation(fit_rf, training_set, class_set):

    n = KFold(n_splits=10)
    scores = cross_val_score(fit_rf, training_set, class_set, cv=n)
    print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})".format(scores.mean(), scores.std() / 2))

    return scores.mean(), scores.std() / 2


def main():

    # edit experiment information
    experiment = '2'
    camera = '2'
    data_type = 'LimbCountsDists-thresh'     # 'LimbDistances'
    testing_flag = 0
    file = open('/Users/deirdre/Documents/DODProject/CELA-Data/heatmaps/heatmaps/' \
        + 'Experiment_' + experiment + '/C' + camera + '-' + data_type + '.csv')

    # read in data
    limbs_data = read_data(file)

    # set up random forest model
    rand_numb = 13
    training_set, test_set, class_set, test_class_set = preprocess_data(limbs_data, rand_numb)

    if testing_flag:

        # find best params for model
        fit_rf = RandomForestClassifier(random_state=rand_numb)
        find_best_params(rand_numb, training_set, class_set, fit_rf)

        # plot for optimal # of estimators
        find_best_estimators(training_set, class_set, fit_rf)

    else:

        # set up final model
        fit_rf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=3, max_features='log2', max_leaf_nodes=None,
            min_impurity_decrease=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=250, n_jobs=1, oob_score=False, random_state=rand_numb,
            verbose=0, warm_start=False)

        # run cross validation
        accuracy_mean, accuracy_std = cross_validation(fit_rf, training_set, class_set)


main()
