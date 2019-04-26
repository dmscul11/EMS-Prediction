
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import struct
import math
import random
import time
from collections import Counter
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


# count occaurances of events
def count_events(project, trials_threshold, use_data, use_exp, use_par):

    # get all event dirs
    dirs = glob.glob(project + 'processed-data/*')
    events = {}

    # count number of files in each event dir
    for d in dirs:

        # get all files in dir and event name
        use_events = ['Administer IM Medication', 'Administer IV Medication', 'Bagging', 'Blood Pressure Cuff', \
                'Intubation', 'IO Placement', 'IV Placement', 'Oral Airway', 'Tie Tourniquet', 'Vital Checking']
        files = glob.glob(d + '/*.csv')
        event = d.split('/')[-1]
        if (event in use_events):

            # only include files in count of the data type, exp, and par
            file_list = []
            for f in files:
                file = f.split('/')[-1]
                tmp = file.split('_')
                exp = tmp[1]
                par = tmp[2]

                # if (exp in use_exp) and (par in use_par) and (use_data[0] in file):
                if (exp in use_exp) and (par in use_par):
                    file_list.append(f)

            # only include events if they are above threshold
            if len(file_list) >= trials_threshold:
                events[event] = len(file_list)

    print(events)
    return events


# read in data into dataframe
def read_in(project, events, use_data):

    # read in data by event folder
    count = 0
    for e in events:
        print(e)

        # read in a certain data type
        for d in use_data:
            files = glob.glob(project + 'processed-data/' + e + '/*' + d + '*.csv')

            for a in files:
                if count == 0:
                    watch_data = pd.read_csv(a, header=0, sep=',', index_col=False)
                    watch_data.insert(0, 'Sample #', count)
                else:
                    # remove column headers then append data
                    data = pd.read_csv(a, header=0, sep=',', index_col=False)
                    data.insert(0, 'Sample #', count)
                    watch_data = watch_data.append(data)

                count = count + 1

    # add index column to individual samples
    watch_data = watch_data.drop(columns=['Unnamed: 0'])

    return watch_data


# normalize data and split into training test sets
def preprocess_data(data, norm):

    # randomize data
    samples = np.asarray(data['Sample #'])
    labels = np.asarray(data['Procedure'])
    data = np.asarray(data.iloc[:, 10:], dtype=np.float32)
    rand_order = list(range(0, data.shape[0]))
    random.shuffle(rand_order)
    data = data[rand_order, :]
    samples = samples[rand_order]
    labels = labels[rand_order]

    # scale data by subtract mean and divide by std
    if norm:
        data_mean = (data.mean(axis=0, keepdims=True)).mean(axis=1, keepdims=True)
        data_std = np.std(np.std(data, axis=0, keepdims=True), axis=1, keepdims=True)
        data = (data - data_mean) / data_std

    # randomize data into training and test
    train_ratio = int(0.8 * data.shape[0])
    random.shuffle(rand_order)
    data = data[rand_order, :]
    samples = samples[rand_order, ]
    labels = labels[rand_order, ]
    x_train = data[0:train_ratio, :]
    x_test = data[train_ratio:data.shape[0], :]
    s_train = samples[0:train_ratio, ]
    s_test = samples[train_ratio:data.shape[0], ]
    y_train = labels[0:train_ratio, ]
    y_test = labels[train_ratio:data.shape[0], ]

    return x_train, y_train, s_train, x_test, y_test, s_test


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
    # {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 4, 'max_features': 'log2'}


def find_best_estimators(training_set, class_set, fit_rf):

    # run model to find optimal # of estimators
    fit_rf.set_params(warm_start=True, oob_score=True)
    min_estimators = 15
    max_estimators = 400
    error_rate = {}
    for i in range(min_estimators, max_estimators + 1):
        print(i)
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
    # n_estimators = 250


def cross_validation(fit_rf, training_set, class_set):

    n = KFold(n_splits=10)
    # scores = cross_val_score(fit_rf, training_set, class_set, cv=n)
    # print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})".format(scores.mean(), scores.std() / 2))
    scores = cross_val_predict(fit_rf, training_set, class_set, cv=n)

    return scores


def visualize(data, x_train, y_train, x_test, y_test, scores):

    # run pca on data
    xdata = data.iloc[:, 10:data.shape[1] + 1]
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(xdata)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1'])

    # write out new dataframe
    pca_data = data.iloc[:, 0:10]
    pca_data['pca 1'] = principalDf
    pca_data['predicted'] = scores
    data['predicted'] = scores
    pca_data.to_csv('PCA.csv', sep=',', header=True, index=True)
    data.to_csv('AllData.csv', sep=',', header=True, index=True)


# create classification csvs
def classify(y_train, s_train, y_test, s_test, scores):

    # combine data
    real_labels = np.hstack((y_train, y_test))
    real_samps = np.hstack((s_train, s_test))
    u_labels = np.unique(real_labels)
    preds = pd.DataFrame(columns=['labels', 'pred', 'sample'])
    acc = pd.DataFrame(columns=['labels', 'pred', 'prob'])
    probs = pd.DataFrame(columns=u_labels)

    # calc label, prediction, and probs for each full sample
    for i in range(0, len(np.unique(real_samps))):
        idx = np.where(real_samps == i)
        label = real_labels[idx][0]
        c = Counter(scores[idx])
        pred, count = c.most_common()[0]
        prob = float(count / len(idx[0]))
        acc.loc[len(acc), :] = [label, pred, prob]

        # for each possible label calc probabililty in sample
        prob_tmp = []
        for j in u_labels:
            jdx = np.where(scores[idx] == j)
            tmp = float(len(jdx[0]) / len(idx[0]))
            prob_tmp.append(tmp)
        probs.loc[len(probs), :] = prob_tmp

    # write to csv
    preds['labels'] = real_labels
    preds['pred'] = scores
    preds['sample'] = real_samps
    preds.to_csv('predictions.csv', sep=',', header=True, index=False)
    acc.to_csv('accuracy.csv', sep=',', header=True, index=False)
    probs.to_csv('probabilities.csv', sep=',', header=True, index=False)


# main function
def main():

    # PARAMETERS TO CHANGE ###
    project = '/Users/deirdre/Documents/DODProject/CELA-Data/NeuralNetwork/'    # path to main directory
    trials_threshold = 0     # 24   # min # of instances of event to include event in analysistesting = 0
    norm = 1
    rand_numb = 13
    testing = 0

    # use any combination of data types but change exp/par: 'AppleWatch', 'Myo_EMG', 'Myo_IMU', 'PatientSpace', 'RawXY'
    use_data = ['Myo_*_EMG']
    use_exp = ['E1', 'E2', 'E3']   # use any combinations of the experiments: 'E1', 'E2', 'E3'
    use_par = ['P1', 'P2', 'P3', 'P4', 'P5']  # use any combination of the participants: 'P1', 'P2', 'P3', 'P4'

    # read in data ###
    events = count_events(project, trials_threshold, use_data, use_exp, use_par)     # use events above threshold
    data = read_in(project, events, use_data)

    # normalize and split data
    x_train, y_train, s_train, x_test, y_test, s_test = preprocess_data(data, norm)
    print("\n Training and Test sizes:")
    print(x_train.shape)
    print(y_train.shape)
    print(s_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(s_test.shape)

    # train fandom forest decision tree
    if testing == 1:
        # find best params for model
        fit_rf = RandomForestClassifier(random_state=rand_numb)
        find_best_params(rand_numb, x_train, y_train, fit_rf)

        # plot for optimal # of estimators
        find_best_estimators(x_train, y_train, fit_rf)
        # {'bootstrap': True, 'criterion': 'gini', 'max_depth': 4, 'max_features': None}

    else:
        # set up final model
        fit_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=4, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=250, n_jobs=1, oob_score=False, random_state=rand_numb,
            verbose=0, warm_start=False)

        # run cross validation
        scores = cross_validation(fit_rf, np.vstack((x_train, x_test)), \
            np.vstack((y_train[:, None], y_test[:, None])).ravel())

        # create classifications output
        classify(y_train, s_train, y_test, s_test, scores)
        print("Done with Classification")

        # create visualizations
        visualize(data, x_train, y_train, x_test, y_test, scores)
        print("Done with Visualization")


main()
