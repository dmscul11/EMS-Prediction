
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import struct
import math
import random


# count occaurances of events
def count_events(project, trials_threshold, use_data, use_exp, use_par):

    # get all event dirs
    dirs = glob.glob(project + 'processed-data/*')
    events = {}

    # count number of files in each event dir
    for d in dirs:

        # get all files in dir and event name
        files = glob.glob(d + '/*.csv')
        event = d.split('/')[-1]

        # only include files in count of the data type, exp, and par
        file_list = []
        for f in files:
            file = f.split('/')[-1]
            tmp = file.split('_')
            exp = tmp[1]
            par = tmp[2]

            if (exp in use_exp) and (par in use_par) and (use_data[0] in file):
                file_list.append(f)

        # only include events if they are above threshold
        if len(file_list) >= trials_threshold:
            events[event] = len(file_list)

    print(events)
    return events


# read in data into arrays
def read_in(project, events, use_data, use_exp, use_par, max_rows):

    # read in data by event folder
    count = 0
    data_all = []
    events_all = []
    for e in events:

        # get each experiment included
        for exp in use_exp:

            # get each participant included
            for par in use_par:
                files = glob.glob(project + 'processed-data/' + e + '/*' + exp + '_' + par + '*.csv')

                # if data exists
                if files != []:
                    # get unique instances
                    instances = []
                    for f in files:
                        instance = f.split('/')[-1].split('_')[3]
                        if instance not in instances:
                            instances.append(instance)

                    # for each instance of event
                    for i in instances:
                        count = count + 1

                        # for each data type in included get files
                        data_list = []
                        for d in use_data:
                            files = glob.glob(project + 'processed-data/' + e + '/*' + exp + '_' + \
                                par + '_' + i + '*' + d + '*.csv')
                            data_list.append(files)

                        # combine all data types and files
                        data = combine_data_types(use_data, data_list, count)
                        events_all.append(e)

                        # pad data with zeros so every sample has same size and save to csv
                        diff = max_rows - data.shape[0]
                        data = np.array(np.append(data, np.zeros((diff, data.shape[1])), axis=0))
                        np.savetxt(project + 'combined-data/' + e + '_' + exp + '_' + par + '_' + i + '_Combined.csv', \
                            data, delimiter=',')

                        # combine data into one matrix
                        if data_all == []:
                            data_all = np.array(data)
                        else:
                            data_all = np.array(np.dstack((data_all, data)))

    return data_all, events_all


# combine all data types data into one
def combine_data_types(use_data, data_list, sample):

    # static variable do not change
    info_cols = 10  # number of first info cols

    # for each data type included
    count = 0
    data_combined = []
    for i, dt in enumerate(use_data):

        # make sure correct number of cols are inserted for data types, pad 0 otherwise
        if dt == 'AppleWatch':
            l = ['Left', 'Right']
            cols = 9
        elif dt == 'EMG':
            l = ['Left', 'Right']
            cols = 8
        elif dt == 'IMU':
            l = ['Left', 'Right']
            cols = 14
        elif dt == 'PatientSpace':
            l = ['Camera']
            cols = 72
        elif dt == 'RawXY':
            l = ['Camera']
            cols = 72

        # for each file of the data type
        for j, f in enumerate(l):
            try:
                # read in data, drop unnecessary cols
                data = pd.read_csv(data_list[i][j], header=0, sep=',', index_col=False)

                # create data matrix
                if j == 0:
                    data_final = np.array(data.iloc[:, info_cols:])
                    curr_secs = data['seconds']

                # join new data with data matrix
                else:
                    data.drop(data.columns[0:info_cols], axis=1, inplace=True)

                    # make sizes match by padding with 0s
                    if data.shape[0] < data_final.shape[0]:
                        diff = data_final.shape[0] - data.shape[0]
                        data = np.array(np.append(data, np.zeros((diff, data.shape[1])), axis=0))
                        curr_secs = data_final['seconds']
                    elif data.shape[0] > data_final.shape[0]:
                        diff = data.shape[0] - data_final.shape[0]
                        data_final = np.array(np.append(data_final, np.zeros((diff, data_final.shape[1])), axis=0))
                        curr_secs = data['seconds']
                    data_final = np.array(np.append(data_final, data, axis=1))

            # if no file for type pad with 0s
            except:
                # add zero data of some size, will be combined anyways
                if j == 0:
                    data_final = np.zeros((3, cols))
                    curr_secs = np.zeros((3, 1))

                # add zero data of same size
                else:
                    data = np.zeros(data_final.shape)
                    data_final = np.array(np.append(data_final, data, axis=1))

            count = count + 1

        # combine data types
        if data_combined == []:
            data_combined = np.array(data_final)
            secs = np.array(curr_secs)
        else:

            # pad with zeros if no data to append
            if (data_final == 0).all():
                data_combined = np.array(np.append(data_combined, np.zeros((data_combined.shape[0], data_final.shape[1])), axis=1))
            elif (data_combined == 0).all():
                data_combined = np.array(np.append(np.zeros((data_final.shape[0], data_combined.shape[1])), data_final, axis=1))
                secs = np.array(curr_secs)

            # join by seconds timepoints otherwise
            else:

                # use right table timepoints
                if data_final.shape[0] > data_combined.shape[0]:
                    tmp_data = np.zeros((data_final.shape[0], data_combined.shape[1] + data_final.shape[1]))

                    # loop through timepoints and find mathcing
                    iterator = 0
                    for t, tp in enumerate(curr_secs):
                        tmp_data[t, data_combined.shape[1]:] = data_final[t, :]
                        if iterator < len(secs):
                            curr_diff = abs(tp - secs[iterator])
                            if t + 1 < data_final.shape[0]:
                                next_diff = abs(curr_secs[t + 1] - secs[iterator])
                            else:
                                next_diff = curr_diff + 10

                        # use closest timepoint row
                        if curr_diff <= next_diff:
                            if iterator < data_combined.shape[0]:
                                tmp_data[t, 0:data_combined.shape[1]] = data_combined[iterator, :]
                                iterator = iterator + 1
                            else:
                                tmp_data[t, 0:data_combined.shape[1]] = np.zeros((1, data_combined.shape[1]))
                        else:
                            tmp_data[t, 0:data_combined.shape[1]] = np.zeros((1, data_combined.shape[1]))

                    # update secs col
                    secs = np.array(curr_secs)

                # use left table timepoints
                elif data_final.shape[0] <= data_combined.shape[0]:
                    tmp_data = np.empty((data_combined.shape[0], data_combined.shape[1] + data_final.shape[1]))

                    # loop through timepoints and find mathcing
                    iterator = 0
                    for t, tp in enumerate(secs):
                        tmp_data[t, 0:data_combined.shape[1]] = data_combined[t, :]
                        if iterator < len(curr_secs):
                            curr_diff = abs(tp - curr_secs[iterator])
                            if t + 1 < data_combined.shape[0]:
                                next_diff = abs(secs[t + 1] - curr_secs[iterator])
                            else:
                                next_diff = curr_diff + 10

                        # use closest timepoint row
                        if curr_diff <= next_diff:
                            if iterator < data_final.shape[0]:
                                tmp_data[t, data_combined.shape[1]:] = data_final[iterator, :]
                                iterator = iterator + 1
                            else:
                                tmp_data[t, data_combined.shape[1]:] = np.zeros((1, data_final.shape[1]))
                        else:
                            tmp_data[t, data_combined.shape[1]:] = np.zeros((1, data_final.shape[1]))

                data_combined = np.array(tmp_data)

    return data_combined


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # regularization
        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        regularizer = tf.contrib.layers.l1_regularizer(scale=0.001)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu, kernel_regularizer=regularizer)
        # conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # conv1 = tf.layers.average_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        # conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        # conv3 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        # conv4 = tf.layers.conv2d(conv3, 64, 1, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv4 = tf.layers.max_pooling2d(conv4, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv1)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, \
    #   use_locking=False, name='Momentum', use_nesterov=False)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.999, epsilon=1e-10, \
    #   centered=False, momentum=0.9, use_locking=False, name='RMSProp')
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# normalize data and split into training test sets
def preprocess_data(data, events):

    # randomize data
    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(events)
    rand_order = list(range(0, data.shape[0], 1))
    random.shuffle(rand_order)
    data = data[rand_order, :, :]
    labels = labels[rand_order, ]

    # scale data by subtract mean and divide by std
    data_mean = (data.mean(axis=0, keepdims=True)).mean(axis=1, keepdims=True)
    data_std = np.std(np.std(data, axis=0, keepdims=True), axis=1, keepdims=True)
    data = (data - data_mean) / data_std

    # reshape data
    data = data.reshape((-1, data.shape[1], data.shape[2], 1))

    # randomize data into training and test
    train_ratio = int(0.8 * data.shape[0])
    random.shuffle(rand_order)
    data = data[rand_order, :, :, :]
    labels = labels[rand_order, ]
    x_train = data[0:train_ratio, :, :, :]
    x_test = data[train_ratio:data.shape[0], :, :, :]
    y_train = labels[0:train_ratio, ]
    y_test = labels[train_ratio:data.shape[0], ]

    return x_train, y_train, x_test, y_test


# main function
def main():

    # PARAMETERS TO CHANGE ###
    project = '/Users/deirdre/Documents/DODProject/CELA-Data/NeuralNetwork/'    # path to main directory
    trials_threshold = 1   # min # of instances of event to include event in analysis
    max_rows = 121274
    # use any combination of data types but change exp/par: 'AppleWatch', 'Myo_EMG', 'Myo_IMU', 'PatientSpace', 'RawXY'
    use_data = ['AppleWatch', 'EMG', 'IMU', 'PatientSpace', 'RawXY']
    use_exp = ['E2']   # use any combinations of the experiments: 'E1', 'E2', 'E3'
    use_par = ['P3', 'P4']  # use any combination of the participants: 'P1', 'P2', 'P3', 'P4'

    # PREPROCESSING DATA ###
    events = count_events(project, trials_threshold, use_data, use_exp, use_par)     # use events above threshold
    data_combined, events_all = read_in(project, events, use_data, use_exp, use_par, max_rows)   # read in/combine data types into instances
    print(data_combined.shape)
    print(len(events_all))
    x_train, y_train, x_test, y_test = preprocess_data(data_combined, events_all)    # normalize, randomize, split train/test
    print("\n Training and Test sizes:")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # SET UP NEURAL NETWORK ###

    # Set up CNN hyperparameters
    global batch_size, num_epochs, learning_rate, num_classes, dropout
    batch_size = 8     # 32, 64, 128, 256, 512
    num_epochs = 12
    learning_rate = 0.001   # learning rate - investigate impact
    num_classes = len(np.unique(events_all))     # classes
    dropout = 0.01  # Dropout, probability to drop a unit

    # Set up hyperparameters
    n_batches = math.ceil(x_train.shape[0] / batch_size)
    iterations = n_batches * num_epochs

    # RUN NEURAL NETWORK ###

    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    # Define the input function for training
    input_train = tf.estimator.inputs.numpy_input_fn(x={'images': x_train}, y=y_train, \
        batch_size=batch_size, num_epochs=num_epochs, shuffle=True)

    # Train the Model
    model.train(input_train, steps=None)

    # Evaluate the Model - Define the input function for evaluating
    input_test = tf.estimator.inputs.numpy_input_fn(x={'images': x_test}, y=y_test, \
        batch_size=batch_size, shuffle=False)

    # print final results
    e_train = model.evaluate(input_train)
    e_test = model.evaluate(input_test)
    print("Training Accuracy:", e_train['accuracy'])
    print("Testing Accuracy:", e_test['accuracy'])

main()
