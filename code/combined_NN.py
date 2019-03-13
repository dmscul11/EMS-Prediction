
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import glob
import math
import random


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
    rand_order = list(range(0, data.shape[2], 1))
    random.shuffle(rand_order)
    data = data[:, :, rand_order]
    labels = labels[rand_order]

    # scale data by subtract mean and divide by std
    data_mean = (data.mean(axis=0, keepdims=True)).mean(axis=1, keepdims=True)
    data_std = np.std(np.std(data, axis=0, keepdims=True), axis=1, keepdims=True)
    data = (data - data_mean) / data_std

    # reshape data
    data = data.reshape((-1, data.shape[0], data.shape[1], 1))

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


# read in any data from combined
def read_data(project, max_rows):

    # get all files in combined dir
    files = glob.glob(project + 'combined-data/' + '*_Combined.csv')

    # read in and combine data
    data = []
    events = []
    for i, f in enumerate(files):
        ######################### remove #########################
        if i < 3:
            data_tmp = pd.read_csv(f, header=None, sep=',', index_col=False)
            event = f.split('/')[-1].split('_')[0]
            events.append(event)

            # pad to one size
            if max_rows > 0:
                diff = max_rows - data_tmp.shape[0]
                data_tmp = np.array(np.append(data_tmp, np.zeros((diff, data_tmp.shape[1])), axis=0))

            # combine data
            if data == []:
                data = np.array(data_tmp)
            else:
                data = np.array(np.dstack((data, data_tmp)))

    return data, events


# main function
def main():

    # PARAMETERS TO CHANGE ###
    project = '/Users/deirdre/Documents/DODProject/CELA-Data/NeuralNetwork/'    # path to main directory
    max_rows = 121274   # max # rows to impute all data to same size, do not pad if 0

    # PREPROCESSING DATA ###
    data_combined, events_all = read_data(project, max_rows)
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
