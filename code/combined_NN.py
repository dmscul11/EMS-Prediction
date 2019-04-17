
# references
# https://github.com/aymericdamien/TensorFlow-Examples
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import glob
import random
import time


# Create model
def conv_net(x, dropout, num_classes):
    # Reshape to match picture format [Height x Width]
    # Tensor input become 3-D: [Batch Size, Height, Width]
    x = tf.reshape(x, shape=[-1, height, width])
    print(x.shape)

    # conv 1 layer
    # conv1 = tf.layers.conv1d(x, activation=tf.nn.relu, filters=1, kernel_size=width)
    conv1 = tf.layers.conv1d(x, activation=tf.nn.relu, filters=2, kernel_size=3)
    print(conv1.shape)

    # 1-layer LSTM with n_hidden units.
    # rnn_cell = rnn.BasicLSTMCell(width)
    rnn_cell = rnn.BasicLSTMCell(5)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, conv1, dtype=tf.float32)
    print(outputs.shape)

    # Output layer, class prediction
    out = tf.layers.dense(outputs, 1)
    print(out.shape)
    out1 = tf.layers.dense(tf.squeeze(out, axis=2), 1)
    print(out1.shape)
    out2 = tf.layers.dense(out1, num_classes)
    print(out2.shape)    

    return out2

    # # Convolution Layer
    # conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # # Max Pooling (down-sampling)
    # conv1 = maxpool2d(conv1, k=2)

    # # Convolution Layer
    # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=2)

    # # Fully connected layer
    # # Reshape conv2 output to fit fully connected layer input
    # fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # fc1 = tf.nn.relu(fc1)
    # # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    # # Output, class prediction
    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # return out


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
        # remove for testing
        # if i < 10:
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

    # convert events to ints from strs
    events = np.array(pd.Series(events).astype('category').cat.codes)

    return data, events


# main function
def main():

    # time script
    start_time = time.time()

    # PARAMETERS TO CHANGE ###
    project = '/home/scullydm/DoDHandsFree/'    # path to main directory
    max_rows = 0   # 121456 max # rows to impute all data to same size, do not pad if 0 when already imputed

    # PREPROCESSING DATA ###
    data_combined, events_all = read_data(project, max_rows)
    x_train, y_train, x_test, y_test = preprocess_data(data_combined, events_all)    # normalize, randomize, split train/test
    print("\n Training and Test sizes:")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # SET UP NEURAL NETWORK ###

    # Training Parameters
    global height, width
    height = x_train.shape[1]
    width = x_train.shape[2]
    learning_rate = 0.001
    num_steps = 100     # 500
    batch_size = 128      # 64
    display_step = 10

    # use GPU if available
    # with tf.device('/device:GPU:0'):

    # Network Parameters
    num_input = int(height * width)     # data input (img shape: rows x cols)
    num_classes = len(np.unique(events_all))    # total classes
    dropout = 0.75  # Dropout, probability to keep units

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Construct model
    logits = conv_net(X, keep_prob, num_classes)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # RUN NEURAL NETWORK ###

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # randomize batch
        for step in range(1, num_steps + 1):
            print("Training Step: " + str(step))
            rand_order = list(range(0, batch_size, 1))
            random.shuffle(rand_order)
            batch_x = np.reshape(x_train[rand_order, :, :], [batch_size, height * width])
            batch_y = np.zeros((batch_size, num_classes))
            batch_y[np.arange(batch_size), y_train[rand_order]] = 1

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy test images
        y_test_out = np.zeros((len(y_test), num_classes))
        y_test_out[np.arange(len(y_test)), y_test] = 1
        print("Testing Accuracy:", sess.run(accuracy, \
            feed_dict={X: np.reshape(x_test, [x_test.shape[0], height * width]), \
            Y: y_test_out, keep_prob: 1.0}))

    # print timing
    end_time = time.time()
    print(end_time - start_time)


main()
