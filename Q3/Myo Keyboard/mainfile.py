import numpy as np
import random
import tensorflow as tf
import csv
from Interval import Interval
from collections import defaultdict
from bisect import bisect_left


# Global Variables

# ==================================================================================
# separateSets
#   This function separate each of backward, forward, left, right and enter buttons'
#   10 inputs as an interval using "no-change periods"
# ==================================================================================
def separateSets(orientation_path, accelero_path):
    orientation_list = read_in_file(orientation_path)
    accelero_list = read_in_file(accelero_path)
    ori_timestamp_list = find_nochange_periods(orientation_list['timestamp'], orientation_list['x'],
                                               orientation_list['y'], orientation_list['z'])
    peak_value = find_peak(accelero_list['y'])
    intervals = make_intervals(peak_value, ori_timestamp_list, accelero_list)
    return intervals


# ==================================================================================
# read_in_file
#   reads in a .csv file
# ==================================================================================
def read_in_file(path):
    columns = defaultdict(list)  # each value in each column is appended to a list
    with open(path) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                columns[k].append(v)  # append the value into the appropriate list
                # based on column name k
    return columns


# ==================================================================================
# find_nochange_periods
#   find the no_change periods from orientationEuler and orientation .csv files
# ==================================================================================
def find_nochange_periods(timestamp, x_axis, y_axis, z_axis):
    to_return = []
    i = 0
    while i < len(timestamp) - 3:
        if (x_axis[i] == x_axis[i + 1] and y_axis[i] == y_axis[i + 1] and
                z_axis[i] == z_axis[i + 1]):
            to_return.append(timestamp[i])
            i = i + 3
        else:
            i = i + 1
    return to_return


# ==================================================================================
# find_peak
#   returns the lowest out of the 20 top peak y values in accelerometer.
# ==================================================================================
def find_peak(y_axis):
    sorted_list = sorted(y_axis)
    return sorted_list[30]


# ==================================================================================
# make_intervals
#   returns 10 intervals that can be used in EMG
# ==================================================================================
def make_intervals(peak_value, timestamp_list, accelero_list):
    i = 1
    accelero_timestamp = accelero_list['timestamp']
    accelero_y = accelero_list['y']
    to_return = []  # to return the list of 10 intervals( could be more than 10 )
    while i < len(timestamp_list):
        start_index = accelero_timestamp.index(timestamp_list[i - 1])  # finds the matching timestamp in accelerometer
        end_index = accelero_timestamp.index(timestamp_list[i])
        while start_index < end_index:
            if accelero_y[start_index] < peak_value:
                # print( timestamp_list[i-1] +", " + timestamp_list[i])
                to_return.append(Interval(timestamp_list[i - 1], timestamp_list[i]))
                break
            start_index = start_index + 1
        i = i + 1
    return to_return


# ==================================================================================
# extract_emg
#   returns the actual EMG data sets that are separated by the intervals 2D array.
#   For example, emg_column1[0][0] contains the first interval's first EMG data record
# ==================================================================================
def extract_emg(intervals, emg_path, emg_num):
    emg = read_in_file(emg_path)
    emg_timestamp = emg['timestamp']
    emg_column = emg[emg_num]  # specific column within the emg .csv file
    emg_intervals = find_closest_timestamp_interval_list(intervals, emg_timestamp)
    to_return = temp = []
    # now we have the emg_intervals. We actually need the "DATA" sets
    add_data = False
    i = j = 0
    while i < len(emg_column) and j < len(emg_intervals):
        if float(emg_intervals[j].start_time) == float(emg_timestamp[i]):
            temp = []
            add_data = True
        if float(emg_intervals[j].end_time) == float(emg_timestamp[i]):
            add_data = False
            to_return.append(temp)
            j = j + 1
        if add_data:
            temp.append(emg_column[i])
        i = i + 1
    emg_column = np.array(emg_column)
    result = np.zeros(3500)
    result[0:emg_column.shape[0]] = emg_column
    return result


# ==================================================================================
# find_closest_timestamp
#   this function finds the closest timestamp to the given interval within the list
#   and returns the new list of intervals with the start and end time that are
#   corresponding to the given list
# ==================================================================================
def find_closest_timestamp_interval_list(intervals, list):
    i = 0
    to_return = []
    while i < len(intervals):
        # now we have two sorted list for both start time and end time.
        start_time = min(map(float, list), key=lambda x: abs(x - float(intervals[i].start_time)))
        end_time = min(map(float, list), key=lambda x: abs(x - float(intervals[i].end_time)))
        to_return.append(Interval(str(start_time), str(end_time)))
        i = i + 1
    return to_return


#Now that we have 10 intervals(9 or 11), we split into training and testing data. 0~8 will be training and 10 will be testing data

# MAIN EXECUTION
#backward_invervals = separateSets('./data/Backward/orientation-1456704054.csv',
 #                                 './data/Backward/accelerometer-1456704054.csv')
#emg_column1 = extract_emg(backward_invervals, './data/Backward/emg-1456704054.csv', 'emg1')
#print("Printing EMG1 data. 10 intervals' actual data")
#print(emg_column1)



def create_feature_sets_and_labels(test_size=0.1):
    forward_intervals = separateSets('./data/Forward/orientation-1456703940.csv',
                                     './data/Forward/accelerometer-1456703940.csv')
    backward_intervals = separateSets('./data/Backward/orientation-1456704054.csv',
                                      './data/Backward/accelerometer-1456704054.csv')
    right_intervals = separateSets('./data/Right/orientation-1456704146.csv',
                                   './data/Right/accelerometer-1456704146.csv')
    left_intervals = separateSets('./data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv')
    enter_intervals = separateSets('./data/Enter/orientation-1456704184.csv',
                                   './data/Enter/accelerometer-1456704184.csv')

    forward_emg_column1 = extract_emg(forward_intervals, './data/Forward/emg-1456703940.csv', 'emg1')
    forward_emg_column2 = extract_emg(forward_intervals, './data/Forward/emg-1456703940.csv', 'emg2')
    forward_emg_column3 = extract_emg(forward_intervals, './data/Forward/emg-1456703940.csv', 'emg3')
    forward_emg_column4 = extract_emg(forward_intervals, './data/Forward/emg-1456703940.csv', 'emg4')

    backward_emg_column1 = extract_emg(backward_intervals, './data/Backward/emg-1456704054.csv', 'emg1')
    backward_emg_column2 = extract_emg(backward_intervals, './data/Backward/emg-1456704054.csv', 'emg2')
    backward_emg_column3 = extract_emg(backward_intervals, './data/Backward/emg-1456704054.csv', 'emg3')
    backward_emg_column4 = extract_emg(backward_intervals, './data/Backward/emg-1456704054.csv', 'emg4')

    left_emg_column1 = extract_emg(left_intervals, './data/Left/emg-1456704106.csv', 'emg1')
    left_emg_column2 = extract_emg(left_intervals, './data/Left/emg-1456704106.csv', 'emg2')
    left_emg_column3 = extract_emg(left_intervals, './data/Left/emg-1456704106.csv', 'emg3')
    left_emg_column4 = extract_emg(left_intervals, './data/Left/emg-1456704106.csv', 'emg4')

    right_emg_column1 = extract_emg(right_intervals, './data/Right/emg-1456704146.csv', 'emg1')
    right_emg_column2 = extract_emg(right_intervals, './data/Right/emg-1456704146.csv', 'emg2')
    right_emg_column3 = extract_emg(right_intervals, './data/Right/emg-1456704146.csv', 'emg3')
    right_emg_column4 = extract_emg(right_intervals, './data/Right/emg-1456704146.csv', 'emg4')

    enter_emg_column1 = extract_emg(enter_intervals, './data/Enter/emg-1456704184.csv', 'emg1')
    enter_emg_column2 = extract_emg(enter_intervals, './data/Enter/emg-1456704184.csv', 'emg2')
    enter_emg_column3 = extract_emg(enter_intervals, './data/Enter/emg-1456704184.csv', 'emg3')
    enter_emg_column4 = extract_emg(enter_intervals, './data/Enter/emg-1456704184.csv', 'emg4')


    features = []
    features.append([forward_emg_column1, [1, 0, 0, 0, 0]])
    features.append([backward_emg_column1, [0, 1, 0, 0, 0]])
    features.append([left_emg_column1, [0, 0, 1, 0, 0]])
    features.append([right_emg_column1,[0, 0, 0, 1, 0]])
    features.append([enter_emg_column1,[0, 0, 0, 0, 1]])

    # features.append([forward_emg_column2, [1, 0, 0, 0, 0]])
    # features.append([backward_emg_column2, [0, 1, 0, 0, 0]])
    # features.append([left_emg_column2, [0, 0, 1, 0, 0]])
    # features.append([right_emg_column2, [0, 0, 0, 1, 0]])
    # features.append([enter_emg_column2, [0, 0, 0, 0, 1]])
    #
    # features.append([forward_emg_column3, [1, 0, 0, 0, 0]])
    # features.append([backward_emg_column3, [0, 1, 0, 0, 0]])
    # features.append([left_emg_column3, [0, 0, 1, 0, 0]])
    # features.append([right_emg_column3, [0, 0, 0, 1, 0]])
    # features.append([enter_emg_column3, [0, 0, 0, 0, 1]])
    #
    # features.append([forward_emg_column4, [1, 0, 0, 0, 0]])
    # features.append([backward_emg_column4, [0, 1, 0, 0, 0]])
    # features.append([left_emg_column4, [0, 0, 1, 0, 0]])
    # features.append([right_emg_column4, [0, 0, 0, 1, 0]])
    # features.append([enter_emg_column4, [0, 0, 0, 0, 1]])

    # shuffle out features and turn into np.array
    random.shuffle(features)
    features = np.array(features)

    # split a portion of the features into tests
    testing_size = int(1)#test_size * len(features))

    # create train and test lists
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = create_feature_sets_and_labels()

# hidden layers and their nodes
n_nodes_hl1 = 8
n_nodes_hl2 = 8

# classes in our output
n_classes = 5
# iterations and batch-size to build out model
hm_epochs = 1000
batch_size = 4

x = tf.placeholder('float')
y = tf.placeholder('float')

# random weights and bias for our layers
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# our predictive model's definition
def neural_network_model(data):
    # hidden layer 1: (data * W) + b
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.sigmoid(l1)

    # hidden layer 2: (hidden_layer_1 * W) + b
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.sigmoid(l2)

    # output: (hidden_layer_2 * W) + b
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output


# training our model
def train_neural_network(x):
    # use the model definition
    prediction = neural_network_model(x)

    # formula for cost (error)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # optimize for cost using GradientDescent
    optimizer = tf.train.GradientDescentOptimizer(1).minimize(cost)

    # Tensorflow session
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('log_ANN_graph', sess.graph)
        # initialize our variables
        sess.run(tf.global_variables_initializer())

        # loop through specified number of iterations
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            # handle batch sized chunks of training data
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
                last_cost = c

            # print cost updates along the way
            if (epoch % (hm_epochs / 5)) == 0:
                print('Epoch', epoch, 'completed out of', hm_epochs, 'cost:', last_cost)

        # print accuracy of our model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

        output_weight = sess.run(output_layer['weight'])
        output_bias = sess.run(output_layer['bias'])

        # print predictions using our model
        for i, t in enumerate(test_x):
            print('prediction for:', test_x[i])
            output = prediction.eval(feed_dict={x: [test_x[i]]})
            # normalize the prediction values
            print(tf.sigmoid(output[0][0]).eval(), tf.sigmoid(output[0][1]).eval(), tf.sigmoid(output[0][2]).eval(), tf.sigmoid(output[0][3]).eval(), tf.sigmoid(output[0][4]).eval())

        return output_weight, output_bias


output_weight, output_bias = train_neural_network(x)