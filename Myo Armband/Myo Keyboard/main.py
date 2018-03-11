import random

import numpy as np
from dataReader import separateSets,  get_input_accelero, get_input_gyro, get_input_multiaxis_accelerometer, get_input_multiaxis_gyrometer
import tensorflow as tf

def create_feature_sets_and_labels(test_size=0.1):


    # features, output_data = get_input_gyro('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv',
    #                        './data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv',
    #                        './data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv',
    #                        './data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv',
    #                        './data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv',
    #                        './data/Forward/gyro-1456703940.csv', './data/Backward/gyro-1456704054.csv',
    #                        './data/Left/gyro-1456704106.csv',
    #                        './data/Right/gyro-1456704146.csv', './data/Enter/gyro-1456704184.csv','y')

    # features = get_input_accelero('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv',
    #                               './data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv',
    #                               './data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv',
    #                               './data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv',
    #                               './data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv', 'x')

    features = get_input_multiaxis_gyrometer('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv',
                           './data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv',
                           './data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv',
                           './data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv',
                           './data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv',
                           './data/Forward/gyro-1456703940.csv', './data/Backward/gyro-1456704054.csv',
                           './data/Left/gyro-1456704106.csv',
                           './data/Right/gyro-1456704146.csv', './data/Enter/gyro-1456704184.csv')


    # features = get_input_multiaxis_accelerometer('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv',
    #                        './data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv',
    #                        './data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv',
    #                        './data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv',
    #                        './data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv')

    features = np.array(features)

    # split a portion of the features into tests
    testing_size = int(test_size * len(features))
    # create train and test lists
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])


    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = create_feature_sets_and_labels()


# hidden layers and their nodes
n_nodes_hl1 = 75
n_nodes_hl2 = 75
n_nodes_hl3 = 75
n_nodes_hl4 = 75

# classes in our output
n_classes = 5
# iterations and batch-size to build out model
hm_epochs = 1000
batch_size = 10


x = tf.placeholder('float')
y = tf.placeholder('float',[None, n_classes])


# random weights and bias for our layers
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

hidden_4_layer = {'f_fum': n_nodes_hl4,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl4]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# our predictive model's definition
def neural_network_model(data):
    # hidden layer 1: (data * W) + b
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.sigmoid(l1)

    # hidden layer 2: (hidden_layer_1 * W) + b
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.sigmoid(l2)

    # hidden layer 3: (hidden_layer_1 * W) + b
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.sigmoid(l3)

    # # hidden layer 4: (hidden_layer_1 * W) + b
    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weight']), hidden_4_layer['bias'])
    l4 = tf.sigmoid(l4)

    #output: (hidden_layer_4 * W) + b
    output = tf.matmul(l4, output_layer['weight']) + output_layer['bias']
    return output



# training our model
def train_neural_network(x):
    # use the model definition
    prediction = neural_network_model(x)

    # formula for cost (error)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # optimize for cost using GradientDescent
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

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
            print('prediction expected:',test_y[i])
            output = prediction.eval(feed_dict={x: [test_x[i]]})
            # normalize the prediction values
            print((tf.nn.softmax(output).eval()))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
    return output_weight, output_bias


output_weight, output_bias = train_neural_network(x)