
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib
import csv
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

MW1 = 2.0
MW2 = -14.0
Mb = 18.0 + 0.5 * np.random.randn()

yScale = 10000

train_X = np.linspace(0, 1, 10)
train_Y = MW1 * train_X *train_X + MW2 *train_X + np.random.randn(*train_X.shape) * 0.33 + Mb


# train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float32")
Y = tf.placeholder("float32")

# Create Model

# Set model weights
b = tf.Variable(rng.randn(), name='bias')
W1 = tf.Variable(rng.randn(), name='weight_1')
W2 = tf.Variable(rng.randn(), name='weight_2')

activation = tf.add(tf.multiply(W1, tf.multiply(X, X)), tf.multiply(W2, X))
activation = tf.add(activation, b)


# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)  # L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    count = 0
    test_X = np.linspace(0, 1, 10) + np.random.randn(*train_X.shape) * 0.05
    test_Y = MW1 * test_X * test_X + MW2 * test_X + Mb
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), \
                  "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b), '\n')

    print("Testing... (L2 loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * test_X.shape[0]),
                            feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute l2 loss difference:", abs(training_cost - testing_cost))
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)

    # ax.set_aspect('equal')
    ax.plot(train_X, train_Y, 'ro', label='Original data')
    ax.plot(test_X, test_Y, 'bo', label='Testing data')

    y = np.add(np.multiply(sess.run(W1), np.multiply(train_X, train_X)), np.multiply(sess.run(W2), train_X))
    y = np.add(y, sess.run(b))
    ax.plot(train_X,  y, label='Fitted line')
    ax.legend()
    # plt.show()
    fig.savefig('LinearRegression{:05d}.png'.format(count), bbox_inches='tight', dpi=100)
    count = count + 1
    plt.close(fig)