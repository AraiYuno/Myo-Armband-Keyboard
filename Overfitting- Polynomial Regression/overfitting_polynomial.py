import csv

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

rng = np.random

train_X = []
train_Y = []

test_X = []
test_Y = []

# Parameters
learning_rate = 0.5
training_epochs = 4000
display_step = 50

MW = 2.0
Mb = 0.5

yScale = 10000

with open('TrainingData.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        train_X.append(row[0])
        train_Y.append(row[1])

with open('TestingData.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_X.append(row[0])
        test_Y.append(row[1])

train_X = np.array(train_X, dtype='f')
train_Y = np.array(train_Y, dtype='f')
test_X = np.array(test_X, dtype='f')
test_Y = np.array(test_Y, dtype='f')


n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float32")
Y = tf.placeholder("float32")

# Create Model

# Set model weights

for order in range(2, 7):

    b = tf.Variable(rng.randn(), name='bias')
    W1 = tf.Variable(rng.randn(), name='weight_1')
    W2 = tf.Variable(rng.randn(), name='weight_2')
    W3 = tf.Variable(rng.randn(), name='weight_3')
    W4 = tf.Variable(rng.randn(), name='weight_4')
    W5 = tf.Variable(rng.randn(), name='weight_5')

    activation = tf.add(b, 0.0)
    for pow_i in range(1, order):
        if pow_i == 1:
            W = W1
        if pow_i == 2:
            W = W2
        if pow_i == 3:
            W = W3
        if pow_i == 4:
            W = W4
        if pow_i == 5:
            W = W5
        activation = tf.add(tf.multiply(tf.pow(X, pow_i), W), activation)

    # Minimize the squared errors
    cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)  # L2 loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        count = 0
        print("\n\nOrder ", order - 1)
        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

        print("\nOptimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

        if order == 2:
            print("Training cost=", training_cost, "W1=", sess.run(W1), "b=", sess.run(b))
            y = sess.run(W1) * test_X + sess.run(b)

        if order == 3:
            print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b))
            y = sess.run(W2) * np.power(test_X, 2) + sess.run(W1) * test_X + sess.run(b)

        if order == 4:
            print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "W3=", sess.run(W3), "b=", sess.run(b))
            y = sess.run(W3) * np.power(test_X, 3) + sess.run(W2) * np.power(test_X, 2) + sess.run(W1) * test_X + sess.run(b)

        if order == 5:
            print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "W3=", sess.run(W3), "W4=", sess.run(W4), "b=", sess.run(b))
            y = sess.run(W4) * np.power(test_X, 4) + sess.run(W3) * np.power(test_X, 3) + sess.run(W2) * np.power(test_X, 2) + sess.run(W1) * test_X + sess.run(b)

        if order == 6:
            print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "W3=", sess.run(W3), "W4=", sess.run(W4), "W5=", sess.run(W5), "b=", sess.run(b))
            y = sess.run(W5) * np.power(test_X, 5) + sess.run(W4) * np.power(test_X, 4) + sess.run(W3) * np.power(test_X, 3) + sess.run(W2) * np.power(test_X, 2) + sess.run(W1) * test_X + sess.run(b)

        print("Testing... (L2 loss Comparison)")
        print("Testing... (Mean square loss Comparison)")
        testing_cost = sess.run(
            tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * len(test_X)),
            feed_dict={X: test_X, Y: test_Y})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(test_X, test_Y, 'bo', label='Testing data')
        ax.plot(train_X, train_Y, 'ro', label='Training data')

        ax.plot(test_X, y, label='Fitted line')
        fig.savefig('PolynomialOrder{:05d}.png'.format(pow_i), bbox_inches='tight', dpi=100)
        plt.close(fig)