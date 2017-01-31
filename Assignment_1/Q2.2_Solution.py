# ECE 521 Assignment 1 Question 2.2 Solution
#
# Code framework adapted from reference code given in ECE 521 course
# Original version linked below:
# http://www.psi.toronto.edu/~jimmy/ece521/mult.py
#
# Modifications written by : (Frank) Yu Cheng Gu
# Date: Jan 31, 2017

import numpy as np
import tensorflow as tf
import csv

# Load the data set
with np.load ("tinymnist.npz") as data :
    trainData, trainTarget = data ["x"], data["y"]
    validData, validTarget = data ["x_valid"], data ["y_valid"]
    testData, testTarget = data ["x_test"], data ["y_test"]

N = 700		# Number of data points
decay = 0	# Weight decay lambda term
mini_batch_size = 50	# Size of the mini-batch
M = 1000	# Number of iterations
a = 0.1	# Learning rate

def buildGraph():
    # Variable creation
    W = tf.Variable(tf.zeros([64,1]), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X,W) + b

    # Error definition
    # Basic mean squared error
    meanSquaredError = tf.reduce_mean(tf.reduce_mean(0.5*tf.square(y_predicted - y_target),
                                                reduction_indices=1,
                                                name='squared_error'),
                                  				name='mean_squared_error')
    l2Loss = 0.5 * tf.reduce_sum(tf.square(W), name='l2_loss')	# l2 penalty
    meanSquaredError = meanSquaredError + decay * l2Loss		# Weight decay term

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = a)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_predicted, meanSquaredError, train


def runMult():

    # Build computation graph
    W, b, X, y_target, y_predicted, meanSquaredError, train = buildGraph()

    # Initialize session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


    initialW = sess.run(W)
    initialb = sess.run(b)


    wList = []
    n_batch = N // mini_batch_size + (N % mini_batch_size != 0) # Number of possible batches in the training data set
    f = open("./2_2.csv","w")
    writer = csv.writer(f)

	# Training model
    for step in xrange(0,M + 1):
        i_batch = (step % n_batch)*mini_batch_size		# mini-batch index size
        batchData = trainData[i_batch:i_batch+mini_batch_size]
        batchTarget = trainTarget[i_batch:i_batch+mini_batch_size]
        _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict={X: batchData, y_target: batchTarget})
        wList.append(currentW)
        data = [step,err]
        writer.writerow(data)

    f.close()
    # Testing model
    errTest = sess.run(meanSquaredError, feed_dict={X: validData, y_target: validTarget})
    print("Final testing MSE: %.4f"%(errTest))


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    runMult()
