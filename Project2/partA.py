# AUTHOR: Shilpa Kalisetty
# CMSC 510: Homework 2 - Part A
# Tom Arodz
# October 15th, 2019

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

x = np.array([-1.67245526, -2.36540279, -2.14724263,  1.40539096,  1.24297767, -1.71043904,  2.31579097,  2.40479939, -2.22112823])
y=np.array([-18.56122168, -24.99658931, -24.41907817,  -2.688209, -1.54725306,  -19.18190097,   1.74117419, 3.97703338, -24.80977847])

# number of data points/ training samples
m = len(x)


## if need to check on each/individual or specific model, manually comment/uncomment the models below
## main function
def main():
    linear_model()
    quadratic_model()
    cubic_model()
    quartic_model()
    quintic_model()

def linear_model():
    x_tf = tf.placeholder('float32')
    y_tf = tf.placeholder('float32')

    w = tf.Variable(np.random.randn(), name="w", dtype='float32')

    #bias or w0
    b_w0 = tf.Variable(np.random.randn(), name="b_w0", dtype='float32')

    # hyperparameters
    learning_rate = 0.001
    number_of_iterations = 1000

    ##linear prediction
    linear_y_predicted = tf.add(tf.multiply(x_tf, w), b_w0)

    # Mean Squared Error Loss Function
    mean_squared_function = tf.reduce_mean(tf.square(linear_y_predicted - y_tf))

    # gradient descent built in function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_squared_function)

    # starts the session
    sess = tf.Session()

    # Initialize varialbles
    sess.run(tf.global_variables_initializer())

    # going through every iteration
    for iterate in range(number_of_iterations):

        # feeding the data to the dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={x_tf: _x, y_tf: _y})

        # results only for every 10 iterations
        if (iterate + 1) % 50 == 0:
            # mse calculated for every iteration
            mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})
            print("Iteration", (iterate + 1), ": MSE =", mse, "w0 =", sess.run(b_w0), "w1 =", sess.run(w))


    print("-----------------LINEAR MODEL RESULTS------------------")

    loss_mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})

    weight1 = sess.run(w)
    bias = sess.run(b_w0)

    #plot of linear model - first degree polynomial
    first_degree_equation = bias + weight1 * x
    print("MSE =", loss_mse, "W1 =", weight1, "bias =", bias, '\n')

    # Plotting the Results
    plt.plot(x, y, 'ro')
    plt.plot(x, first_degree_equation)
    plt.title('First Degree Polynomial')
    plt.show()


def quadratic_model():
    x_tf = tf.placeholder('float32')
    y_tf = tf.placeholder('float32')

    w1 = tf.Variable(np.random.randn(), name="w1", dtype='float32')
    w2 = tf.Variable(np.random.randn(), name="w2", dtype='float32')

    #bias or w0
    b_w0 = tf.Variable(np.random.randn(), name="b_w0", dtype='float32')

    # hyperparameters
    learning_rate = 0.001
    number_of_iterations = 2000

    ##linear prediction
    quadratic_y_predicted = b_w0 + w1 * x_tf + w2 * tf.pow(x_tf,2)

    # Mean Squared Error Loss Function
    mean_squared_function = tf.reduce_mean(tf.square(quadratic_y_predicted - y_tf))

    # gradient descent built in function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_squared_function)

    # starts the session
    sess = tf.Session()

    # Initialize varialbles
    sess.run(tf.global_variables_initializer())

    # going through every iteration
    for iterate in range(number_of_iterations):

        # feeding the data to the dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={x_tf: _x, y_tf: _y})

        # results only for every 10 iterations
        if (iterate + 1) % 50 == 0:
            # mse calculated for every iteration
            mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})

            print("Iteration", (iterate + 1), ": MSE =", mse, "w0 =", sess.run(b_w0), "w1 =", sess.run(w1), "w2 =", sess.run(w2))

    print("-----------------QUADRATIC MODEL RESULTS------------------")

    loss_mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})

    weight1 = sess.run(w1)
    weight2 = sess.run(w2)

    bias = sess.run(b_w0)

    #plot of 2nd degree model - second degree polynomial
    second_degree_equation = (bias + (weight1 * x) + (weight2 * x ** 2))
    print("MSE =", loss_mse, "W1 =", weight1, "W2 =", weight2, "bias =", bias, '\n')

    # Plotting the Results
    plt.plot(x, y, 'ro')
    plt.plot(x, second_degree_equation)
    plt.title('Second Degree Polynomial')
    plt.show()

def cubic_model():
    x_tf = tf.placeholder('float32')
    y_tf = tf.placeholder('float32')

    w1 = tf.Variable(np.random.randn(), name="w1", dtype='float32')
    w2 = tf.Variable(np.random.randn(), name="w2", dtype='float32')
    w3 = tf.Variable(np.random.randn(), name="w3", dtype='float32')

    #bias or w0
    b_w0 = tf.Variable(np.random.randn(), name="b_w0", dtype='float32')

    # hyperparameters
    learning_rate = 0.001
    number_of_iterations = 2000

    ##linear prediction
    cubic_y_predicted = b_w0 + w1 * x_tf + w2 * tf.pow(x_tf, 2) + w3 * tf.pow(x_tf, 3)

    # Mean Squared Error Loss Function
    mean_squared_function = tf.reduce_mean(tf.square(cubic_y_predicted - y_tf))

    # gradient descent built in function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_squared_function)

    # starts the session
    sess = tf.Session()

    # Initialize varialbles
    sess.run(tf.global_variables_initializer())
    # going through every iteration
    for iterate in range(number_of_iterations):

        # feeding the data to the dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={x_tf: _x, y_tf: _y})

        # results only for every 10 iterations
        if (iterate + 1) % 50 == 0:
            # mse calculated for every iteration
            mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})

            print("Iteration", (iterate + 1), ": MSE =", mse, "w0 =", sess.run(b_w0), "w1 =", sess.run(w1), "w2 =", sess.run(w2), "w3 =", sess.run(w3))

    print("-----------------CUBIC MODEL RESULTS------------------")

    loss_mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})

    weight1 = sess.run(w1)
    weight2 = sess.run(w2)
    weight3 = sess.run(w3)
    bias = sess.run(b_w0)

    #plot of 3rd degree model - third degree polynomial
    third_degree_equation = bias + weight1 * x + weight2 * x ** 2 + weight3 * x ** 3
    print("MSE =", loss_mse, "W1 =", weight1, "W2 =", weight2, "W3 =", weight3, "bias =", bias, '\n')

    # Plotting the Results
    plt.plot(x, y, 'ro')
    plt.plot(x, third_degree_equation)
    plt.title('Third Degree Polynomial')
    plt.show()

def quartic_model():
    x_tf = tf.placeholder('float32')
    y_tf = tf.placeholder('float32')

    w1 = tf.Variable(np.random.randn(), name="w1", dtype='float32')
    w2 = tf.Variable(np.random.randn(), name="w2", dtype='float32')
    w3 = tf.Variable(np.random.randn(), name="w3", dtype='float32')
    w4 = tf.Variable(np.random.randn(), name="w3", dtype='float32')

    #bias or w0
    b_w0 = tf.Variable(np.random.randn(), name="b_w0", dtype='float32')

    # hyperparameters
    ## learning rate greater than this fails so 0.001 fails
    learning_rate = 0.0001
    number_of_iterations = 2000

    ##linear prediction
    quartic_y_predicted = b_w0 + w1 * x_tf + w2 * tf.pow(x_tf,2) + w3 * tf.pow(x_tf,3) + w4 * tf.pow(x_tf,4)

    # Mean Squared Error Loss Function
    mean_squared_function = tf.reduce_mean(tf.square(quartic_y_predicted - y_tf))

    # gradient descent built in function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_squared_function)

    # starts the session
    sess = tf.Session()

    # Initialize varialbles
    sess.run(tf.global_variables_initializer())

    # going through every iteration
    for iterate in range(number_of_iterations):

        # feeding the data to the dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={x_tf:_x, y_tf:_y})

        # results only for every 10 iterations
        if (iterate + 1) % 50 == 0:
            # mse calculated for every iteration
            mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})

            print("Iteration", (iterate + 1), ": MSE =", mse, "w0 =", sess.run(b_w0), "w1 =", sess.run(w1), "w2 =", sess.run(w2), "w3 =", sess.run(w3), "w4 =", sess.run(w4))


    print("-----------------QUARTIC MODEL RESULTS------------------")

    loss_mse = sess.run(mean_squared_function, feed_dict ={x_tf: x, y_tf: y})

    weight1 = sess.run(w1)
    weight2 = sess.run(w2)
    weight3 = sess.run(w3)
    weight4 = sess.run(w4)

    bias = sess.run(b_w0)

    #plot of 3rd degree model - fourth degree polynomial
    fourth_degree_equation = bias + weight1 * x + weight2 * x **2 + weight3 * x **3 + weight4 * x **4
    print("MSE =", loss_mse, "W1 =", weight1, "W2 =", weight2, "W3 =", weight3, "W4 =", weight4, "bias =", bias, '\n')

    # Plotting the Results
    plt.plot(x, y, 'ro')
    plt.plot(x, fourth_degree_equation)
    plt.title('Fourth Degree Polynomial')
    plt.show()

def quintic_model():
    x_tf = tf.placeholder('float32')
    y_tf = tf.placeholder('float32')

    w1 = tf.Variable(np.random.randn(), name="w1", dtype='float32')
    w2 = tf.Variable(np.random.randn(), name="w2", dtype='float32')
    w3 = tf.Variable(np.random.randn(), name="w3", dtype='float32')
    w4 = tf.Variable(np.random.randn(), name="w3", dtype='float32')
    w5 = tf.Variable(np.random.randn(), name="w3", dtype='float32')

    #bias or w0
    b_w0 = tf.Variable(np.random.randn(), name="b_w0", dtype='float32')

    # hyperparameters
    ##don't lower it than 0.0001
    learning_rate = 0.0001
    number_of_iterations = 10000

    ##linear prediction
    quintic_y_predicted = b_w0 + w1 * x_tf + w2 * tf.pow(x_tf,2) + w3 * tf.pow(x_tf,3) + w4 * tf.pow(x_tf,4) + w5 * tf.pow(x_tf,5)

    # Mean Squared Error Loss Function
    mean_squared_function = tf.reduce_mean(tf.square(quintic_y_predicted - y_tf))

    # gradient descent built in function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_squared_function)

    # starts the session
    sess = tf.Session()

    # Initialize varialbles
    sess.run(tf.global_variables_initializer())

    # going through every iteration
    for iterate in range(number_of_iterations):

        # feeding the data to the dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={x_tf:_x, y_tf:_y})

        # results only for every 10 iterations
        if (iterate + 1) % 50 == 0:
            # mse calculated for every iteration
            mse = sess.run(mean_squared_function, feed_dict={x_tf:x, y_tf:y})

            print("Iteration", (iterate + 1), ": MSE =", mse, "w0 =", sess.run(b_w0), "w1 =", sess.run(w1), "w2 =", sess.run(w2), "w3 =", sess.run(w3), "w4 =", sess.run(w4), "w5 =", sess.run(w5))

    print("-----------------QUINTIC MODEL RESULTS------------------")

    loss_mse = sess.run(mean_squared_function, feed_dict={x_tf: x, y_tf: y})

    weight1 = sess.run(w1)
    weight2 = sess.run(w2)
    weight3 = sess.run(w3)
    weight4 = sess.run(w4)
    weight5 = sess.run(w5)

    bias = sess.run(b_w0)

    #plot of 5th degree model - fifth degree polynomial
    fifth_degree_equation = (bias + (weight1 * x) + (weight2 * x **2) + (weight3 * x **3) + (weight4 * x **4) + (weight5 * x **5))
    print("MSE =", loss_mse, "W1 =", weight1, "W2 =", weight2, "W3 =", weight3, "W4 =", weight4, "W5 =", weight5, "bias =", bias, '\n')

    # Plotting the Results
    plt.plot(x, y, 'ro')
    plt.plot(x, fifth_degree_equation)
    plt.title('Fifth Degree Polynomial')
    plt.show()



if __name__ == '__main__':
    main()


# The main problem that I faced in both Assignment 1 and PART A of Assignment 2 is plotting
# The polynomial functions and how to get them to look like their respective n-deg functions, but they ended up looking unsmooth,
# I tried searching & implementing for all the possibilities


# Assignment 2 PART A, I got so much BETTER results than Assignment 1. I think its because of the understanding of the task even MORE
#and because the use of tensorflow was reasonably easily explained from the links i used




