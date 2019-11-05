# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 1 19:55:03 2019

@author: shilpamac

"""

import numpy as np;
import matplotlib.pyplot as plt;


def main(x, y, number_of_iterations, gamma, polynomial_degree):
    ##size of m
    total_samples_m = x.shape[0]



    # Initializing x_train matrix: consits of each of the polynomial degree term of x
    x_train = np.ones((total_samples_m, polynomial_degree + 1), dtype=float)

    # Initializing y_predicted
    y_predicted = np.zeros((total_samples_m, 1))

    ### initializing coefficients
    np.random.seed(100)
    w_old = np.random.rand(polynomial_degree + 1, 1)
    # print(w_old)

    x_rows_sum = np.zeros((total_samples_m, 1), dtype=float)

    # Initialize loss vector
    loss = np.zeros((total_samples_m, 1))

    # Initializing g vector
    g = np.zeros((total_samples_m, 1), dtype=float)

    # Initializing new W vector
    w_new = np.zeros((polynomial_degree + 1, 1))

    ##creating a matrix of [1, x, x**2, x**3, x**4, x**5]
    for i in range(total_samples_m):
        for j in range(1, polynomial_degree + 1):
            x_train[i][j] = x[i] ** j
    # print("-------X vectors---------")
    # print(x_train)

    # adding up all the individual rows --- (x) values for calculating g
    # second part in g formula (1+x+x**2.....)
    for i in range(total_samples_m):
        for j in range(polynomial_degree + 1):
            x_rows_sum[i] = x_rows_sum[i] + x_train[i][j]

    # print("-------------------x_rows_sum------")
    # print(x_rows_sum)

    for n in range(number_of_iterations):
        ##getting h(x,w) values
        y_predicted = np.matmul(x_train, w_old)
        #print("----------predicted y--------")
        #print(y_predicted)

        y_difference = y_predicted[i] - y[i]
        # print("----------Difference between predicted and actual Y--------")
        # print(y_difference)

        ##defining loss vector for each training point: L(h(x), y) = (h(x) - y)**2
        for i in range(total_samples_m):
            loss[i] = (y_predicted[i] - y[i]) ** 2
        # print("----------loss for each sample --------")
        # print(loss)

        # adding up all the loss values
        sum_squared_error = 0
        for i in range(total_samples_m):
            sum_squared_error += loss[i]
        # print("----------Sum of all the loss  --------")
        # print(sum_squared_error)

        # average loss ---- MEAN SQUARED ERROR------
        mean_squared_error = sum_squared_error / (total_samples_m)
        # print("----------MSE--------")
        # print(mean_squared_error)

        ##GRADIENT DESCENT-----derivation
        # gradient formula for each sample
        for i in range(total_samples_m):
            g[i] = 2 * ((y_predicted[i]) - y[i]) * (x_rows_sum[i])

        #print("-------g values------")
        #print(g)

        # Sum of g for each sample
        sum_of_g = 0
        for i in range(total_samples_m):
            sum_of_g += g[i]
        # print("----------Sum of G--------")
        # print(sum_of_g)

        mean_g = sum_of_g / (total_samples_m)
        print("----------Mean_G--------")
        print(mean_g)

        ##modification of w values---step 4
        # checking if L = 0, if its greater than 0, then keep modifying, will be printing new w values
        for i in range(polynomial_degree + 1):
            w_new[i] = w_old[i] - (gamma * mean_g)
        # print("----------NEW W VALUES-------")
        # print(w_new)

        ##this step doesn't do anything---trying to update the w values so that the new w_new values will be directly referenced as w_old
        for i in range(polynomial_degree + 1):
            w_old[i] = w_new[i]

        # print("----------NEW W VALUES assigned to w old-------")
        # print(w_new)

        # print("--------iteration", n, "done----------")

    print("----------NEW W VALUES-------")
    print(w_new)

    print("----------Difference between predicted and actual Y--------")
    print(y_difference)

    print("----------MSE--------")
    print(mean_squared_error)


if __name__ == "__main__":
    x = np.array([-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904, 2.31579097, 2.40479939,
                  -2.22112823])
    y = np.array(
        [-18.56122168, -24.99658931, -24.41907817, -2.688209, -1.54725306, -19.18190097, 1.74117419, 3.97703338,
         -24.80977847])

    ##----INITIALIZING HYPERPARAMETERS-----##
    # gamma (learning rate), number_of_iterations, degree of polynomial (n)
    for gamma in (0.0001, 0.00001, 0.000001):
        for number_of_iterations in (5000,):
            for polynomial_degree in (2,):
                main(x, y, number_of_iterations, gamma, polynomial_degree)
                input(
                    "gamma: {}\nnumber_of_iterations: {}\npolynomial_degree: {}\n\nClick Enter to see the next run...".format(
                        gamma, number_of_iterations, polynomial_degree))


##-----------PLOTTING-------------------------###

def first_degree_graph(equation, x):
    x = np.array([-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904, 2.31579097, 2.40479939,
                  -2.22112823])
    y = eval(equation)
    plt.scatter(x, y, 'b')
    plt.show()


plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('First Degree Polynomial')

first_degree_graph('6.123*x-11.06', range(-5, 5))


def second_degree_graph(equation, x):
    x = np.array([-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904, 2.31579097, 2.40479939,
                  -2.22112823])
    y = eval(equation)
    plt.scatter(x, y, 'b')
    plt.show()


plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Second Degree Polynomial')

second_degree_graph('-0.492*x**2+6.382*x-8.983', range(-5, 5))


def third_degree_graph(equation, x):
    x = np.array([-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904, 2.31579097, 2.40479939,
                  -2.22112823])
    y = eval(equation)
    plt.scatter(x, y, 'b')
    plt.show()


plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Third Degree Polynomial')

third_degree_graph('0.293*x**3-0.783*x**2+5.123*x-8.345', range(-5, 5))


def fourth_degree_graph(equation, x):
    x = np.array([-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904, 2.31579097, 2.40479939,
                  -2.22112823])
    y = eval(equation)
    plt.scatter(x, y, 'b')
    plt.show()


plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fourth Degree Polynomial')

fourth_degree_graph('0.935*x**4+0.687*x**3-7.889*x**2+2.735*x+3.978', range(-5, 5))


def fifth_degree_graph(equation, x):
    x = np.array([-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904, 2.31579097, 2.40479939,
                  -2.22112823])
    y = eval(equation)
    plt.scatter(x, y, 'b')
    plt.show()


plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fifth Degree Polynomial')

fifth_degree_graph('0.00000855*x**5+1.0032*x**4+0.085*x**3-7.988*x**2+3.081*x+4.416', range(-5, 5))
