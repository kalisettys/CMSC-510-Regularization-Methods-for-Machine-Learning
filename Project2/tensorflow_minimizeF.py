#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:36:26 2017

@author: tarodz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf;


minimum_w=[1.0, 3.0]


#define some function using tensorflow operations 
# this function is f(w)=||w-minimum||^2, and so has minimum at minimum_w, i.e. at vector [1.0,3.0]
def f(w):
    shiftedW=w-np.array(minimum_w);
    return tf.reduce_sum(tf.multiply(shiftedW,shiftedW));

#define starting value of W for gradient descent
#here, W is a 2D vector
initialW = np.random.rand(2)

#create a tf.Variable (i.e. a variable that persists between calls to a tensorflow session calls like .run()) 
w = tf.Variable(initialW,name="w");

#define output of applying f to w
#out goal will be to minimize f(w), i.e. find w with lowest possible f(w)
z=f(w);
# z is not a tf.Variable - it does not persist between tensorflow session calls like .run()) 
# z only exists within a single call, and has value recalculated from w through function f()
# e.g. in a classifier, the loss would be a variable like z 

# check also: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
# check also: https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(z)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

with sess:
    for i in range(1000):
        train.run();
        npw=w.eval();
        print(npw)
#sess.close()

print("True minimum: "+str(minimum_w));
print("Found minimum:"+str(npw));
