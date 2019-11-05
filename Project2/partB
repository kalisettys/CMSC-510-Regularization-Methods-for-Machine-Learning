# AUTHOR: Shilpa Kalisetty
# CMSC 510: Homework 2 - Part B
# Tom Arodz
# October 15th, 2019



from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


np.random.seed(200)

## hyperparameters
total_features = 784
number_of_iterations = 200
batch_size = 128
learning_rate = 0.01

## my eID (V#): V00561106 --- focusing on the last 2 digits = 0 & 6
classA_digit = 0
classB_digit = 6

# loading data using keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# replacing the features set with my last 2 digits of my V#
train_class= np.where((y_train == classA_digit) | (y_train == classB_digit))
test_class= np.where((y_test == classA_digit) | (y_test == classB_digit))
x_train, y_train = x_train[train_class], y_train[train_class]
x_test, y_test = x_test[test_class], y_test[test_class]

# reshape training and testing data
# training data
x_train = x_train.astype(dtype='float64')
y_train = y_train.astype(dtype='float64')
y_train = y_train.reshape(y_train.shape[0], 1)

# testing data
x_test = x_test.astype(dtype='float64')
y_test = y_test.astype(dtype='float64')
y_test = y_test.reshape(y_test.shape[0], 1)

# flattening the images and normalizing the values to either 0 or 1
x_train = x_train.reshape(x_train.shape[0], total_features) / float(255)
x_test = x_test.reshape(x_test.shape[0], total_features) / float(255)
#print("shilpa:", x_train.shape[1])

# encoding classes as +1 or -1
def convert_class(xT, yT, c1, c2):
    x_train_new = []
    y_train_new = []
    for i in range(xT.__len__()):
        if yT[i] == c1 or yT[i] == c2:
            x_train_new.append(xT[i])
            if yT[i] == c1:
                y_train_new.append(1)
            elif yT[i] == c2:
                y_train_new.append(-1)
    return np.array(x_train_new), np.array(y_train_new).reshape(y_train_new.__len__(), 1)

x_train, y_train = convert_class(x_train, y_train, classA_digit, classB_digit)
x_test, y_test = convert_class(x_test, y_test, classA_digit, classB_digit)


# ------- TRAINING PROCESS -------- #
data_train_shape = x_train.shape[0]
features_shape = x_train.shape[1]
#print(x_train.shape[1])


#defining weights
w = tf.Variable(np.random.rand(features_shape, 1).astype(dtype='float64'), name='w')

# defining bias
b = tf.Variable(0.0, name='b', dtype=tf.float64,)

## x = features
x = tf.placeholder(dtype=tf.float64, name='x')
## y = samples
y = tf.placeholder(dtype=tf.float64, name='y')

#predicted value -- LINEAR MODEL = W^t * X + b
predicted_value  = tf.matmul(x,w) + b

#loss and the risk function == tells us if the classifier matches the training data well
##logistic loss from notes: min 1/m sum of ln(1 + e^-y-wTxi)
logistic_loss = tf.log(1 + tf.exp(tf.multiply(-1.0 * y, predicted_value)))
#print("Initial Logistic Loss:", logistic_loss)

#risk
risk = tf.reduce_mean(logistic_loss)
#print("Initial Risk: ", risk)

#gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(risk)

# starts the tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


## y_predicted and the MSE for the testing dataset using random weights and bias
y_predicted = sess.run([predicted_value], feed_dict={x:x_test, y:y_test})[0]
mean_squared_error = np.mean(np.square(y_predicted-y_test), axis=0)[0]


### development stage: randomly selecting a subset of samples to speed up the calculations
for n in range(0, number_of_iterations):
    for o in range(0, data_train_shape, batch_size):
        subsample_dataset= min(data_train_shape, o + batch_size)
        x_subsample = x_train[o:subsample_dataset, :]
        y_subsample = y_train[o:subsample_dataset, :]

        # training risk -- feeding training data as subsamples to the dictionary
        _, final_risk_train, predicted_y_batch, final_w = sess.run(
            [optimizer, risk, predicted_value, w], feed_dict={x: x_subsample, y: y_subsample})

    # testing risk --- feeding testing data set to the dictionary
    y_predicted, final_w, final_b, final_risk_test = sess.run([predicted_value, w, b, risk], feed_dict={x: x_test, y: y_test});


    ### SHOWING RESULTS FOR EVERY 10 ITERATIONS
    if (n + 1) % 10 == 0:
        mse = np.mean(np.mean(np.square(y_predicted-y_test), axis=1), axis=0)
        print("Mean Squared Error after the subsampling")
        print("Iteration", (n+1), ": MSE = ", mse, "Training Risk: ", final_risk_train, "Testing Risk: ", final_risk_test)





#print("Weights at the end of training.....")
#print(final_w)

print("Bias at the end of training........")
print(final_b)


# ----------- TESTING PROCESS ---------- #
testing_values = []
classA_val = 0
classB_val = 0

w_new = final_w.reshape(final_w.__len__(), 1)

##linear model
linear_model = np.matmul(x_test, w_new) + final_b

for i in range(0, x_test.__len__()):
    if linear_model[i][0] < 0:
        classA_val += 1
        testing_values.append(-1)
    elif linear_model[i][0] > 0:
        classB_val += 1
        testing_values.append(1)

testing_values = np.array(testing_values)
y_test = y_test.reshape(1, y_test.__len__())[0]
#print(testing_values)
#print(y_test)


# call later at the end to calculate accuracy between actual vs. predicted
def calculate_accuracy(actual_y_value, my_predicted_y_value):
    total_n = actual_y_value.__len__()
    accPred = 0
    for i in range(0, total_n):
        accPred += (actual_y_value[i] == my_predicted_y_value[i])
    return accPred / total_n

## accuracy of our calculations
accuracy_test = calculate_accuracy(y_test, testing_values) * 100

def get_class_val(x_val, y_val, get_feature):
    class_digs = []
    for d in range(0, x_val.__len__()):
        if y_val[d] == get_feature:
            class_digs.append(x_val[d])
    return class_digs

#testing predicted and actual values:
y_test_digitA = get_class_val(x_test, y_test, -1)
y_test_digitB = get_class_val(x_test, y_test, 1)

print("--------------------------------Final results:---------------------------------")
print('\n')
print("Class A: Predicted vs. Actual: ")
print("Predicted: ", classA_val, "images of my second to last digit of V#: ", classA_digit)
print("Actual: ", y_test_digitA.__len__(), "images of my second to last digit of V# :", classA_digit)
print('\n')
print("Class B: Predicted vs. Actual: ")
print("Predicted: ", classB_val, "images of my last digit of V#: ", classB_digit)
print("Actual: ", y_test_digitB.__len__(), "images of my last digit of V# :", classB_digit)
print('\n')
print("Test set MSE: ", mse)
print("Test set accuracy: ", accuracy_test)

print("Training Risk: ", final_risk_train)
print("Testing Risk: ", final_risk_test)

# ------------------------- PLOTTING ------------------------- #
# manually got the training risk values for ONLY EVERY 10 iterations as I had issues plotting

training_risk = np.array([0.10029846454020026, 0.03946236793207633, 0.02086664719232979,  0.013816912211305574, 0.010617027921032411, 0.00893084408955733, 0.007935550704356097,
                         0.0072926045769607034, 0.0068452016467403735, 0.0065139387002704965,  0.0062555118702305275, 0.006044894377614434,  0.005867043978496106, 0.005712671540954743,
                         0.00557589584611205,  0.0054528830071823645, 0.005341044792204223, 0.005238564131470404, 0.0051441127445317645, 0.005056681340031834])

testing_risk = np.array([0.26002986696148234, 0.1341184753499526, 0.10016251795091212, 0.08448836240625196,  0.07531905170792219, 0.06921213029656365,
                        0.06481719946190782, 0.0614714205911792, 0.05880779605895222, 0.05660972668337951, 0.05474359740971413, 0.05312362762914316,
                        0.05169266457353484, 0.05041132753541506, 0.049251624598867845, 0.04819305600077041, 0.047220168591290516, 0.04632099142155886, 0.045486019522284975, 0.044707543213412304])

training_iterations_label = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])


#Plot of training risk value vs. iterations of training
plt.plot(training_iterations_label, training_risk)
plt.title('Training Risk Values vs. Number of Iterations')
plt.xlabel("Number of Iterations")
plt.ylabel("Training Risk Values")
plt.show()


#Plot of testing risk value vs. iterations of training
plt.plot(training_iterations_label, testing_risk)
plt.title('Testing Risk Values vs. Number of Iterations')
plt.xlabel("Number of Iterations")
plt.ylabel("Testing Risk Values")
plt.show()

## FOR COMPARISON PURPOSES:
#Plot of TRAINING AND TESTING risk value vs. iterations of training
plt.plot(training_iterations_label, training_risk, 'r', label='Training Risk' )
plt.plot(training_iterations_label, testing_risk, 'b', label='Testing Risk')
plt.title('Train & Test Risk Values vs. Number of Iterations')
plt.xlabel("Number of Iterations")
plt.ylabel("Risk Values")
plt.legend(loc='center')
plt.show()


#Plot of training set error and test set error vs. iterations of training --- predicted vs. actual values
## weird plot with this so not including in the report ---- RUNNING OUT OF TIME--- i apologize!!!!!
plt.plot(testing_values, y_test, 'r')
plt.title('Error Rate vs. Number of Iterations')
plt.xlabel("Number of Iterations")
plt.ylabel("Error Rate")
plt.show()



##LINKS USED:
#https://katbailey.github.io/post/neural-nets-in-python/
#https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
#https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
