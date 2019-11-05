import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as T
from sklearn.model_selection import train_test_split
import itertools

### START OF SOME FAKE DATASET GENERATION

# the dataset does not have column names
# so we add column names using the wdbc-names.csv file
column_names = ['ID', 'Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
                'Compactness', 'Concavity', 'ConcavePoints', 'Symmetry', 'FractalDimension',
                'seRadius', 'seTexture', 'sePerimeter', 'seArea', 'seSmoothness', 'seCompactness', 'seConcavity',
                'seConcavePoints', 'seSymmetry', 'seFractalDimension', 'worstRadius', 'worstTexture',
                'worstPerimeter', 'worstArea', 'worstSmoothness', 'worstCompactness', 'worstConcavity',
                'worstConcavePoints', 'worstSymmetry', 'worstFractalDimension']

#reading the data file
file_name = "wdbc-data.csv"
wdbc_data = pd.read_csv(file_name, header=None, names=column_names)
#print(wdbc_data)

##Two-class classification = class Malignant and class Benign = binary representation
classM = 1
classB = 0

class_classification = {'M': classM, 'B': classB}
wdbc_data['Diagnosis'] = wdbc_data['Diagnosis'].map(class_classification)
#print("Below is the data:")
#print(wdbc_data)

## dropping unnecessary columns
drop_columns = ['ID','seRadius', 'seTexture', 'sePerimeter', 'seArea', 'seSmoothness', 'seCompactness', 'seConcavity',
                'seConcavePoints', 'seSymmetry', 'seFractalDimension', 'worstRadius', 'worstTexture',
                'worstPerimeter', 'worstArea', 'worstSmoothness', 'worstCompactness', 'worstConcavity',
                'worstConcavePoints', 'worstSymmetry', 'worstFractalDimension']

dropped_data = wdbc_data.drop(drop_columns, axis=1, inplace=True)

#representing features (x) and y_predicted (class M and class B)
features = list(wdbc_data.columns[1: wdbc_data.shape[1]])
target_y = wdbc_data.columns[0]


'''
# Malignant class
malignant_class = len(wdbc_data[wdbc_data['Diagnosis']==1])
#Benign class
benign_class = len(wdbc_data[wdbc_data['Diagnosis']==0])
print("Testing: ")
print("There are a total of {}".format(malignant_class)+" malignant features.")
print("There are a total of {}".format(benign_class)+" benign features. \n")
'''

# setting up the parameters for probability model
X = wdbc_data[features]
Y = wdbc_data[target_y]

#print("Diagnosis values")
#print(Y)

#print("X: ", X.shape)
#print("Y: ", Y.shape)

#normalizing the data
def normalize(data):
    norm_data = pd.DataFrame()
    for i in data:
        if i in data.select_dtypes(include=[np.float]):
            norm_data[i] = ((data[i] - data[i].min()) / (data[i].max() - data[i].min()))
        else:
            norm_data[i] = data[i]
    return norm_data

X_normalized = normalize(X)

# splitting the data into training and testing sets
# used the most common/default training set = 80% and testing set = 20%
x_train, x_test, y_train, y_test = train_test_split(X_normalized, Y, test_size=0.20, random_state=42)

# initializing parameters
numberOfFeatures = x_train.shape[1]

# reshape training and testing data
# training data
x_train = x_train.astype(dtype='float64')
y_train = y_train.astype(dtype='float64')

# testing data
x_test = x_test.astype(dtype='float64')
y_test = y_test.astype(dtype='float64')
'''
print("x train values are: ")
print(x_train)
print("y train values are: ")
print(np.array(y_train))
print("x test values are: ")
print(x_test)
print("y test values are: ")
print(np.array(y_test))
'''

#### END OF FAKE DATASET GENERATION

#### START OF MODEL BUILDING AND ESTIMATION

# instantiate an empty PyMC3 model
basic_model = pm.Model()

# fill the model with details:
with basic_model:
    # parameters for priors for gaussian means
    mu_prior_cov = 100 * np.eye(numberOfFeatures)
    mu_prior_mu = np.zeros((numberOfFeatures,))

    # Priors for gaussian (Gaussian prior): w, b
    w = pm.MvNormal('estimated_w', mu=mu_prior_mu, cov=mu_prior_cov, shape=numberOfFeatures)
    b = pm.Normal('estimated_b', 0, 1000)

    # w,b needs to be updated
    # we get x from the dataset wdbc

    # linear model u = w^Transpose * x + b
    u = pm.Deterministic('u', T.dot(X, w) + b)

    # defining prob for y_obs prediction
    prob = pm.Deterministic('prob', 1.0 / (1.0 + T.exp(-1.0 * u)))

    Y_obs = pm.Bernoulli('Y_obs', p=prob, observed=Y)

# done with setting up the model

# now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
# map_estimate1 is a dictionary: "parameter name" -> "its estimated value"
map_estimate1 = pm.find_MAP(model=basic_model)

actual_w = map_estimate1['estimated_w']
actual_b = map_estimate1['estimated_b']

print("Estimated w: ", actual_w)
print("Estimated b: ", actual_b)

w_final = np.transpose(actual_w)

#normalizing x test
#x_test_normalized = normalize(x_test)

# converting the probability into values of class M or class B
def predictor(x):
    y_predicted = np.array([])
    initial_classM = 0
    initial_classB = 0
    for i in range(0, x.__len__()):

        u_val = np.any((np.dot(x, w_final) + actual_b))
        testing_class_value = 1.0 / (1.0 + np.exp(-1.0 * u_val))

        if testing_class_value >= 0.5:
            initial_classM +=1
            y_predicted = np.append(y_predicted, 1)
        elif testing_class_value < 0.5:
            initial_classB += 1
            y_predicted = 0
            y_predicted = np.append(y_predicted, 0)
    return y_predicted

training_accuracy = []
testing_accuracy = []

y_predicted_train = predictor(x_train)
y_predicted_test = predictor(x_test)


from sklearn.metrics import accuracy_score

accuracy_train = 100 - (accuracy_score(y_train, y_predicted_train) * 100)
print("Training Accuracy: ", accuracy_train, "%")

accuracy_test = 100 - (accuracy_score(y_test, y_predicted_test) * 100)
print("Testing Accuracy: ", accuracy_test, "%")

'''
print('\n')
print("------------Accuracy Results----------")
print("Training accuracy: {} %".format(100 - np.mean(np.abs(y_predicted_train - y_train)) * 100))
train_accry = (100 - np.mean(np.abs(y_predicted_train - y_train)) * 100)
training_accuracy.append(train_accry)

print("Testing accuracy: {} %".format(100 - np.mean(np.abs(y_predicted_test - y_test)) * 100))
test_accry = (100 - np.mean(np.abs(y_predicted_test - y_test)) * 100)
testing_accuracy.append(test_accry)
'''
