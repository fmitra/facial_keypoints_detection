""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://www.kaggle.com/c/facial-keypoints-detection

Competition goal is to predict keypoint positions on face images.

This is a learning exercise and should not be used as a reference for neural 
net implementations. I chose to build out the network manually in Python
to get a better understanding of the inner workings of the process, and due
to that, some of the initial settings or process may not be ideal. 

Process

1. Parse and feature scale the dataset 
2. Initialize random theta parameters for the input and hidden layer
3. Implement feedforward propagation to get outputs
4. Implement back propagation to calculate gradients

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

import json
import os
import pickle
import pandas as pd
import numpy as np


TRAIN = 'data/training.csv'
TEST  = 'data/test.csv'
FLOOKUP = 'data/IdLookupTable.csv'

def load_data():
    print("")
    print("Loading data set...")

    # Load the training or test set
    df_train = pd.read_csv(TRAIN, header=0)
    df_test  = pd.read_csv(TEST, header=0)

    print("Cleaning data set...")
    # Convert values in image column into a numpy array
    # and drop empty values
    df_train['Image'] = df_train['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df_test['Image'] = df_test['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Perform feature scaling to speed up gradient descent
    # Image values originally took on a range of 0-255
    X = np.vstack(df_train['Image'].values) / 255
    X = X.astype(np.float32)
    x = np.vstack(df_test['Image'].values) / 255
    x = X.astype(np.float32)

    # If we're running the training set, we can identify
    # the target values. The target values are the coordinates
    # of 15 facial keypoints (ex. right_eyebrow_inner_end)
    # https://www.kaggle.com/c/facial-keypoints-detection/data
    y = df_train[df_train.columns[:-1]].values
    y = (y-48) / 48
    y = y.astype(np.float32)

    return X, x, y

def initial_settings(X):
    """
    Define the shape of the neural network based on the data set 
    and feature size. Set parameters of theta (input and hidden) to
    take on random small values.
    """
    # Start the random generation at 0
    np.random.seed(0)
    # Total size of the training set
    num_examples = len(X)
    # Total size of the input layer given 96x96 pixel images
    nn_input = 9216
    # Total size of hidden layer (picked randomly)
    nn_hidden = 100
    # Total size of the output layer based on targets in
    # the data set (facial keypoint positions)
    nn_output = 30

    print("Initializing random values for theta...")
    np.random.seed(0)
    # Initialize random values of theta to break symmetry
    # Initial parameters for our input layer
    # Data set shape is 9216 x 100 
    theta_1 = np.random.randn(nn_input, nn_hidden) / np.sqrt(nn_input)
    # Initial parameters for our hidden layer
    # Data set shape is 100 x 30
    theta_2 = np.random.randn(nn_hidden, nn_output) / np.sqrt(nn_hidden)

    settings = {
        'num_examples': num_examples,
        'nn_input': nn_input,
        'nn_hidden': nn_hidden,
        'nn_output': nn_output,
        'iterations': 30,
        'epsilon': 0.01,
        'theta_1': theta_1,
        'theta_2': theta_2
    }

    return settings

def save_submission(predictions):
    """
    Scale features back to their original size 
    and parse file into submission format
    """
    predictions = predictions * 48 + 48
    predictions = predictions.clip(0, 96)
    columns = pd.read_csv(TRAIN, header=0).columns[:-1]
    df = pd.DataFrame(predictions, columns=columns)
    lookup_table = pd.read_csv(FLOOKUP)
    values = []
    for i, r in lookup_table.iterrows():
        values.append((
            r['RowId'],
            df.ix[r.ImageId - 1][r.FeatureName],
            ))

    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv("submission.csv", index=False)
    print("Predictions saved for submission")

def sigmoid(z):
    """
    Sigmoid function will be used as the activation function
    during feedforward propagation
    """
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s

def predict(model, X, m):
    """
    Implements forward propagation with a defined set of theta
    matrices to predict test data
    """
    print("Calculating predictions...")
    theta_1 = model['theta_1']
    theta_2 = model['theta_2']
    theta_1 = theta_1.T
    theta_1 = np.concatenate((np.ones([len(theta_1), 1]), theta_1), axis=1)
    theta_1 = theta_1.T
    theta_2 = theta_2.T
    theta_2 = np.concatenate((np.ones([len(theta_2), 1]), theta_2), axis=1)
    theta_2 = theta_2.T

    # Forward propagation
    a1 = np.concatenate((np.ones([m,1]), X), axis=1)
    z2 = np.matrix(a1) * np.matrix(theta_1)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones([len(a2), 1]), a2), axis=1)
    z3 = a2 * theta_2
    a3 = sigmoid(z3)

    save_submission(a3)

def build_model(settings, X, y):
    """
    Implements forward and back propagation to determine ideal
    values of theta
    """
    model = {}
    m = settings['num_examples']
    iterations = settings['iterations']
    epsilon = settings['epsilon']
    theta_1 = settings['theta_1']
    theta_2 = settings['theta_2']
    # print("Original theta_1 {}".format(theta_1.shape))
    # print("Original theta_2 {}".format(theta_2.shape))

    for i in range(iterations):
        # This is the beginning of forward propagation
        # We multiply our training set matrix against the
        # theta matrices, supplemented with bias vectors
        # containing the value one. Our activation function (sigmoid)
        # will transition our dataset from the input layer
        # into the final layer of the network
        print("")
        print("Beginning forward propagation algorithm...")
        print("Setting bias vectors for thetas and starting activations...")
        # Transpose theta so we can attach a bias vector
        # filled with ones and perform matrix multiplication
        theta_1 = theta_1.T
        theta_1 = np.concatenate((np.ones([len(theta_1), 1]), theta_1), axis=1)
        theta_1 = theta_1.T
        # a1 represents the input layer
        a1 = np.concatenate((np.ones([m,1]), X), axis=1)
        # z2 represents transition from the input layer
        # into the output layer
        z2 = np.matrix(a1) * np.matrix(theta_1)
        # Perform te same transpose operation to theta_2 
        # in order to attach the bias vector
        theta_2 = theta_2.T
        theta_2 = np.concatenate((np.ones([len(theta_2), 1]), theta_2), axis=1)
        theta_2 = theta_2.T
        # a2 represents the hidden layer. Data moves
        # into the hidden layer after running through
        # our activation function
        a2 = sigmoid(z2)
        # Attach a bias vector of 1's to the front of the 
        # hidden layer matrix
        a2 = np.concatenate((np.ones([len(a2), 1]), a2), axis=1)
        # We don't need to convert to a matrix this time around
        # because a2 was previously converted
        z3 = a2 * theta_2

        # a3 represents our output layer, or hypothesis. Once
        # we have the  hypothesis we can work our way backwards
        # via the backpropagation algorithm to computer our
        # gradients
        a3 = sigmoid(z3)
        # print("a3 {}".format(a3.shape))
        # print("a2 {}".format(a2.shape))
        # print("a1 {}".format(a1.shape))
        # print("X {}".format(X.shape))
        # print("y {}".format(y.shape))

        # This is the beginning of back propagation. We calculate
        # the values of delta by moving backwards through the layers
        # after receiving the hypothesis
        print("Beginning back propagation algorithm...")
        # delta values are the errors of of our nodes within each layer.
        # We'll calculate it by moving backwards from the hypothesis (output)
        # layer. Note that we do not calculate delta_1 because we do not
        # associate error with the input layer
        delta_3 = a3 - y
        delta_2 = np.dot(delta_3, theta_2.T)
        # Drop the bias vector
        delta_2 = delta_2[:,1:]
        theta_2_grad = np.dot(a2.T, delta_3)
        theta_1_grad = np.dot(a1.T, delta_2)
        # print("delta_2 {}".format(delta_2.shape))
        # print("delta_3 {}".format(delta_3.shape))
        # print("theta_1 {}".format(theta_2.shape))
        # print("theta_2 {}".format(theta_1.shape))
        # print("theta_1_grad {}".format(theta_1_grad.shape))
        # print("theta_2_grad {}".format(theta_2_grad.shape))

        # Adjust the values of theta with learning rate epsilon
        # and calculated gradients. We're not implementing 
        # regularization for this example
        print("Adjusting values of theta matrices...")
        theta_2 += -epsilon * theta_2_grad
        theta_1 += -epsilon * theta_1_grad
        # Drop the bias vectors so we can return theta_1/2 to
        # it's original shape prior to running the next iteration
        theta_2 = theta_2[1:]
        theta_1 = theta_1[1:]
        # print("New theta_2 {}".format(theta_2.shape))
        # print("New theta_1 {}".format(theta_1.shape))

        model = {'theta_1': theta_1, 'theta_2':theta_2}

    print("")
    return model

def start():
    X, x, y = load_data()
    settings = initial_settings(X)
    model = build_model(settings, X, y)
    predictions = predict(model, x, settings['num_examples'])

if __name__ == '__main__':
    start()





