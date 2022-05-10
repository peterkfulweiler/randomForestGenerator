'''
Wesleyan University, COMP 343, Spring 2022
Homework 8: Neural Network Training
Name: Peter Fulweiler

Important: If you modify this file, you should submit to your homework directory

'''


import math
import numpy as np
import pandas as pd
import random

########################## Data helper functions #############################


def load_data(filename):
    ''' Returns a dataframe (df) containing the data in filename. You should
        specify the full path plus the name of the file in filename, relative
        to where you are running code
    '''
    df = pd.read_csv(filename)
    return df


def randomize_features(features, random_subspace=None):
    # get the number of columns of the dataframe
    n_col = len(features)
    # get the indeces of the columns excluding target column
    n_col_indeces = list(range(n_col))
    # check if random
    if random_subspace != None:
        features = random.sample(
            [*features], k=random_subspace)
    # initialize empty list of random features
    #random_features = []
    # append randomly chosen features to the random_features list
    """
    for index in n_col_indeces:
        feat = features[index]
        random_features.append(feat)
    """
    # returns set
    return set(features)


def sigmoid(z):
    ''' Sigmoid function '''
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    ''' Derivative of sigmoid function '''
    sig_z = sigmoid(z)
    return sig_z * (1-sig_z)


def split_data(df, train_proportion):
    ''' Inputs
            * df: dataframe containing data
            * train_proportion: proportion of data in df that will be used for
                training. 1-train_proportion is proportion of data to be used
                for testing
        Output
            * train_df: dataframe containing training data
            * test_df: dataframe containing testing data
    '''
    # Make sure there are row numbers
    df = df.reset_index(drop=True)

    # Reorder examples and split data according to train proportion
    train = df.sample(frac=train_proportion, axis=0)
    test = df.drop(index=train.index)
    return train, test


def divide_k_folds(df, num_folds):
    ''' Inputs
            * df: dataframe containing data
            * num_folds: number of folds
        Output
            * folds: lists of folds, each fold is subset of df dataframe
    '''
    folds = []
    for subset in np.array_split(df, num_folds):
        folds.append(subset)

    return folds


def to_numpy(df):
    a = df.to_numpy()
    return a.T


def get_X_y_data(df, features, target):
    ''' Split dataframe into X and y numpy arrays '''
    X_df = df[features]
    Y_df = df[target]
    X = to_numpy(X_df)
    Y = to_numpy(Y_df)
    return X, Y


def get_rmse(vec_true, vec_pred):
    ''' Compute root mean square error between two numpy arrays '''
    rmse = np.sqrt(np.mean(np.subtract(vec_pred, vec_true)**2))
    return rmse


def get_accuracy(ytrue, ypred):
    return 1 - sum(abs(ytrue - ypred)) / len(ytrue)


def init_parameters(num_hidden, num_input, num_output):
    ''' Initialize weights and biases for neural network.
        Weights are initialized randomly, biases are set to 0
        Return: dictionary containing weights and biases
    '''
    # Use numpy arrays to create
    Weightinput = np.random.randn(num_hidden, num_input) * 0.01
    Weighthidden = np.random.randn(num_output, num_hidden) * 0.01
    Biasinput = np.zeros((num_hidden, 1))
    Biashidden = np.zeros((num_output, 1))

    parameters = {"Weightinput": Weightinput, "Weighthidden": Weighthidden,
                  "Biasinput": Biasinput, "Biashidden": Biashidden}
    return parameters


def logloss(ytrue, ypred):
    ''' Compute cross-entropy loss function over all examples '''
    num_examples = ytrue.shape[0]
    tinynum = tinynum = 10 ** -5
    totallogloss = np.dot(ytrue, math.log(ypred + tinynum).transpose()) +  \
        np.dot(1 - ytrue, math.log((1 - ypred) + tinynum).transpose())
    logloss = - np.sum(totallogloss) / num_examples

    return logloss


def normalize_features(df):
    normalized_df = (df-df.min())/(df.max()-df.min())
    return normalized_df


def normalize_feature(df, label):
    i = 0
    location = df.columns.get_loc(label)
    for i in range(len(df)):
        if df.iat[i, location] == -1:
            df.iat[i, location] = 0
    return df
