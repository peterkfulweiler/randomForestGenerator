'''
Wesleyan University, COMP 343, Spring 2022
final project: checkpoint1
Name: Andres Cojuangco and Peter Fulweiler
'''

# Python modules
import numpy as np
import pandas as pd
import random
import sys
import time
from scipy import stats

# Helper libraries
import perceptron
import decisiontree
import util

"""
Helper functions will iniclude scikit's split_data function, scikit's get_accuracy, get_X_y_data, and
the get_ID3_accuracy and get_ID3_num_correct functions that we implemented in
class.
"""

# Bootstrap


def bootstrapping(df, num_df):
    """
    Takes a train dataset and gets x number random dataframe samples with replacement
    (i.e. some examples can be present in mulitple samples). Each bootstrapped
    dataset will have y number of examples.
    Returns an array of subsets of the original dataframe
    """
    for i in range(num_df):
        indices = np.random.randint(0, len(df), size=num_df)
        df_bootstrapped = df.iloc[indices]
    return df_bootstrapped
# Everything for Decision Trees

# Making predictions with only decision trees


def build_decision_trees():
    """
    Builds a decision trees from each bootstrapped dataset using a
    random subspace of the features (x features out of the total number of
    features).
    """


def ID3_decision_tree_all():
    """
    Takes modified ID3 decision tree algorithm to make predictions for each
    decision tree. This makes predictions for all examples in a dataframe
    """


def ID3_decision_tree_prediction():
    """
    Takes modified ID3 decision tree algorithm to make predictions for each
    decision tree. This makes a prediction for one example in a dataframe.
    """

# Everything for Perceptrons


# Making predictions with only perceptrons

def perceptron_forrest(train_df, features, label, n_submodels, n_bootstrap, n_features, num_iterations, learning_rate):
    """
    Creates a Random Perceptron Forest with a training dataset
    returns features and weights
    """
    perceptronforest = []
    # Iterate through number of models
    for i in range(n_submodels):
        # Randomize features
        random_features = util.randomize_features(features, n_features)
        # Bootstrap dataset
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        # Get X Y data from random features and bootstraped df
        #print(random_features, ": Rand features")
        X, Y = util.get_X_y_data(df_bootstrapped, random_features, label)
        # Get Weights
        #print(X, ": X")
        percept_w = perceptron.perceptron(
            X.transpose(), Y, learning_rate, num_iterations)
        # return weights and random_features
        perceptronforest.append([percept_w, random_features])
        # print(random_features)
    return perceptronforest


def get_perceptron_all(df, forest, label):
    """
    Get's perceptron predictions for all examples in a dataframe "Votes" on most common vote.
    inputs:
    Df: df to test on
    forest: List of weights and features subspace gotten for each submodel in the forrest.
    """
    i = 0
    predict = np.array([])
    for i in range(len(forest)):
        # Get features for specific model
        # print("######### Forest shapes #########",forest[i][1], forest[i][0][0])
        predictions = perceptron.get_perceptron_predictions(
            df, forest[i][1], forest[i][0][0])
        predict = np.append(predict, [predictions])
    # Transpose
    predicts = predictions.transpose()
    finalpredictions = []
    for x in range(len(predicts)):
        # Find most common element for each example
        vote = stats.mode(predicts[x])
        # Append common element to final predicts
        finalpredictions.append(vote[0][0])

    # return final predictions

    accuracy = perceptron.get_accuracy(df[label], finalpredictions)
    return finalpredictions, accuracy

    ######## PERCEPTRON FOREST #############


def perceptron_forrest(train_df, features, label, n_submodels, n_bootstrap, n_features, num_iterations, learning_rate):
    """
    Creates a Random Perceptron Forest with 
    """
    perceptronforest = []
    # Iterate through number of models
    for i in range(n_submodels):
        # Randomize features
        random_features = util.randomize_features(features, n_features)
        # Bootstrap dataset
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        # Get X Y data from random features and bootstraped df
        #print(random_features, ": Rand features")
        X, Y = util.get_X_y_data(df_bootstrapped, random_features, label)
        # Get Weights
        #print(X, ": X")
        percept_w = perceptron.perceptron(
            X.transpose(), Y, learning_rate, num_iterations)
        # return weights and random_features
        perceptronforest.append([percept_w, random_features])
        # print(random_features)
    return perceptronforest


def get_hyper_parameters(train_df, test_df, features, label, num_iterations, learning_rate, num_models, num_straps, num_features):
    """
    Gets Best hyperparameters, num models num features, num iterations , and learning rate from a training dataset
    also gets accuracy on testing with best hyperparameters
    """
    trainaccuracy, testaccuracy, bestnumiterations, best_lr = perceptron.test_perceptron(
        train_df, test_df, features, label, num_iterations, learning_rate)
    bestaccuracy = -1
    best_model = None
    best_num_features = None
    for i in range(1, num_models):
        for j in range(1, num_features):
            forest = perceptron_forrest(
                train_df, features, label, i, num_straps, j, bestnumiterations, best_lr)
# print("############## Forest ################# ")
#print(forest, ": Forest")
# print(len(forest))
            print("############## Predictions ################# ")
            predictions, accuracy = get_perceptron_all(test_df, forest, label)
            print("Number of Models: ", i, "Number of features: ",
                  j, "Accuracy: ", accuracy)

            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                best_model = i
                best_num_features = j

    print("Best Train Accuracy 1 Perceptron with all features: ", trainaccuracy)
    forest = perceptron_forrest(train_df, features, label, best_model,
                                num_straps, best_num_features, bestnumiterations, best_lr)
    predictions, accuracy = get_perceptron_all(test_df, forest, label)
    print("Best num of models: ", best_model,
          "Best num features: ", best_num_features)
    print("Best Testing Accuracy with 1 Perceptron model with all features: ", testaccuracy)
    print("Best Testing Accuracy with Perceptron Forest with best hyperparameters: ", accuracy)

############ COMBINATION FOREST ###############


def submodel_combination(train_df, features, label, n_submodels, n_bootstrap, n_features, max_depth, num_iterations, learning_rate):
    """
    Takes half of the bootstrapped dataframes and makes predictions using decision
    trees and the other half makes predictions using perceptrons.
    """
# Prediction
    forest = []
    perceptronforest = []
    i = 0
    # Iterate through number of models
    for i in range(n_submodels):
        # Randomize Features
        random_features = util.randomize_features(features, n_features)
        # If were in the first half of models create decisions trees
        if (i < n_submodels/2):
            df_bootstrapped = bootstrapping(train_df, n_bootstrap)
            tree_id3 = decisiontree.ID3_decision_tree(
                df_bootstrapped, random_features, label, max_depth, n_features)
            forest.append(tree_id3)
        # If wee in the second half of models create mini perceptrons This way it is half perceptron
        # half decision tree
        else:
            df_bootstrapped = bootstrapping(train_df, n_bootstrap)
            # Split X y on Random_features
            X, Y = util.get_X_y_data(df_bootstrapped, random_features, label)
            # Call perceptron
            percept_w = perceptron.perceptron(
                X, Y, learning_rate, num_iterations)
            # Append trained weights and random features
            # features are to make predictions later on.
            perceptronforest.append([percept_w, random_features])

    return forest, perceptronforest


def final_prediction():
    """
    Tallies the predictions of the models and chooses the majority prediction
    """

############# RANDOM FOREST ##############


def random_forest_algorithm(train_df, features, label, n_trees, n_bootstrap, n_features, max_depth):
    """ Inputs:
    * train_df: the training dataset
    * features:
    * n_trees...
    Output:
    * forest: list of decision trees aka the random forest 
    """
    forest = []
    for tree in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree_id3 = decisiontree.ID3_decision_tree(
            df_bootstrapped, features, label, max_depth, n_features)
        forest.append(tree_id3)
    return forest


# Data visualization


"""
Will use a line plot to plot the accuracies, recall, and precision of each set of submodels for bagging.
One set will be all decision trees. One set will be all perceptrons. The last Set
will be half perceptrons and half decision trees. We will be using two datasets:
1 linearly separable data set and the other non-linearly separable data set.
"""
