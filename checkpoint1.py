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

# Helper libraries
import perceptron
import decisiontree

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

def perceptron():
    """
    Perceptron algorithm that gets learned weights and average number of mistakes
    per iteration
    """


def get_perceptron_all():
    """
    Get's perceptron predictions for all examples in a dataframe
    """


def get_perceptron_prediction():
    """
    Get's perceptron prediction for one example in a dataframe
    """

# Making predictions with both perceptrons and decision trees


def submodel_combination():
    """
    Takes half of the bootstrapped dataframes and makes predictions using decision
    trees and the other half makes predictions using perceptrons.
    """
# Prediction


def final_prediction():
    """
    Tallies the predictions of the models and chooses the majority prediction
    """

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
        tree_id3 = ID3_decision_tree(df_bootstrapped, features, label, max_depth,n_features)
        forest.append(tree_id3)
    return forest


# Data visualization


"""
Will use a line plot to plot the accuracies, recall, and precision of each set of submodels for bagging.
One set will be all decision trees. One set will be all perceptrons. The last Set
will be half perceptrons and half decision trees. We will be using two datasets:
1 linearly separable data set and the other non-linearly separable data set.
"""
