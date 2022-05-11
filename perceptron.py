

from argon2 import Parameters
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

########################## Helper functions ##################################


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


def divide_k_folds(df, k):
    ''' Inputs
            * df: dataframe containing data
            * k: number of folds
        Output
            * folds: lists of folds, each fold is subset of df dataframe
    '''
    folds = []
    for subset in np.array_split(df, k):
        folds.append(subset)
    return folds


def sgn(prod):
    if prod >= 0:
        return 1
    else:
        return -1


def get_accuracy(y_true, y_pred):
    ''' Using function from sklearn.metrics compute statistics '''
    return accuracy_score(y_true, y_pred)


def get_X_y_data(df, features, label):
    X = np.array([np.array(x) for _, x in df[features].iterrows()])
    y = np.array(df[label])
    return X, y


def perceptron(X, y, learning_rate, num_iterations):
    ''' Inputs
            * X: dataframe columns for features
            * y: dataframe column for label
            * learning_rate: by default 1
            * num_iterations: how many passesthrough all examples to do
        Output
            * w: learned weights
            * num_mistakes: average number of mistakes made per iteration (i.e.,
                through one iteration of all examples)
    '''
    weight = np.zeros(X.shape[1])
    i = 0
    num_mistakes = 0
    while i < num_iterations:
        for idx in range(len(X)):
            # Do Dot product and sum
            wt = weight.transpose()
            dot = wt.dot(X[idx])
            # Check to see if prediction is correct
            sign = sgn(dot)
            if y[idx] != sign:
                # If not equal adjust weight num_mistakes +1
                num_mistakes += 1

                weight = weight + (learning_rate * y[idx] * X[idx])

        i = i + 1

    #print(num_mistakes/num_iterations, 'num_mistakes')
    return weight, num_mistakes / num_iterations


def get_perceptron_predictions(df, features, w):
    ''' Inputs
            * df: dataframe containing data, each row is example
            * features: features used for X
            * w: weight vector
        Output
            * predictions: array of perceptron predictions, one for each row
                    (i.e., example) in df
    '''
    predictions = np.array([])
    df = df[features]
    w = w.transpose()
    for idx in range(len(df)):
        x = np.array(df.iloc[idx])
        dot = np.dot(w, x)
        sign = sgn(dot)
        predictions = np.append(predictions, sign)
    #####
    # Todo: update this function to make prediction using weights w and features
    #####

    return predictions


def get_df_num_rows(df):
    ''' Returns number of rows in dataframe '''
    return len(df)


def perceptron_cross_validation(df, num_folds, features, label, learning_rate, num_iterations):

    #####
    # Todo: Implement cross-validation and train perceptron for different best_num_iterations
    ####
    """
    listoffolds = divide_k_folds_and_remainder(df, num_folds)
    validation_accuracies = []
    validation_mistakes = []
    for fold in listoffolds:
        X1, y1 = get_X_y_data(fold[0], features, label)
        X2, y2 = get_X_y_data(fold[1], features, label)
       # print(X1, "fold1", X2, "fold2")
        weight, num_mistakes = perceptron(X2,y2, learning_rate, num_iterations)
        print(num_mistakes)
        predictions = get_perceptron_predictions(fold[0], features, weight)
        validation_mistakes.append(num_mistakes)
        validation_accuracies.append(get_accuracy(y1, predictions))
        #print(X2, validation_accuracies)
    """
    listoffolds = divide_k_folds(df, num_folds)
    validation_accuracies = []
    validation_mistakes = []
    for fold in range(num_folds):
        test_fold = listoffolds[fold]
        foldcopy = listoffolds.copy()
        foldcopy.pop(fold)
        trainingfold = pd.concat(foldcopy)

        X1, y1 = get_X_y_data(trainingfold, features, label)
       # print(X1, "fold1", X2, "fold2")
        weight, num_mistakes = perceptron(
            X1, y1, learning_rate, num_iterations)
        # print(num_mistakes)
        predictions = get_perceptron_predictions(test_fold, features, weight)
        validation_mistakes.append(num_mistakes)
        validation_accuracies.append(
            get_accuracy(test_fold[label], predictions))
        #print(get_accuracy(test_fold[label], predictions), 'accuracy')

    return sum(validation_accuracies) / num_folds, sum(validation_mistakes) / num_folds

########################## Scikit Perceptron implementation ###################


def get_scikit_perceptron_accuracy(clf, X, y):
    accuracy = clf.score(X, y)
    return accuracy


def scikit_perceptron(X, y, num_iterations):

    ####
    # Todo: return trained scikit perceptron classifier
    ####
    clf = Perceptron(max_iter=num_iterations, random_state=0)
    clf.fit(X, y)

    return clf


def get_scikit_X_y(df, features, label):
    #recoded_df = scikit_df(df, features)
    X = df[list(features)]
    y = df[label]
    return X, y


def scikit_perceptron_cross_validation(df, num_folds, features, label, num_iterations):
    # Do cross-validation
    validation_accuracies = []
    folds = divide_k_folds(df, num_folds)
    for f in range(folds):
        validation = folds[f]
        train = pd.concat(folds[:f] + folds[f+1:])

        X1, y1 = get_scikit_X_y(f[0], features, label)

        clf = scikit_perceptron(X1, y1, num_iterations)

        X, y = get_X_y_data(validation, features, label)
        accuracy = get_scikit_perceptron_accuracy(clf, X, y)
        validation_accuracies.append(accuracy)
    ####
    # Todo: implement cross-validation for scikit perceptron
    ####

    return sum(validation_accuracies) / num_folds


learning_rate = 1
num_folds = 5


def test_perceptron(train_df, test_df, features, label, num_iterations, learning_rate):
    best_num_iterations = None
    best_accuracy = -1
    for k in num_iterations:
        validation_accuracy, validation_mistakes = perceptron_cross_validation(
            train_df, num_folds, features, label, learning_rate, k)
        print('num_iterations:', k, ', validation accuracy:',
              validation_accuracy, ', validation mistakes:', validation_mistakes)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_num_iterations = k

# Accuracy on training and testing data
    print('Best_Num_Iterations: ', best_num_iterations,
          'Best Learning Rate: ', learning_rate)
    X, y = get_X_y_data(train_df, features, label)
    w, train_mistakes = perceptron(X, y, learning_rate, best_num_iterations)
    predictions = get_perceptron_predictions(train_df, features, w)
    train_accuracy = get_accuracy(train_df[label], predictions)
    predictions = get_perceptron_predictions(test_df, features, w)
    test_accuracy = get_accuracy(test_df[label], predictions)
    print('train accuracy:', train_accuracy, ', test accuracy:', test_accuracy)

    return train_accuracy, test_accuracy, best_num_iterations
