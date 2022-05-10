'''
Wesleyan University, COMP 343, Spring 2022
Homework 8: Neural Network Training
Name: Peter Fulweiler
'''


# Python modules
import numpy as np
import pandas as pd
import random
import sys
import time
import util

# Project modules


def main():

    random.seed(time.time())

    # Load data
    features = {"Ticket Class", "Sex", "Age", "# of siblings and spouses aboard",
                "# of parents and children aboard", "Fare", "Embarked Cherbourg?",
                "Embarked Queenstown?", "Embarked Southampton?"}
    target = "Survived"
    filename = 'titanic-1.csv'
    df = util.load_data(filename)

    # Split data into training and test
    train_proportion = 0.70
    train_df, test_df = util.split_data(df, train_proportion)
    print(train_df)

    # Complete the forward pass


if __name__ == '__main__':
    main()
