"""
This module is used for loading the datasets from CSV. The CSV-Files are integrated into the library.
"""

import pandas as pd
import os

d = os.path.dirname(__file__)


##############
# REGRESSION #
##############


def load_regression_train():
    X = pd.read_csv(d + "/reg/train_features.csv")
    y = pd.read_csv(d + "/reg/train_label.csv")
    df = pd.merge(X, y)

    # Drop id column, since it is already in the index of the df
    return df.drop(columns=["Id"])


def load_regression_test():
    X = pd.read_csv(d + "/reg/test_features.csv")

    # Drop id column, since it is already in the index of the df
    return X.drop(columns=["Id"])


##############
# CLASSIFICATION #
##############


def load_classification_train(X_y_split=False):
    X = pd.read_csv(d + "/clf/train_features.csv")
    y = pd.read_csv(d + "/clf/train_label.csv")

    if X_y_split:
        # Drop id column, since it is already in the index of the df
        return X.drop(columns=["Id"]), y.drop(columns=["Id"])

    df = pd.merge(X, y)

    # Drop id column, since it is already in the index of the df
    return df.drop(columns=["Id"])


def load_classification_test():
    X = pd.read_csv(d + "/clf/test_features.csv")

    # Drop id column, since it is already in the index of the df
    return X.drop(columns=["Id"])
