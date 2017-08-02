#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    feature scaling
    ~~~~~~~~~~~~~~~

    Module to scale features
    before being used in a machine
    learning pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def scale(datadict, feature_list):
    """
    Scale features within the data dictionary.

    Most machine learning approaches require
    data to be close to zero mean and unit variance.

    Uses a robust scaler to account for outliers
    within the dataset.

    Parameters
    ----------
    datadict = dict
        Containing the data within a dictionary
    feature_list = list
        Contains all varibles to be scaled.
    Returns
    -------
    data_dict = dict
        Data dicitonary after variable scaling.
    """

    # Convert data dictionary, tranpose
    # to have columns as variables
    df = pd.DataFrame(datadict)
    df = df.replace('NaN', np.NaN)
    df = df.transpose()

    # Replace NaNs with zeros to so the pipeline works
    df = df.replace(np.NaN, 0)
    a = df['exercised_stock_options'].mean()

    # Robust scaler due to outliers in data
    scl = RobustScaler()

    # Return scaled featured back to DataFrame
    df[feature_list] = scl.fit_transform(df[feature_list])

    # Report results
    b = df['exercised_stock_options'].mean()
    print("Mean changed from {0} to {1}".format(a, b))

    # Tranpose back before convertin back to a
    # dictionary
    df = df.transpose()
    data_dict = df.to_dict(orient='dict')

    return data_dict
