#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    feature engineering
    ~~~~~~~~~~~~~~~~~~~~

    Module to create new features
    for the machine learning pipeline.
"""
import pandas as pd
import numpy as np

def email_ratios(datadict):
    """

    Parameters
    ----------

    Returns
    -------
    """

    # Convert data dictionary, tranpose
    # to have columns as variables
    df = pd.DataFrame(datadict)
    df = df.replace('NaN', np.NaN)

    df = df.transpose()

    df["ratio_to_poi"] = df["from_this_person_to_poi"].dropna()/df["from_messages"].dropna()
    df["ratio_from_poi"] = df["from_poi_to_this_person"].dropna()/df["to_messages"].dropna()

    df = df.replace(np.NaN, 0)

    # Tranpose back before convertin back to a
    # dictionary
    df = df.transpose()
    data_dict = df.to_dict(orient='dict')

    return data_dict
