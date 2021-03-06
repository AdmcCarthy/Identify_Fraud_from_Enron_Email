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
    Add new features into the data
    dictionary based on email data.

    Create a ratio based on the number
    of emails sent to a person and the
    number of emails sent to this person
    from a person of interst (POI).

    Create a ratio based on the number of
    emails from a person and the number of
    emails this person to sent to a POI.

    Parameters
    ----------
    datadict = dictionary
        A dictionary storing all of the dataset.
        Each key relates to a person while the value
        is a dictionary containing all of the variables.
    Returns
    -------
    data_dict = dictionary
        Almost identical to input but with two new variables
        added.
    """

    # Convert data dictionary, tranpose
    # to have columns as variables
    df = pd.DataFrame(datadict)
    df = df.replace('NaN', np.NaN)
    df = df.transpose()

    df["ratio_to_poi"] = (
                          df["from_this_person_to_poi"].dropna()
                          /df["from_messages"].dropna()
                          )
    df["ratio_from_poi"] = (
                            df["from_poi_to_this_person"].dropna()
                            /df["to_messages"].dropna()
                            )

    # Replace NaNs with zeros to so the pipeline works
    df = df.replace(np.NaN, 0)

    # Tranpose back before convertin back to a
    # dictionary
    df = df.transpose()
    data_dict = df.to_dict(orient='dict')

    return data_dict
