#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    feature selection
    ~~~~~~~~~~~~~~~~~

    This module allows for automated feature
    selection during a machine learning pipeline.
"""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from . import feature_format


def get_fs_clf():
    """
    Get a classifier to be used
    for feature selection.
    """
    clf_fs = AdaBoostClassifier()

    return clf_fs


def selection(dataset, feature_list, clf, cut_off=0.01):
    """
    Creates a new list of features after
    removing the lowest important features.

    Uses a cut off value to decide which features
    to eliminate.

    Cut off should vary depending on which type
    of classifier is used.

    Parameters
    ----------
    data

    Returns
    -------

    """

    # Extract features and labels from dataset for local testing
    data = feature_format.featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = feature_format.targetFeatureSplit(data)

    clf.fit(features, labels)
    f_weight = clf.feature_importances_

    # poi removed as this has been seperated into the label.
    del feature_list[0]

    df_f = pd.DataFrame(feature_list)

    df_f["Feature_importance"] = f_weight

    df_f = df_f[df_f.Feature_importance > cut_off]

    # Convert the feature names back into a list.
    fs_list = df_f.iloc[:, 0].tolist()

    return fs_list
