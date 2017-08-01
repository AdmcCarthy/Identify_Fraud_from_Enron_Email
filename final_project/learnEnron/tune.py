#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    tune
    ~~~~

    Tune the classifier using cross-validation
    and grid search across a range of parameter
    options.

    This will optimize the parameters used for
    machine learning algorithm.
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold


def param_optimize(features, labels, clf, grid_search=True):
    """
    Hyper parameter optimization
    through parameter grid search
    during cross validation.

    Parameters
    ----------
    datadict = dict
        Containing the data within a dictionary
    feature_list = list
        Contains all varibles to be scaled.
    clf = sklearn clf object
        clf type to be applied during grid search
    params = dict
        Parameters to be used in GridSearchCv,
        must be suitable for the choosen
        classifier.

    Returns
    -------
    clf_f = sklearn clf object
        Fitted classifier including the hyper
        parameter optimization.
    """

    # How many splits
    n = 2
    cv = StratifiedKFold(n_splits=n, shuffle=True)

    # Which metric should be used to optimize the
    # cross validation
    score = "f1_weighted"

    parameters = [{
                   "loss": ["deviance", "exponential"],
                   "n_estimators": [120, 300, 500, 800, 1200],
                   "max_depth": [3, 5, 7, 9, 12, 15, 17, 25],
                   "min_samples_split": [2, 5, 10, 15, 100],
                   "min_samples_leaf": [2, 5, 10],
                   "subsample": [0.6, 0.7, 0.8, 0.9, 1],
                   "max_features": ["sqrt", "log2", None]
                   }]

    # Parameters based on best results from exhaustive grid
    # search of all parameters.
    if grid_search is not True:
        parameters = [{
                       'subsample': [0.8],
                       'n_estimators': [120],
                       'max_depth': [25],
                       'loss': ['deviance'],
                       'min_samples_split': [2],
                       'min_samples_leaf': [2],
                       'max_features': ['sqrt']
                       }]

    clf = GridSearchCV(
                       estimator=clf,
                       param_grid=parameters,
                       cv=cv,
                       scoring=score
                       )

    # Will take time...
    clf.fit(features, labels)

    print("Best classifier score:", clf.best_score_, ":", clf.best_params_)

    return clf
