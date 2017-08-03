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
from sklearn import (
                     ensemble,
                     linear_model,
                     feature_selection,
                     pipeline,
                     decomposition
                     )


def param_optimize_gb(features, labels, grid_search=True):
    """
    Hyper parameter optimization
    through parameter grid search
    during cross validation.

    Tailored for a GradientBoosting classifer.

    Parameters
    ----------
    datadict = dict
        Containing the data within a dictionary
    feature_list = list
        Contains all varibles to be scaled.

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

    clf = ensemble.GradientBoostingClassifier()

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

    clf = clf.best_estimator_

    return clf


def param_optimize_lr(features, labels, grid_search=True, folds=2):
    """
    Hyper parameter optimization
    through parameter grid search
    during cross validation.

    Tailored for a Logistic Regression classifer.

    Parameters
    ----------
    datadict = dict
        Containing the data within a dictionary
    feature_list = list
        Contains all varibles to be scaled.

    Returns
    -------
    clf_f = sklearn clf object
        Fitted classifier including the hyper
        parameter optimization.
    """

    # How many splits
    n = folds
    cv = StratifiedKFold(n_splits=n, shuffle=True)

    # Which metric should be used to optimize the
    # cross validation
    score = "recall"

    clf = linear_model.LogisticRegression()

    parameters = [{
                   "C": [0.01, 0.1, 1, 10, 100],
                   }]

    # Parameters based on best results from exhaustive grid
    # search of all parameters.
    if grid_search is not True:
        parameters = [{
                       "C": [10]
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

    clf = clf.best_estimator_

    return clf


def param_optimize_lr_pipe(features, labels, grid_search=True, folds=2):
    """
    Hyper parameter optimization
    through parameter grid search
    during cross validation.

    Tailored for a Logistic Regression classifer.
    Stored as a pipeline and including,
    feature selction and PCA decomposition.

    Parameters
    ----------
    datadict = dict
        Containing the data within a dictionary
    feature_list = list
        Contains all varibles to be scaled.

    Returns
    -------
    clf_f = sklearn clf object
        Fitted classifier including the hyper
        parameter optimization.
    """

    # Create an anova feature selection for classification.
    anovafilter = feature_selection.SelectKBest(feature_selection.f_classif)

    # Set principal component analysis
    pca = decomposition.PCA()

    # How many splits
    n = folds
    cv = StratifiedKFold(n_splits=n, shuffle=True)

    # Which metric should be used to optimize the
    # cross validation
    score = "f1_weighted"

    lrclf = linear_model.LogisticRegression()

    # Store all steps into pipelines
    estimators = [("anova", anovafilter), ("r_dim", pca), ("clf", lrclf)]
    pipe = pipeline.Pipeline(estimators)

    # Using <estimator>__<parameter> syntax to adjust parameters
    # within the pipeline.
    #
    # Pipeline steps can be ignored by setting None.

    parameters = [{
                   "anova__k": [6, 8, 10, 12, "all"],
                   "r_dim__n_components": [2, 4],
                   "r_dim__whiten": [True, False],
                   "clf__C": [0.01, 0.1, 1, 10, 100],
                   "clf__class_weight": ["balanced"],
                   }]

    # Parameters based on best results from exhaustive grid
    # search of all parameters.
    if grid_search is not True:
        parameters = [{
                       "C": [10]
                       }]

    clf = GridSearchCV(
                       estimator=pipe,
                       param_grid=parameters,
                       cv=cv,
                       scoring=score
                       )

    # Will take time...
    clf.fit(features, labels)

    print("Best classifier score:", clf.best_score_, ":", clf.best_params_)

    clf = clf.best_estimator_

    return clf
