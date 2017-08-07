#!/usr/bin/python

from __future__ import print_function
import pickle
import os
from sklearn.cross_validation import train_test_split
from tester import dump_classifier_and_data
from learnEnron import (
                        feature_format,
                        feature_engineering,
                        feature_selection,
                        feature_scaling,
                        tune
                        )

ro = True  # Outlier selection
fs = True  # Feature selection
fe = True  # Feature engineering
sc = True  # Feature scaling
tu = True  # Cross validation and parameter optimization

gb = False  # Use gradient boosting
lr = True  # Use logistic regression
pipe = True  # Use pipe > anova features > pca > classifer


# Features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi',
                 'bonus',
                 'deferral_payments',
                 'deferred_income',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'loan_advances',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'salary',
                 'shared_receipt_with_poi',
                 'total_payments',
                 'total_stock_value',
                 "ratio_to_poi",
                 "ratio_from_poi"
                 ]

# Use os.path.abspath to access the file
file_dir = os.path.dirname(os.path.realpath(__file__))
f = os.path.join(file_dir, "resources", "data", "final_project_dataset.pkl")

# Changed to rb for python to read binary
with open(f, "rb") as data_file:
    data_dict = pickle.load(data_file)

# Remove outliers
if ro:
    data_dict.pop("TOTAL", None)
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", None)

# Feature engineering
if fe:
    data_dict = feature_engineering.email_ratios(data_dict)

# Feature selection
if fs:
    # Chooses an AdaBoost classifier for feature selection.
    #
    # Overwrite this to try different SkLearn Classifiers.
    clf_fs = feature_selection.get_fs_clf()
    # Overwrite feature list after feature selection
    features_list = feature_selection.selection(
                                                 data_dict,
                                                 features_list,
                                                 clf_fs,
                                                 cut_off=0.01
                                                 )

    print("Features to be used after feature selection:")
    print(features_list)

# Feature scaling
if sc:
    data_dict = feature_scaling.scale(data_dict, features_list)
    print("All features scaled")

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = feature_format.featureFormat(
                                    my_dataset,
                                    features_list,
                                    sort_keys=True
                                    )
labels, features = feature_format.targetFeatureSplit(data)

# No train test split is used.
#
# Final validation uses tester.py to compare to a test set.
#
# GridSearch CV involves an inner-loop stratified K-Fold
# cross validation.
features_train = features
label_train = labels

# Tune the classifier to achieve better than .3 precision and recall
#
# View the tune module for details on full implementation.
if tu:

    if gb:
        clf = tune.param_optimize_gb(features_train, labels_train,
                                     grid_search=False)

    if lr:
        if pipe:
            clf = tune.param_optimize_lr_pipe(features_train, labels_train,
                                              grid_search=True, folds=3)
        else:
            clf = tune.param_optimize_lr(features_train, labels_train,
                                         grid_search=True)

# Dump the classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
