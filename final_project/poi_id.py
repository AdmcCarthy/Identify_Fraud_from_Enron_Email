#!/usr/bin/python

from __future__ import print_function
import pickle
import os
from learnEnron import feature_format
from tester import dump_classifier_and_data
from sklearn import (
                     model_selection, naive_bayes,
                     ensemble, cluster, linear_model
                    )

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary']

# Use os.path.abspath to access the file
file_dir = os.path.dirname(os.path.realpath(__file__))
f = os.path.join(file_dir, "final_project_dataset.pkl")

# Changed to rb for python to read binary
with open(f, "rb") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
data_dict.pop("TOTAL", None)

# Task 3: Create new feature(s)


# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = feature_format.featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = feature_format.targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html


# Set classifier
clf = naive_bayes.GaussianNB()
# clf = ensemble.RandomForestClassifier()
# clf = ensemble.AdaBoostClassifier()
# clf = cluster.KMeans()



# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    model_selection.train_test_split(features, labels,
                                     test_size=0.3,
                                     random_state=42)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
