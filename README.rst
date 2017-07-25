===============================
Identify Fraud from Enron Email
===============================

*Adam McCarthy*

*Problem posed in Udacity Intro to machine learning*

Getting Started
---------------

To test results:

.. code-block:: bash
    
    $ cd final_project
    $ tester.py

To re-run:

.. code-block:: bash

    $ cd final_project
    $ poi_id.py


Results
-------

Using the default setting of one label and one feature we can take an intitial review. of the prediction.

.. code-block:: python

    features_list = ['poi', 'salary']

The outputs for the initial algorithm (Gaussian Naive Bayes) is compared to three other algorithms.

.. csv-table:: Algorith comparisson
   :header: "Algorithm", "Accuracy", "Precision", "Recall", "F1", "F2", "Tot. pred.", "True pos.", "False pos.", "False neg.", "True neg."
   :widths: 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5

   "GaussianNaiveBayes", 0.256, 0.185, 0.798, 0.300, 0.480, 10000, 1596, 7040, 404, 960
   "RandomForest", 0.705, 0.223, 0.191, 0.205, 0.197, 10000, 382, 1328, 1618, 6672
   "AdaBoost", 0.719, 0.246, 0.196, 0.217, 0.204, 10000, 391, 1201, 1609, 6799
   "KMeans", 0.738, 0.043, 0.013, 0.020, 0.015, 370, 1, 22, 75, 272

Adaboost performs considerably slower.

KMeans gives warning about predicted labels not equal to 0 or 1.

Naive Bayes gives a very high recall valye (0.798).

Questions
---------


No1
~~~

Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]


No2
~~~

What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]


No3
~~~

What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]


No4
~~~

What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]


No5
~~~

What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]


No6
~~~

Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]


Code issues and Python 2 to 3 changes
-------------------------------------

^^^^^^^^^^^^^
File Location
^^^^^^^^^^^^^

Kept getting errors about not being able to locate the file based off of the string in the original code.
Changed to:

.. code-block:: Pythoon

    f = os.path.abspath("final_project/final_project_dataset.pkl")

^^^^^^
Pickle
^^^^^^

Changed code in both poi_id.py and tester.py to fit with python 3 and pickle otherwise a TypeError is returned.
Now has to include "rb" (read binary) and "wb" (write binary) instead of "r" and "w" respectively.

From:

.. code-block:: Python

   with open(f, "r") as data_file:
       data_dict = pickle.load(data_file)

To:

.. code-block:: Python

    with open(f, "rb") as data_file:
        data_dict = pickle.load(data_file)

^^^^^^^^^^^^^^^^^^
Depreciation of CV
^^^^^^^^^^^^^^^^^^

Code returns this warning.

    DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functio
    ns are moved. Also note that the interface of the new CV iterators are different from that of this module. This module w
    ill be removed in 0.20.

Changed to:

.. code-block: python

    cv = model_selection.StratifiedShuffleSplit(labels, folds, random_state=42)

.. code-block: python

    features_train, features_test, labels_train, labels_test = \
    model_selection.train_test_split(features, labels,
                                      test_size=0.3,
                                      random_state=42)

^^^^^^
urllib
^^^^^^

urlib has been changed in python 3, needed within startup.py
New code requires urllib.request module.

Changed to:

.. code-block: python

    urllib.request.urlretrieve(url, filename="../enron_mail_20150507.tgz")

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.
