#!/usr/bin/python
"""
    ml plots
    ~~~~~~~~~
    Module to store plots to investigate dataset
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
from learnEnron.feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

sys.path.append("../tools/")


def regression_plot(data_dict, target, variable,
                    both_lines=True, split_size=0.5):
    """
    Make a scatter plot example of two
    a two variable scatter plot.

    Function based on the regression mini-project.

    Will split the data into two to investigate
    how robust a linear regression will be.

    Parameters
    ----------

    data_dict = dict
        dict of data, this will not with a pandas dataframe
    target = string
        feature to be predicted from within data_dict
    variable = string
        feature to be used to make prediction from
        within data_dict
    both_lines = Boolean
        True sets regression lines for both splits to be
        plotted. False sets only the first split to be plotted.
    """

    sns.set_style("whitegrid")
    sns.set_style("ticks",
                  {'axes.grid': True,
                   'grid.color': '.99',  # Very faint grey grid
                   'ytick.color': '.4',  # Lighten the tick labels
                   'xtick.color': '.4'}
                  )
    sns.set_context(
                    "poster",
                    font_scale=0.8,
                    rc={'font.sans-serif': 'Gill Sans MT',
                        "lines.linewidth": 2}
                    )

    # List the features you want to look at.
    # First item in the
    # list will be the "target" feature.
    features_list = [target, variable]
    data = featureFormat(
                         data_dict,
                         features_list,
                         remove_any_zeroes=True
                         )
    target, features = targetFeatureSplit(data)

    train_color = "#E24E42"
    test_color = "#008F95"

    # Training-testing split needed in regression.
    (feature_train,
     feature_test,
     target_train,
     target_test) = train_test_split(
                                     features,
                                     target,
                                     test_size=split_size,
                                     random_state=42
                                     )

    # Fit Linear Regression model.
    reg = LinearRegression()
    reg.fit(feature_train, target_train)

    coefficient = reg.coef_
    intercept = reg.intercept_
    print("split 1: y={0}x + {1}"
          .format(coefficient,
                  intercept)
          )

    # Draw the scatterplot.
    # Color-coded training and testing points
    for feature, target in zip(feature_test, target_test):
        plt.scatter(feature, target, color=test_color)
    for feature, target in zip(feature_train, target_train):
        plt.scatter(feature, target, color=train_color)

    # Labels for the legend
    plt.scatter(
                feature_test[0],
                target_test[0],
                color=test_color,
                label="Split 1"
                )
    plt.scatter(
                feature_test[0],
                target_test[0],
                color=train_color,
                label="Split 2"
                )

    # Draw the regression line, once it's coded
    try:
        plt.plot(feature_test,
                 reg.predict(feature_test),
                 color=train_color)
 
        if both_lines:
            reg.fit(feature_test, target_test)
            plt.plot(
                     feature_train,
                     reg.predict(feature_train),
                     color=test_color
                     )
            coefficient = reg.coef_
            intercept = reg.intercept_
            print("split 2: y={0}x + {1}"
                  .format(coefficient, intercept)
                  )
    except NameError:
        pass

    plt.xlabel(features_list[1])
    plt.ylabel(features_list[0])
    plt.legend()
    plt.show()
