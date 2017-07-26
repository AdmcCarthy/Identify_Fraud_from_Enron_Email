#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    figures
    ~~~~~~~

    This module provides common plotted graphs following a minimal style,
    settings are designed to effectively communicate the data.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from .transformation import (
                            log_10, sq_rt
                            )

# Color schemes
custom_bw = ['#192231', '#3C3C3C', '#CDCDCD', '#494E6B']


def common_set_up(ax_size):
    """Applies plot set up and style to a figure.

    Parameters
    ----------
    ax_size : tuple
        tuple containing ax size. First value is
        width, second value is height.
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
                    rc={"figure.figsize": ax_size,
                        'font.sans-serif': 'Gill Sans MT'}
                    )


def formatting_text_box(ax, parameters, formatting_right):
    """ Draws within the ax(axes within figures) a
    text box describing all parameters used.

    Parameters
    ----------
    ax : matplotlib axes
        axes of the figure in which the text
        box is applied.
    parameters : string
        multiline string
    formatting_right : boolean
        True or False

    Returns
    -------
    ax : matplotlib axes
    """

    font_colour = '#9099A2'  # Light grey

    # Set text box to be white with 50% transparency
    # will not be seen unless overlapping data
    text_box_patch = dict(
                          boxstyle='round',
                          facecolor='white',
                          alpha=0.5,
                          edgecolor='white')

    # Text box position to avoid overlap
    # with graphs data.
    if formatting_right:
        box_vertical = 0.83
        box_horizontal = 0.845
    else:
        box_vertical = 0.83
        box_horizontal = 0.05

    ax.text(
            box_horizontal, box_vertical, parameters,
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', color=font_colour,
            bbox=text_box_patch
            )

    return ax


def annotation_text(ax, string, vert_pos, horz_pos,
                    color_set=custom_bw,
                    strong_color=True,
                    font_size=12):
    """ Adds a text box with an annotation onto the
    ax(axes within figures)

    Parameters
    ----------
    ax : matplotlib axes
        axes of the figure in which the text
        box is applied.
    string : string
        text string
    vert_pos : int/float
        co-ordinate position within ax
    horz_pos : int/float
        co-ordinate position within ax
    color_set : list
        list of four matplotlib acceptable colors
    strong_color : boolean
        True or False, True equals dark grey,
        False always equals light pale grey.
    font_size : int
        Interger for size of the font.

    Returns
    -------
    ax : matplotlib axes
    """

    if strong_color:
        font_c = color_set[0]
    else:
        font_c = '#9099A2'  # Light pale grey

    # Text box set up
    text_box_patch = dict(
                          boxstyle='round',
                          facecolor='white',
                          alpha=0.2,
                          edgecolor='white')

    ax.text(
            horz_pos, vert_pos, string,
            transform=ax.transAxes,
            fontsize=font_size,
            verticalalignment='top',
            color=font_c,
            bbox=text_box_patch
            )

    return ax


def univariate(
               x,
               univariate_name,
               color_set=custom_bw,
               bin_n='all_values',
               ax_size=(12, 6),
               rug=True,
               formatting_right=True,
               x_truncation_upper=None,
               x_truncation_lower=None,
               ax=None
               ):
    """
    Create a histogram and kernel density estimate
    plot to show the univariate distribution
    of a variable.

    Includes a rug plot showing a series of vertical
    ticks just along the x-axis showing the distribution
    of each individual value.

    Includes text communicating all settings applied to
    figure.

    Returns an object to be plotted.

    Parameters
    ----------
    x : array_like
        list of values, pandas series or single
        column from a pandas dataframe.
    univariate_name : string
        name of variable, include units or other
        information to be displayed in plot
    color_set : list
        list of three colors to be used in plot
    bin_n : string/None/int
        Default is 'all_values' which calculates
        the range of values to be used as the number
        of bins.
        None means automatic selection.
        Interger will set bin numbers manually.
    ax_size : tuple
        tuple containing ax size. First value is
        width, second value is height.
    rug : boolean
        True uses rug, False turns it off
    formatting_right : boolean
        True is place parameters on right of figure.
        False places them to the left of figure.
    x_truncation_upper : None/int/float
        Number to set upper limit of the x-axis.
        None means automatically set.
    x_truncation_lower : None/int/float
        Number to set lower limit of the x-axis.
        None means automatically set.

    Returns
    -------
    fig : matplotlib axes
        Will plot an individual figure or ax
        can be reused within a figure containing
        more than one subplot.
    """

    common_set_up(ax_size)  # Apply basic plot style

    # Calulate the range of values
    # and use this as the number of bins.
    #
    # Does not work well if values are not intergers
    # or between 0 and 1.
    if bin_n == 'all_values':
        x_max = x.max()
        x_min = x.min()
        bin_n = int(x_max)-int(x_min)

    fig = sns.distplot(
                       x,
                       bins=bin_n,
                       rug=rug,
                       ax=ax,
                       hist_kws={"histtype": "bar",
                                 "linewidth": 1,
                                 'align': 'mid',
                                 'log': False,
                                 'edgecolor': 'white',  # Edge hist. bars.
                                 "alpha": 1,
                                 "color": color_set[2],
                                 'label': 'Histogram'},  # Legend label
                       kde_kws={"color": color_set[0],
                                "lw": 3,
                                "label": "KDE"},  # Legend label
                       rug_kws={"color": color_set[1],
                                'lw': 0.3,
                                "alpha": 0.5,
                                'height': 0.05}
                        )

    title_color = '#192231'  # Dary grey
    font_colour = '#9099A2'  # Light grey

    # Let title state when rug plot is active
    # as it will not display in legend.
    if rug:
        rugstr = ', with rug plot'
    else:
        rugstr = ''

    # Do not add a title in a multi-figure plot.
    #
    # Title will be added to figure with all sub-plots
    # instead in this case.
    if ax is None:
        fig.set_title(
                      ('Distribution of {0}'.format(univariate_name) + rugstr),
                      fontsize=20,
                      color=title_color
                      )
    fig.set_xlabel(
                   '{0}'.format(univariate_name),
                   color=font_colour
                   )
    fig.set_ylabel(
                   'Frequency',
                   color=font_colour
                   )

    # Apply limits to the x axis.
    if x_truncation_upper or x_truncation_lower:
        axes = fig.axes
        axes.set_xlim(x_truncation_lower, x_truncation_upper)
        # To be communicated back in Formatting notes
        x_truncation_upper_str = (
                                  'x axis truncated by {0}\n'
                                  .format(x_truncation_upper)
                                  )
        x_truncation_lower_str = (
                                  'x axis truncated after {0}\n'
                                  .format(x_truncation_lower)
                                 )
    # Set string as empty when not being used.
    else:
        x_truncation_upper_str = ''
        x_truncation_lower_str = ''

    # Used to describe the format of plot
    if bin_n is None:
        bin_n_str = 'automatic'
    else:
        bin_n_str = bin_n

    # Strings within text box
    parameters = (
                  'Formatting:\n'
                  + x_truncation_lower_str
                  + x_truncation_upper_str
                  + 'bins = {0}'.format(bin_n_str)
                  )

    fig = formatting_text_box(fig, parameters, formatting_right)

    # Will not work on multiple subplots within a figure
    # gives an error instead.
    if ax is None:
        # Seaborn despine to remove boundaries around plot
        sns.despine(offset=2, trim=True, left=True, bottom=True)

    return fig


def boolean_bar(
                data,
                name,
                color_set=custom_bw,
                ax_size=(2, 5),
                annotate=True
                ):
    """
    A plotted bar chart for a True/False question.

    Can include a text annotation of the proportion
    of responses.

    Parameters
    ----------
    data : array_like
        List, pandas series, pandas dataframe column.
        Should be an array of booleans.
    name : string
        String describing the input data.
    color_set : list
        List of three colors that are compatible
        with matplotlib.
    ax_size : tuple
        tuple containing ax size. First value is
        width, second value is height.
    annotate : boolean
        True uses annotation.
        False turns it off.

    Returns
    -------
    fig : matplotlib axes
        Will plot an individual figure or ax
        can be reused within a figure containing
        more than one subplot.
    """

    common_set_up(ax_size)  # Apply basic plot style

    fig = sns.countplot(
                        data,
                        saturation=1,
                        color=color_set[2],
                        label=name
                        )

    # Trims off unnecessary parts of the figure
    sns.despine(offset=2, trim=True, left=True, bottom=True)

    # Set title and axes
    title_color = '#192231'  # Dark grey
    fig.set_title(
                  '{0}'.format(name),
                  fontsize=20,
                  color=title_color
                  )
    fig.set_ylabel('')
    fig.set_xlabel('')

    # Fraction annotation within bars.
    #
    # Does not work when one of the bars is close to zero.
    if annotate:
        total = float(len(data))

        for patch in fig.patches:  # patches is matplotlib term
            fraction = patch.get_height()/total
            fig.annotate(
                         '{:.2f}'.format(fraction),  # Value to be anootated
                         (
                          patch.get_x()+patch.get_width()/2.,      # X position
                          patch.get_height()-1300                  # y position
                         ),
                         ha='center',
                         label='Fraction',
                         color=color_set[0]
                        )

    return fig


def qq_plot(data, name, distribution="norm", ax_size=(7, 7)):
    """
    Creates a qq (quantile quantile) plot using one data
    value against an ideal distribution, like the normal
    distribution.

    Parameters
    ----------
    data : array_like
        List, pandas series, pandas dataframe column.
        Should be an array of booleans.
    name : string
        String describing the input data.
    distribution : string
        string of scipy distributions accepted by
        scipy.stats.probplot
    ax_size : tuple
        tuple containing ax size. First value is
        width, second value is height.
    """

    common_set_up(ax_size)

    fig = plt.figure(figsize=ax_size)
    ax = fig.add_subplot(111)  # Make one axes

    # Use scipy stats probplot and get out only values
    (x, y) = stats.probplot(data, dist=distribution, plot=None, fit=False)

    # Add a best fit line to the plot.
    #
    # Not using probplot version to be able to
    # customize the style of the line.
    slope, intercept, r, prob, sterrest = stats.linregress(x, y)
    ax.plot(
            x,
            (slope*x + intercept),
            '#9099A2',  # Choose color for line
            linestyle='--',  # Dashed line
            linewidth=1
            )

    ax.scatter(
               x,
               y,
               s=70,  # Scale of points on scatter plot
               facecolors='none',  # Transparent, no fill
               edgecolors='#192231',  # Dark grey
               linewidths=1.4
               )

    title_color = '#192231'  # Dark grey
    font_colour = '#9099A2'  # Light grey

    ax.set_title(
                 "Q-Q plot of {0}".format(name),
                 fontsize=20,
                 color=title_color
                 )
    ax.set_ylabel(
                  'Quantiles of {0}'.format(name),
                  color=font_colour
                  )
    ax.set_xlabel(
                  'Quantiles of {0} dist.'.format(distribution),
                  color=font_colour
                  )

    sns.despine(ax=ax, offset=2, trim=True, left=True, bottom=True)


def qq_plot_var(data_a, data_b, name_a, name_b, ax_size=(7, 7), fit_zero=True):
    """
    Creates a qq (quantile quantile) plot comparing two data
    values against each other.

    Parameters
    ----------
    data_a : array_like
        List, pandas series, pandas dataframe column.
        Should be an array of booleans.
        Corresponds to the x-axis.
    data_b : array_like
        List, pandas series, pandas dataframe column.
        Should be an array of booleans.
        Corresponds to the y-axis.
    name_a : string
        String describing the input data_a.
    name_b : string
        String describing the input data_b.
    ax_size : tuple
        tuple containing ax size. First value is
        width, second value is height.
    fit_zero : boolean
        True will fit expand the plot to include 0, 0.
        False will automatically fit the plot scales to the data.
    """

    common_set_up(ax_size)

    fig = plt.figure(figsize=ax_size)
    ax = fig.add_subplot(111)  # Make one plot within a figure

    # Manually calculate quantiles from 1 to 100.
    x = []
    for i in range(1, 100):
        v = np.percentile(data_a, i)
        x.append(v)
    y = []
    for i in range(1, 100):
        v = np.percentile(data_b, i)
        y.append(v)

    # Plot a base line of y = 1x + 0
    ax.plot(
            x,
            (1*x),
            '#9099A2',  # Color of line, light grey
            linestyle='--',  # Dashed line style
            linewidth=1
            )

    ax.scatter(
               x,
               y,
               s=40,  # Scale of scatter point
               facecolors='none',  # Transparent fill
               edgecolors='#192231',  # Darky grey
               linewidths=0.5
               )

    # To be able to see the figure back to 0, 0
    if fit_zero:
        axes = ax.axes
        axes.set_xlim(0,)
        axes.set_ylim(0,)

    title_color = '#192231'  # Dark grey
    font_colour = '#9099A2'  # Light grey

    ax.set_title(
                 "Q-Q plot of {0} vs {1}".format(name_a, name_b),
                 fontsize=20,
                 color=title_color
                 )
    ax.set_ylabel(
                  'Quantiles of {0}'.format(name_b),
                  color=font_colour
                  )
    ax.set_xlabel(
                  'Quantiles of {0}'.format(name_a),
                  color=font_colour
                  )

    sns.despine(ax=ax, offset=2, trim=True, left=True, bottom=True)

    return ax


def count_bar(data, name, color_set=custom, ax_size=(20, 6), funky=False, highlight=None, ax=None):
    """Make a univariate distribution
    of a variable.

    Returns an object to be plotted.
    """

    if funky:
        color_set = ToddTerje

    common_set_up(ax_size) # Apply basic plot style

    fig = sns.countplot(data, saturation=1, ax=ax,
                       color=color_set[2], label=name,
                      )

    sns.despine(offset=2, trim=True, left=True, bottom=True)

    # Set title and axes
    title_color = '#192231'
    font_colour = '#9099A2'
    if ax is None:
        fig.set_title('{0}'.format(name),
                    fontsize=20, color=title_color)
    fig.set_ylabel('Frequency',
                   color=font_colour)
    fig.set_xlabel('{0}'.format(name),
                   color=font_colour)
    
    if highlight:
        bars = fig.patches
        bars[highlight].set_color(color_set[1])
    
    return fig


def univariate_overdispersed(x, univariate_name, transform='log_10', color_set=custom, bin_n='all_values', ax_size=(12, 6), funky=False, rug=False, formatting_right=True, x_truncation_upper=None, x_truncation_lower=None,  ax=None):
    """Retrun plot using data transformation to correct
    for overdispersed data.
    """

    if bin_n == 'all_values':
        x_max = x.max()
        x_min = x.min()
        bin_n = int(x_max)-int(x_min)

    # The function applied to pandas objects are
    # from .transformation
    if transform == 'log_10':
        x = x.apply(log_10)
        univariate_name = univariate_name + ' in logarithms (log10)'
    elif transform == 'sqrt':
        x = x.apply(sq_rt)
        univariate_name = univariate_name + ' square root'

    fig = univariate(x, univariate_name, color_set=custom, bin_n=bin_n, ax_size=ax_size, funky=funky, rug=rug, formatting_right=formatting_right, x_truncation_upper=x_truncation_upper, x_truncation_lower=x_truncation_lower, ax=ax)

    return fig


def dist_transform_plot(x, univariate_name, fig_size=(18, 16), color_set=custom, bin_n='all_values', ax_size=(12, 6), funky=False, rug=True, formatting_right=True, x_truncation_upper=None, x_truncation_lower=None, ax=None):
    """Returns a plot including
    three individual plots alligned
    as three rows based on two data
    transforms.
    """

    common_set_up(fig_size)

    fig_plot, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_size), facecolor='white')
    fig_plot.suptitle("Distribution of {0}".format(univariate_name), fontsize=16)
    fig_plot.subplots_adjust(hspace=0.18, top=0.95)

    univariate_overdispersed(x, univariate_name, transform=None, color_set=color_set, bin_n=bin_n, ax_size=ax_size, funky=funky, rug=rug, formatting_right=formatting_right, x_truncation_upper=x_truncation_upper, x_truncation_lower=x_truncation_lower, ax=ax1)
    univariate_overdispersed(x, univariate_name, transform='sqrt', color_set=color_set, bin_n=bin_n, ax_size=ax_size, funky=funky, rug=rug, formatting_right=formatting_right, x_truncation_upper=x_truncation_upper, x_truncation_lower=x_truncation_lower, ax=ax2)
    univariate_overdispersed(x, univariate_name, transform='log_10', color_set=color_set, bin_n=bin_n, ax_size=ax_size, funky=funky, rug=rug, formatting_right=formatting_right, x_truncation_upper=x_truncation_upper, x_truncation_lower=x_truncation_lower, ax=ax3)

    sns.despine(offset=2, trim=True, left=True, bottom=True)

    return fig_plot
