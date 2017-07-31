#!/usr/bin/python
"""Data transformations to be applied
before creating figures.
"""

import numpy as np
import pandas as pd


def log_10(x):
    """Return data transformed by log10
    
    To be used in a pandas apply function.
    """

    value = np.log10(x)

    return value


def sq_rt(x):
    """Return data transformed by log10
    
    To be used in a pandas apply function.
    """

    value = np.sqrt(x)

    return value
