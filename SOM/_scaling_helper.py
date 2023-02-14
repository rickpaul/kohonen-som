
# %% ---------------------------------- Package Imports

import numpy as np

# %% ---------------------------------- Helper Functions

def scale_data_min_max_columnar(X):
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0)
    X = (X - min_)/(max_ - min_)
    scaling_dict = {"min_": min_, "max_": max_}
    return(X, scaling_dict)

def scale_data_min_max_global(X):
    min_ = np.min(X)
    max_ = np.max(X)
    X = (X - min_)/(max_ - min_)
    scaling_dict = {"min_": min_, "max_": max_}
    return(X, scaling_dict)

def de_scale_data_min_max(X, scaling_dict):
    max_ = scaling_dict["max_"]
    min_ = scaling_dict["min_"]
    X = X * (max_ - min_) + min_
    return(X)
