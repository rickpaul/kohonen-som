
# %% ---------------------------------- Package Imports

import numpy as np

# %% ---------------------------------- Local Imports

from Abstract_SOM import abstract_SOM
from _scaling_helper import (
    scale_data_min_max_columnar, 
    de_scale_data_min_max
)

# %% ---------------------------------- Constants

DATA_DIMENSION = 3
SPREAD_FACTOR = 0.01 # Value between 0 and 1
GROWTH_THRESHOLD = -DATA_DIMENSION*np.log(SPREAD_FACTOR)

# %% ---------------------------------- Class

class Basic_SOM(abstract_SOM):
    """
    Reference:
    https://towardsdatascience.com/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
    """
    
    def __init__(self,
                 learning_rate=0.5,
                 neighborhood_radius=4,
                 manhattan_distance=True,
                 ):
        self.init_learning_rate = learning_rate
        self.init_n_radius = neighborhood_radius
        self.manhattan_distance = manhattan_distance

    @property
    def learning_rate(self):
        return(self.init_learning_rate * np.exp(-self.curr_iter/self.max_iter))
    
    @property
    def n_radius(self):
        return(self.init_n_radius * np.exp(-self.curr_iter/self.max_iter))

    def _scale_data(self, X):
        return scale_data_min_max_columnar(X)
    
    def _descale_data(self, X, scaling_dict):
        return de_scale_data_min_max(X, scaling_dict)

    def _fit_SOM(self):
        for iter in range(self.max_iter):
            if(iter % 1000 == 0):
                print(f"Iteration: {iter}")
            idx = np.random.randint(self.training_size)
            bmu, bmu_dist = self._find_bmu(self._X[idx])
            neighbors = np.where(self._find_neighbors(bmu))[0]
            self.vecs[neighbors] += self.learning_rate * (self._X[idx] - self.vecs[neighbors]) ###### This is the key line in question
            self.curr_iter = iter # TODO: I don't think we need this