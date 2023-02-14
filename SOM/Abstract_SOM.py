
# %% ---------------------------------- Package Imports

import math
#
import numpy as np
import pandas as pd
#
from abc import ABC, abstractmethod, abstractproperty

# %% ---------------------------------- Class

class abstract_SOM(ABC):
    
    @abstractproperty
    def learning_rate(self):
        pass
    
    @abstractproperty
    def n_radius(self):
        pass

    @abstractmethod
    def _fit_SOM(self):
        pass
    
    @abstractmethod
    def _scale_data(self, X):
        pass
    
    @abstractmethod
    def _descale_data(self, X):
        pass
    
    def _update_dataset(self, X):
        self._X, self._scaling_params = self._scale_data(X)
        self.training_size, self.num_dims = self._X.shape
    
    def _update_architecture(self):
        num_nodes = 5*math.ceil(math.sqrt(self.training_size))
        self.SOM_size = math.ceil(math.sqrt(num_nodes))
        self.vecs = np.random.uniform(low=0, high=1, size=(self.SOM_size**2, self.num_dims))
        self.locs = [[i, j] for i in range(self.SOM_size) for j in range(self.SOM_size)]
        # Initialize iterations
        self.max_iter = self.SOM_size**2 * 500
        self.curr_iter = 0 # TODO: I don't think we need this
    
    def _find_neighbors(self, bmu_idx):
        bmu_loc = self.locs[bmu_idx]
        if(self.manhattan_distance):
            dists = np.sum(np.abs(np.array(self.locs) - bmu_loc), axis=1)
        else:
            dists = np.linalg.norm(np.array(self.locs) - bmu_loc, axis=1) # Euclidean distance
        return(dists <= self.n_radius)
    
    def _find_bmu(self, x):
        dists = np.linalg.norm(self.vecs - x, axis=1)
        idx_ = np.argmin(dists)
        return(idx_, dists[idx_])
            
    def fit(self, X):
        self._update_dataset(X)
        self._update_architecture()
        self._fit_SOM()
        self.X_hat = self._descale_data(self.vecs, self._scaling_params)