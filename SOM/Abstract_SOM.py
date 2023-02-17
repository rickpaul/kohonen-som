
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
    
    # @abstractproperty
    # def X_hat(self):
    #     pass
    
    @property
    def X_hat(self):
        try:
            return(self._descale_data(self.vecs, self._scaling_params))
        except:
            return(np.array([]))
    
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

    def _save_fit_distances(self):
        sum_dists = [0 for _ in range(len(self.vecs))]
        num_dists = [0 for _ in range(len(self.vecs))]
        for idx in range(self._X.shape[0]):
            bmu, bmu_dist = self._find_bmu(self._X[idx])
            sum_dists[bmu] += bmu_dist
            num_dists[bmu] += 1   
        self.sum_dists = np.array(sum_dists)
        self.num_dists = np.array(num_dists)
            
    def fit(self, X):
        self._update_dataset(X)
        self._update_architecture()
        self._fit_SOM()
        self._save_fit_distances()
        
        
    
# %%
