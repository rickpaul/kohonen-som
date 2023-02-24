
# %% ---------------------------------- Package Imports

import math
#
import numpy as np
import pandas as pd

# %% ---------------------------------- Package Imports

from SOM.Abstract_SOM import Abstract_SOM

# %% ---------------------------------- Constants

DISTANCE_FNS = {
    'manhattan': lambda x, y: np.sum(np.abs(x-y), axis=1),
    'euclidean': lambda x, y: np.sum(np.linalg.norm(x-y), axis=1),
    'sqeuclidean': lambda x, y: np.linalg.norm(x-y)**2,
    'chebyshev': lambda x, y: np.max(np.abs(x-y)),
    'gaussian': lambda x, y: np.exp(-np.linalg.norm(x-y)**2/2),
    'mexican_hat': lambda x, y: (1 - np.linalg.norm(x-y)**2)*np.exp(-np.linalg.norm(x-y)**2/2),
    'canberra': lambda x, y: np.sum(np.abs(x-y)/(np.abs(x) + np.abs(y))),
    'braycurtis': lambda x, y: np.sum(np.abs(x-y))/(np.sum(np.abs(x)) + np.sum(np.abs(y))),
    'cosine': lambda x, y: 1 - np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)),
    'correlation': lambda x, y: 1 - np.corrcoef(x, y)[0, 1],
    'hamming': lambda x, y: np.sum(x != y)/len(x),
    'jaccard': lambda x, y: np.sum(np.minimum(x, y))/np.sum(np.maximum(x, y)),
    'kulsinski': lambda x, y: np.sum(np.minimum(x, y))/np.sum(np.maximum(x, y)),
    'rogerstanimoto': lambda x, y: np.sum(np.minimum(x, y))/(np.sum(np.maximum(x, y)) + np.sum(np.minimum(x, y))),
    'russellrao': lambda x, y: np.sum(np.minimum(x, y))/np.sum(np.maximum(x, y)),
    'sokalmichener': lambda x, y: np.sum(np.minimum(x, y))/(np.sum(np.maximum(x, y)) + np.sum(np.minimum(x, y))),
    'sokalsneath': lambda x, y: 2*np.sum(np.minimum(x, y))/(np.sum(x) + np.sum(y)),
    'yule': lambda x, y: np.sum(np.minimum(x, y))/(np.sum(np.maximum(x, y)) + np.sum(np.minimum(x, y))),
    'dice': lambda x, y: 1 - 2*np.sum(np.minimum(x, y))/(np.sum(x) + np.sum(y)),
    'matching': lambda x, y: 1 - np.sum(np.minimum(x, y))/np.sum(np.maximum(x, y)),
    'jensenshannon': lambda x, y: np.sqrt(0.5*np.sum((np.sqrt(x) - np.sqrt(y))**2)),
    # 'minkowski': lambda x, y, p: np.sum(np.abs(x-y)**p)**(1/p),
    # 'mahalanobis': lambda x, y, V: np.sqrt(np.dot(np.dot((x-y).T, V), (x-y))),
}

# %% ---------------------------------- Class

class Base_SOM(Abstract_SOM):
    
    def __init__(self, distance_metric='manhattan') -> None:
        self.distance_fn = DISTANCE_FNS[distance_metric]
        
    @property
    def X_hat(self):
        try:
            return(self._descale_data(self.vecs, self._scaling_params))
        except:
            return(np.array([]))
    
    def _update_dataset(self, X):
        self._X, self._scaling_params = self._scale_data(X)
        self.training_size, self.num_dims = self._X.shape
    
    def _initialize_vecs(self):
        print('Initializing vectors (su)')
        self.vecs = np.random.uniform(low=0, high=1, size=(self.SOM_size**2, self.num_dims))
    
    def _update_architecture(self):
        num_nodes = 5*math.ceil(math.sqrt(self.training_size))
        self.SOM_size = math.ceil(math.sqrt(num_nodes))
        self._initialize_vecs()
        self.locs = [[i, j] for i in range(self.SOM_size) for j in range(self.SOM_size)]
        # Initialize iterations
        self.max_iter = self.SOM_size**2 * 500
        self.curr_iter = 0 # TODO: I don't think we need this -- just for descriptive purposes
    
    def _find_neighbors(self, bmu_idx):
        bmu_loc = self.locs[bmu_idx]
        dists = self.distance_fn(np.array(self.locs), bmu_loc)
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
