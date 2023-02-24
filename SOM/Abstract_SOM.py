
# %% ---------------------------------- Package Imports

from abc import ABC, abstractmethod, abstractproperty

# %% ---------------------------------- Local Imports



# %% ---------------------------------- Class

class Abstract_SOM(ABC):

    @abstractproperty
    def learning_rate(self):
        pass
    
    @abstractproperty
    def n_radius(self):
        pass
    
    @abstractproperty
    def X_hat(self):
        pass

    @abstractmethod
    def _find_neighbors(self):
        pass
    
    @abstractmethod
    def _find_bmu(self):
        pass

    @abstractmethod
    def _update_dataset(self):
        pass
    
    @abstractmethod
    def _update_architecture(self):
        pass
    
    @abstractmethod
    def _fit_SOM(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def _save_fit_distances(self):
        pass
    
    @abstractmethod
    def _scale_data(self, X):
        pass
    
    @abstractmethod
    def _descale_data(self, X):
        pass
