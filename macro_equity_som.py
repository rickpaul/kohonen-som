# %% ---------------------------------- Notes
"""
Sources: Analyzed and Discarded
+   https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4
...   https://github.com/abhinavralhan/kohonen-maps/blob/master/som-random.ipynb
Sources: 
+   https://www.superdatascience.com/blogs/self-organizing-maps-soms-how-do-self-organizing-maps-learn-part-1/
+   https://www.superdatascience.com/blogs/self-organizing-maps-soms-how-do-self-organizing-maps-learn-part-2/
+   https://medium.com/geekculture/dynamic-self-organizing-maps-gsom-60a785fbe39d
+   https://towardsdatascience.com/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
    ... neurons should be 5 * sqrt(n) where n is the number of training samples
    ... max_iter should be 500 * n * m where n * m is the number of neurons
+   https://www.kaggle.com/code/nilsschlueter/self-organizing-maps-for-anomaly-detection
    ... rgb exercise
    ... anomaly detection
    ... plotting
+   https://www.esann.org/sites/default/files/proceedings/legacy/es2015-77.pdf
    ... SOM for regression
    ... Importance of output function
+   https://d1wqtxts1xzle7.cloudfront.net/47400953/j.biortech.2008.06.04220160721-9001-r7lre3-libre.pdf?1469103546=&response-content-disposition=inline%3B+filename%3DApplication_of_the_self_organizing_map_a.pdf&Expires=1675351159&Signature=JDavgPzuUi4FGzm8dQB4lg8t2QCImSZlXp~Z0mhCqFgkeKaX6OLMqZlkGx6AZz5e5G-1O74t9N1O2O~57~97XXstGJFXmh543DZza3TeQPujOAwfpPv9ssK1NaWBUr9bkw1KzLhsBqfHfWT1mqtAbfkZSAxpAOnlMKXpb~OI4jxJSpWlFI1j3DyfNmdCmG8pt5GSLETVupttpyxZg0FGqH1twXVlkv6ELe7Gu7VpHP9~L3tL3SKzhzPXUiZahpOJODi36KXI3pSFUGT4VjTSLGQ1XCfbeGqwek3gdXGFdfcF1Us~JAZOUCeDEygbKjWCwu0W-1DPspZk1jZVMv51NA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
    ... SOM for imputation?
+   https://www.sciencedirect.com/science/article/abs/pii/S016974391500060X
    ... SOM for imputation? Abstract only.
+   https://www.researchgate.net/profile/Barbara-Hammer/publication/242506480_Perspectives_of_Neural-Symbolic_Integration/links/00463531ec4827629d000000/Perspectives-of-Neural-Symbolic-Integration.pdf#page=139
    ... SOM for Time Series
+   https://perso.uclouvain.be/michel.verleysen/papers/neuralnetworks02jl.pdf
TODO:
+   How to determine if a SOM is good?
+   X and y data (i.e. independent and dependent)
+   toroidal topology
+   PCA Initialization
+   hexagonal topology
+   gaussian neighborhood function
+   Weighting of the vectors (if X and y, based on kullback-leibler divergence)
+   How to deal with missing data?
+   Incorporate mexican hat update function
+   n-dimensional SOM
+   use for anomaly detection: https://www.analyticsvidhya.com/blog/2021/09/beginners-guide-to-anomaly-detection-using-self-organizing-maps/
+   Implement with PCA?
+   Compare with SOMPY?
+   Compare with GEMA (https://github.com/ufvceiec/GEMA)?
+   Visualization - net
+   Visualization - component plane
+   Visualization - U-Matrix
+   Visualization - Sammon's mapping
+   Visualization - https://weber.itn.liu.se/~aidvi/courses/06/dm/Seminars2011/SOM(3).pdf

"""

# %% ---------------------------------- Package Imports

import numpy as np
import pandas as pd
#
from matplotlib import pyplot as plt
from matplotlib import patches as patches
#
from sklearn.datasets import make_blobs
#

# %% ---------------------------------- Local Imports
from sys import modules
from importlib import reload

for m in ['SOM.Basic_SOM', 'SOM.Abstract_SOM', 'SOM.Base_SOM']: 
    if(m in modules):
        modules[m] = reload(modules[m])

from SOM.Basic_SOM import Basic_SOM as bsom

# %% ---------------------------------- Testing Basic SOM (blobs)
if(1):
    s = bsom()
    # Generate
    X = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.5)[0]
    s._update_dataset(X)
    s._update_architecture()
    plt.scatter(X[:, 0], X[:, 1], c="k", s=10, alpha=.6)
    plt.scatter(s.X_hat[:, 0], s.X_hat[:, 1], c="b", s=10, alpha=0.6, label='unfit vectors')
    # Fit
    s.fit(X)
    # Plot
    plt.scatter(s.X_hat[:, 0], s.X_hat[:, 1], c="r", s=10, alpha=0.8, label='fit vectors')


# %% ---------------------------------- Testing Basic SOM (colors)

def plot_som(s):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((0, s.SOM_size+1))
    ax.set_ylim((0, s.SOM_size+1))
    ax.set_title(f'Self-Organising Map after {s.curr_iter} iterations')
    # plot
    for x in range(s.SOM_size):
        for y in range(s.SOM_size):
            ax.add_patch(patches.Rectangle((x+0.5, y+0.5), 1, 1,
                        facecolor=s.vecs[x*s.SOM_size+y,:],
                        edgecolor='none'))
    plt.show()

if(0):
    s = bsom()
    # Set random seed
    np.random.seed(42)
    X = np.random.randint(0, 255, (1000, 3))
    s._update_dataset(X)
    s._update_architecture()
    plot_som(s)
    s.fit(X)
    plot_som(s)