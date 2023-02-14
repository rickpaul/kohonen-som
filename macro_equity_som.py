
"""
Sources: 
+   https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4
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
TODO:
+   X and y data
+   Weighting of the vectors (if X and y, based on kullback-leibler divergence)
+   How to deal with missing data?
+   Incorporate mexican hat update function
+   n-dimensional SOM
+   use for anomaly detection: https://www.analyticsvidhya.com/blog/2021/09/beginners-guide-to-anomaly-detection-using-self-organizing-maps/
+   Implement with PCA?
+   Compare with SOMPY?
+   Compare with GEMA (https://github.com/ufvceiec/GEMA)?
"""

# %% ---------------------------------- Package Imports

import math
import numpy as np
import pandas as pd
#
from xbbg import blp
#
from matplotlib import pyplot as plt
from matplotlib import patches as patches
#
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_blobs

#

# %% ---------------------------------- Classes


# %% ---------------------------------- Basic SOM



# %% ---------------------------------- Testing Basic SOM
if(0):
    # Generate
    X = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=1.5, random_state=101)[0]
    # Fit
    s = Basic_SOM()
    s.fit(X)
    # Plot
    plt.scatter(X[:, 0], X[:, 1], c="k", s=10, alpha=0.2)
    plt.scatter(s.X_hat[:, 0], s.X_hat[:, 1], c="r", s=10, alpha=0.8)


fig = plt.figure()

ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim((0, s.SOM_size+1))
ax.set_ylim((0, s.SOM_size+1))
ax.set_title(f'Self-Organising Map after {s.max_iter} iterations')

# plot
for x in range(1, net.shape[0] + 1):
    for y in range(1, net.shape[1] + 1):
        ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=net[x-1,y-1,:],
                     edgecolor='none'))
plt.show()