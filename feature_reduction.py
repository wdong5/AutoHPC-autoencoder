import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.decomposition import PCA

def Entropy(labels, base=2):
    probs = pd.Series(labels).value_counts() / len(labels)
    en = stats.entropy(probs, base=base)
    return en

# make sure after shannon and PCA reduction, the size of all samples will be different
def Shannon_reduction(sparseX, entropy_ratio):
    entropy_ratio = int(sparseX.shape[0] * entropy_ratio)
    if entropy_ratio == 0:
        entropy_ratio = 1
    #sparseX = np.array(sparseX).todense()
    entropy_temp = []
    for row in range(sparseX.shape[0]):
        temp = Entropy(sparseX[row, :])
        entropy_temp.append(temp)
    entropy_temp = np.array(entropy_temp)
    idx = (-entropy_temp).argsort()[:entropy_ratio]
    after_entropy_sparseX = sparseX[idx]
    return after_entropy_sparseX

def PCA_reduction(sparseX, pca_ratio):
    pca = PCA()
    pca.fit(sparseX)
    pca_temp = []
    pca_ratio = int(sparseX.shape[0] * pca_ratio)
    if pca_ratio == 0:
        pca_ratio = 1
    features = pca.explained_variance_ratio_
    for row in range(sparseX.shape[0]):
        temp = features[row]
        pca_temp.append(temp)
    pca_temp = np.array(pca_temp)
    idx = (-pca_temp).argsort()[:pca_ratio]
    after_pca_sparseX = sparseX[idx]
    return after_pca_sparseX

def fig_vis(matrix_x):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(matrix_x, vmin=0, vmax=1, cmap='magma')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
