import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance
        source: Python Data Science Handbook"""
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


class GMMAnomalyScorer:
    def __init__(self, n_components=2):
        self.model = GaussianMixture(n_components=n_components)
        self._data = None
        self._labels = None

    def fit(self, data, labels):
        self._data = data
        self._labels = labels
        self.model = self.model.fit(self._data, self._labels)
        return self

    def score(self):
        if self._data is None:
            raise ValueError("Model not fit, please run fit() method first")
        return self.model.aic(self._data)

    def plot(self, label=True, ax=None):
        ax = ax or plt.gca()
        labels = self.model.fit(self._data).predict(self._data)
        if label:
            ax.scatter(self._data[:, 0], self._data[:, 1], c=labels, s=40, cmap='cividis', zorder=2, alpha=0.3)
        else:
            ax.scatter(self._data[:, 0], self._data[:, 1], s=40, zorder=2, alpha=0.3)
        ax.axis('equal')

        w_factor = 0.2 / self.model.weights_.max()
        for pos, covar, w in zip(self.model.means_, self.model.covariances_, self.model.weights_):
            draw_ellipse(pos, covar, ax=ax, alpha=w * w_factor)

        return ax

    def f1_score(self):
        if self._data is None:
            raise ValueError("Model not fit, please run fit() method first")

        labels_predicted = self.model.predict(self._data)

        return f1_score(self._labels, labels_predicted)
