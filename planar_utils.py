import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix
    Y = np.zeros((m, 1), dtype='uint8')  # labels
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():
    np.random.seed(3)
    noisy_circles, noisy_moons = sklearn.datasets.make_circles(n_samples=300, factor=.5, noise=.05), \
                                  sklearn.datasets.make_moons(n_samples=300, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=300, random_state=5)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(n_samples=300, n_features=2, n_classes=2, random_state=5)
    no_structure = np.random.rand(300, 2), np.random.rand(300, 1)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
