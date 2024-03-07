import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel("ERROR")


def plot_distribution(X, y, title, fig_title):
    # Plot the data's distribution
    plt.figure(figsize=(10, 7))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(title)
    colors = [
        "lime",
        "darkorange",
        "deepskyblue",
        "orchid",
        "gold",
        "fuchsia",
        "salmon",
        "slategray",
    ]
    cmap = ListedColormap(colors[: len(np.unique(y))])
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=cmap(idx),
            marker="o",
            edgecolor="k",
            label=cl,
        )
    plt.legend(loc=0)
    plt.tight_layout()


def plot_decision_boundries(X, y, classifier, title, fig_title):
    """
    This function plots the decision boundary of a 2d dataset

    Parameter
    ---------
    X and y of the dataset as an array
    Classifier = the classifier e.g. a logistic regression model
    Titel = the titel of the plot.
    fig_title = the title the plot should be saved on the computer.

    """

    resolution = 0.001
    # setup marker generator and color map
    markers = ("H", "X", "^", "o", "s", "p", "*", "d")
    colors = [
        "lime",
        "darkorange",
        "deepskyblue",
        "orchid",
        "gold",
        "fuchsia",
        "salmon",
        "slategray",
    ]
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    fig = plt.figure(figsize=(12, 7))
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title("Logistic Regression - Decision Regions")

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=cmap(idx),
            marker=markers[idx],
            label=cl,
        )

    plt.legend(loc=0)
    plt.tight_layout()


def plot_confusion_matrix(cm, classes, title, fig_title):
    """
    This function prints and plots the confusion matrix.
    With an arbitrary number of classes.

    Parameter
    ---------
    cm: Confuion martrix/contincency table
    classes: array-like number of classes like [1,2,3]
    title: the desired plot title

    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=24, pad=20)

    ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted 0s", "Predicted 1s"))
    ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual 0s", "Actual 1s"))
    ax.set_ylim(1.5, -0.5)

    thresh = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)
    plt.tight_layout()
