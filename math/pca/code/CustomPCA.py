# +
import numpy as np


class CustomPCA:
    """
    This is a custom implementation of the principal component analysis (PCA).

    """

    def __init__(self, n_components=None, whiten=False):
        # keep whitenig as False, as this is the same way it's done in sklearn, to get same results.
        self.n_components = n_components
        self.whiten = bool(whiten)

    def center(self, A):
        """
        Function to center and whitening the data. Whitening (or sphering)
        this is an important preprocessing step prior to performing
        principal component analysis.

        Parameter
        ---------
        A = Features-Matrix

        """
        # center the data
        self.mean_ = A.mean(axis=0)
        A = A - self.mean_

        # whiten if it is choosen
        if self.whiten:
            self.std = A.std(axis=0)
            A = A / self.std
        return A

    def fit(self, A):
        n, m = A.shape
        # center the data
        A = self.center(A)

        # Apply Eigendecomposition of the covariance matrix
        # TODO:
        # Implement here your solution for computing the covariance matrix
        # and the eigenvalues and eigenvectors.

        # Compute the covariance matrix
        self.covariance_matrix = np.cov(A, rowvar=False)

        # Compite the eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)

        # Apply the dimensionality reduction if we received an input value.
        if self.n_components is not None:
            self.eigenvectors = self.eigenvectors[:, 0 : self.n_components]
            self.eigenvalues = self.eigenvalues[0 : self.n_components]

        # sort values to be sure, they are in the correct order
        # our Eigenvectors should be in descending order
        sort = np.flip(np.argsort(self.eigenvalues))
        self.eigenvectors = self.eigenvectors[:, sort]
        self.eigenvalues = self.eigenvalues[sort]
        print("PCA(n_components={})".format(self.n_components))

        return self.eigenvalues, self.eigenvectors

    def transform(self, A):
        """
        Function to transform (compress) the data.

        Parameter
        ---------
        A = Features-Matrix

        """
        # center the data
        A = self.center(A)

        return np.einsum("mk,kn", A, self.eigenvectors)

    @property
    def variance_explained(self):
        """
        Function to compute the explained variance for given Eigenvalues,
        while doing PCA.

        Parameter
        ---------
        Eigenvalues

        """
        return self.eigenvalues / np.sum(self.eigenvalues)
