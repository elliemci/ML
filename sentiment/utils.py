import numpy as np
from sklearn.utils import check_random_state


def make_classification(
    mu, sigmas, num_cluster, points, dimensions=2, random_state=0, multi=False
):
    """
    Function to generate  a synthetic dataset for classification. It takes  the following parameters:
    mean (mu), standard deviation (sigma), number of clusters (num_cluster), number of data points (points),
    and a flag (multi) to indicate whether the data should be generated from a single or multiple distributions and
    random_state.

    If multi is True, it generates multivariate normal data for each class with the provided mu and sigmas,
    and number of data points. The generated data for each  class is  concatenated togetehere to form the
    final dataset and corresponding  labels.

    If multi is False, it generates univariate normal data for each class with the provided mu and sigmas.

    Notes: For random samples from N(\mu, \sigma^2), use: sigma * np.random.randn(...) + mu

    """

    n = num_cluster
    # to avoid random state from the computer's time
    generator = check_random_state(random_state)

    if multi:
        # number of zero labels
        N_0 = points[0]
        # number of Nnl labels
        Nnl = points[1]
        y = np.ones(Nnl, dtype=int)
        X = []
        Y = []

        # Zeros with first distribution
        x_zeros = generator.multivariate_normal(
            mu[0], np.array([sigma[0], np.flipud(sigma[0])]), size=N_0
        )
        y_zeros = np.zeros(N_0, dtype=int)

        # other labels and the other distributions
        for i in range(1, n):
            x = generator.multivariate_normal(
                mu[i], np.array([sigma[i], np.flipud(sigma[i])]), size=Nnl
            )
            X.append(x)
            Y.append(y * i)

            X_ = np.concatenate(X)
            y = np.concatenate(Y)

        # concat all together
        X = np.vstack([x_zeros, X_])
        y = np.concatenate([y_zeros, y])

    else:
        X = []

        for c in range(n):
            for d in range(dimensions):
                # randn returns a sample (or samples) from the “standard normal” distribution
                x = generator.randn(1, points[c])[0]
                # modify distribution as noted above
                x = mu[c][d] + x * sigmas[c][d]
                X.append(x)

        X = np.asarray(X)
        zeros = np.zeros(points[0])
        ones = np.ones(points[1])
        X_0 = np.stack((X[0], X[1]), axis=-1)
        X_1 = np.stack((X[2], X[3]), axis=-1)
        X = np.vstack([X_0, X_1])

        y = np.concatenate([zeros, ones])

    return X, y
