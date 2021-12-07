import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import prevision_quantum_nn as qnn


def get_gaussians_dataset(num_samples=50, holdout=20):
    """
    num_samples, default 50
    holdout in percentage, default 20%
    """
    samples = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]],
                                            num_samples)
    samples2 = np.random.multivariate_normal([-5, -5], [[1, 0], [0, 1]],
                                             num_samples)
    samples3 = np.random.multivariate_normal([-5, 5], [[1, 0], [0, 1]],
                                             num_samples)
    samples4 = np.random.multivariate_normal([5, -5], [[1, 0], [0, 1]],
                                             num_samples)
    y_sample = np.zeros((num_samples, 1))
    y_sample2 = np.zeros((num_samples, 1))
    y_sample3 = np.ones((num_samples, 1))
    y_sample4 = np.ones((num_samples, 1))

    x = np.vstack((samples, samples2, samples3, samples4))
    y = np.vstack((y_sample, y_sample2, y_sample3, y_sample4)).flatten()
    samplesA = np.vstack((samples, samples2))
    samplesB = np.vstack((samples3, samples4))
    x, y = shuffle(x, y)
    num_samples = len(x)
    return x, y


if __name__ == "__main__":
    # get dataset
    X, y = get_gaussians_dataset()

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42)
    dataset = qnn.get_dataset_from_numpy(x_train, y_train,
                                         val_features=x_val, val_labels=y_val)

    # get customized application
    preprocessing_params = {
        "polynomial_degree": 1
    }

    postprocessing_params = {
        "phase_space_plotter": {
            "dim": 2,
            "min_max_array": [[min(X[:, 0]), max(X[:, 0])],
                              [min(X[:, 1]), max(X[:, 1])]]
        }
    }
    application = qnn.get_application(
        "classification",
        prefix="gaussians",
        preprocessing_params=preprocessing_params,
        postprocessing_params=postprocessing_params,
    )
    application.solve(dataset)
