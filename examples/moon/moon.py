from sklearn import datasets
from sklearn.model_selection import train_test_split

import prevision_quantum_nn as qnn

import numpy as np

if __name__ == "__main__":
    # prepare data
    num_samples = 500
    X, y = datasets.make_moons(n_samples=num_samples,
                               noise=0.05, random_state=0)
    # shift label from {0, 1} to {-1, 1}
    y = y * 2 - np.ones(len(y))
    x_train, x_val, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=40)

    # build dataset
    dataset = qnn.get_dataset_from_numpy(x_train,
                                         y_train,
                                         val_features=x_val,
                                         val_labels=y_test)

    # customize preprocessing
    preprocessing_params = {
        "polynomial_degree": 2
    }

    # customize model
    model_params = {
        "architecture": "qubit",
        "num_q": 5,
        "encoding": "angle",
        "use_early_stopper": True,
        "early_stopper_patience": 20,
        "max_iterations": 10000,
        "interface": "autograd",
        "layer_type": "template",
        "snapshot_frequency": 5,
        "verbose": True,
        "prefix": "moon",
        "num_layers": 4,
        "optimizer_name": "Adam",
        "learning_rate": 0.05,
    }

    # customize postprocessing
    postprocessing_params = {
        "phase_space_plotter": {
            "dim": 2,
            "min_max_array": [[min(X[:, 0]), max(X[:, 0])],
                              [min(X[:, 1]), max(X[:, 1])]],
            "prefix": "moon"
        }
    }

    # build application
    application = qnn.get_application(
        "classification",
        prefix="moon",
        preprocessing_params=preprocessing_params,
        model_params=model_params,
        postprocessing_params=postprocessing_params)

    # solve application
    application.solve(dataset)
