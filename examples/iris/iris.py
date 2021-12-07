import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    # retrieve data and build dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    X = df.to_numpy()
    y = iris.target

    train_features, val_features, train_labels, val_labels = train_test_split(
        X, y, test_size=0.5, random_state=42)

    dataset = qnn.get_dataset_from_numpy(train_features,
                                         train_labels,
                                         val_features=val_features,
                                         val_labels=val_labels)

    # customize application
    model_params = {
        "num_q": 4,
        "max_iterations": 100000,
        "verbose": True,
        "prefix": "iris",
        "num_layers": 4,
        "interface": "autograd",
        "learning_rate": 0.05,
        "encoding": "angle",
        "early_stopper_patience": 50,
        "snapshot_frequency": 10,
    }

    preprocessing_params = {
        "polynomial_degree": 1
    }

    postprocessing_params = {
        "phase_space_plotter": {
            "dim": 4,
            "min_max_array": [[min(X[:, 0]), max(X[:, 0])],
                              [min(X[:, 1]), max(X[:, 1])]],
            "prefix": "iris"
        }
    }

    # build application
    application = qnn.get_application(
        "multiclassification",
        prefix="iris",
        preprocessing_params=preprocessing_params,
        model_params=model_params,
        postprocessing_params=postprocessing_params)

    application.solve(dataset)
