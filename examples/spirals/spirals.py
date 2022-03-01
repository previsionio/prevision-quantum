from sklearn.model_selection import train_test_split
import numpy as np

import prevision_quantum_nn as qnn
import matplotlib.pyplot as plt


def twospirals(turns, noise=0.7, random_state=None):
    """Returns the two spirals dataset."""
    if random_state is None:
        rng_sp = np.random
    else:
        rng_sp = np.random.RandomState(random_state)
    n_points = int(200 * turns)
    n = np.sqrt(rng_sp.rand(n_points, 1)) * turns * (2 * np.pi)
    d1x = -np.cos(n) * n + (rng_sp.rand(n_points, 1)-0.5) * noise
    d1y = np.sin(n) * n + (rng_sp.rand(n_points, 1)-0.5) * noise
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack(
        (-1 * np.ones(n_points).astype(int), np.ones(n_points).astype(int)))
    return x, y


if __name__ == "__main__":
    # prepare data
    nb_turns = 1.0
    X, y = twospirals(nb_turns, random_state=2)

    figsize = (6, 6)
    dpi = 300
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(X[y == 1, 0], X[y == 1,1])
    plt.scatter(X[y == -1,0], X[y == -1,1])
    plt.title(f'Spirals dataset with $turns={nb_turns}$')
    plt.show()
else:

    x_train, x_val, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

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
        "max_iterations": 10000,
        "interface": "tf",
        "layer_type": "template",
        "snapshot_frequency": 5,
        "verbose": True,
        "prefix": "spirals",
        "num_layers": 4,
        "optimizer_name": "Adam",
        "learning_rate": 0.05,
        "use_early_stopper": True,
        "early_stopper_patience": 5,
        "val_verbose_period": 1,
    }

    # customize postprocessing
    postprocessing_params = {
        "phase_space_plotter": {
            "dim": 2,
            "min_max_array": [[min(X[:, 0]), max(X[:, 0])],
                              [min(X[:, 1]), max(X[:, 1])]]
        }
    }

    model_params["prefix"] = model_params["prefix"] + "_" + \
                             model_params["architecture"] + "_" + \
                             str(model_params["num_q"]) + "_" + \
                             str(preprocessing_params[
                                     "polynomial_degree"]) + "_" + \
                             model_params["encoding"] + "_" + \
                             str(model_params["num_layers"]) + "_" + \
                             str(nb_turns)

    postprocessing_params["phase_space_plotter"]["prefix"] = model_params[
        "prefix"]

    # build application
    application = qnn.get_application(
        "classification",
        prefix=model_params["prefix"],
        preprocessing_params=preprocessing_params,
        model_params=model_params,
        postprocessing_params=postprocessing_params)

    application.solve(dataset)
