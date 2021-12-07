import numpy as np
import matplotlib.pylab as plt

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    # prepare data
    train_features = np.linspace(-1. * np.pi, 1. * np.pi, 50)
    train_labels = np.asarray((1 + np.sin(train_features)) / 2)
    train_features = train_features / np.pi
    val_features = np.linspace(-1. * np.pi, 1. * np.pi, 50)
    val_labels = np.asarray((1 + np.sin(val_features)) / 2)
    val_features = val_features / np.pi

    train_features = train_features.reshape((len(train_features), 1))
    val_features = val_features.reshape((len(val_features), 1))

    plt.plot(train_features, train_labels)
    plt.savefig("sinusoid.png")

    dataset = qnn.get_dataset_from_numpy(train_features,
                                         train_labels,
                                         val_features=val_features,
                                         val_labels=val_labels)

    # customize postprocessing
    model_params = {
        "architecture": "cv",
        "num_q": 2,
        "encoding": "displacement",
        "snapshot_frequency": 5,
        "prefix": "sinusoid",
        "early_stopper_patience": 50,
        "use_early_stopper": True,
        "num_layers": 5,
        "interface": "autograd",
        "layer_type": "template",
    }

    # customize postprocessing
    postprocessing_params = {
        "phase_space_plotter": {
            "dim": 1,
            "min_max_array": [[
                min(np.vstack([train_features, val_features])[:, 0]),
                max(np.vstack([train_features, val_features])[:, 0])
            ]]
        }
    }

    # get application
    application = qnn.get_application(
        "regression",
        prefix="sinusoid",
        model_params=model_params,
        postprocessing_params=postprocessing_params)

    application.solve(dataset)
