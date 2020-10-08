import numpy as np
import matplotlib.pylab as plt

import prevision_quantum_nn as qnn


if __name__=="__main__":

    # prepare data
    train_features = np.linspace(0, np.pi, 50)
    train_labels = np.asarray(np.sin(train_features))
    val_features = np.linspace(0, np.pi, 50)
    val_labels = np.asarray(np.sin(val_features))

    train_features = train_features.reshape((len(train_features), 1))
    val_features = val_features.reshape((len(val_features), 1))

    plt.plot(train_features, train_labels)
    plt.savefig("sinusoid.png")

    dataset = qnn.get_dataset_from_numpy(train_features,
                                         train_labels,
                                         val_features=val_features,
                                         val_labels=val_labels)

    # get application

    # customize postprocessing
    postprocessing_params = {
        "phase_space_plotter": {
            "dim": 1,
            "min_max_array": [[
                min(np.vstack([train_features, val_features])[:,0]),
                max(np.vstack([train_features, val_features])[:,0])
            ]]
        }
    }
    application = qnn.get_application("regression",
                                      prefix="sinusoid",
                                      postprocessing_params=postprocessing_params)

    application.solve(dataset)
