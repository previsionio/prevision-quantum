from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import prevision_quantum_nn as qnn

if __name__ == "__main__":

    # prepare data
    centers = [[np.pi * 0.6, np.pi * 0.6], [np.pi * 1.4, np.pi * 1.4]]
    cluster_std = np.pi * 0.3

    X, y = datasets.make_blobs(n_samples=500, random_state=0, centers=centers,
                               cluster_std=cluster_std)

    # shift label from {0, 1} to {-1, 1}
    y = y * 2 - np.ones(len(y))

    x_temp, x_test, y_temp, y_test = train_test_split(X, y, test_size=0.25,
                                                      random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp,
                                                      test_size=0.25,
                                                      random_state=42)

    dataset = qnn.get_dataset_from_numpy(x_train, y_train,
                                         val_features=x_val, val_labels=y_val)

    prefix = "results/acc_expr_ent_6"
    # customize model
    model_params = {
        "layer_name": "basic_circuit_6",
        "num_layers": 3,
        "early_stopper_epsilon": 0.3,
        "max_iterations": 101,
        "prefix": prefix,
    }

    # customize postprocessing
    postprocessing_params = {
        "phase_space_plotter": {
            "dim": 2,
            "min_max_array": [[0, 2*np.pi],
                              [0, 2*np.pi]],
            "prefix": prefix,
        }
    }

    classification_application = qnn.get_application(
        "classification",
        prefix=prefix,
        model_params=model_params,
        postprocessing_params=postprocessing_params
    )
    classification_application.solve(dataset)

    y_pred = classification_application.predict(x_test)
    y_test = 0.5 + 0.5 * y_test

    acc = 0
    for l_true, l_pred in zip(y_test, y_pred):
        if abs(l_true - l_pred) < 1e-5:
            acc = acc + 1
    accuracy = acc / len(y_test)
    print("accuracy", accuracy)

    # build application
    descriptor_application = qnn.get_application(
        "descriptor_computation",
        prefix="desc",
        model_params=model_params
    )

    expr = descriptor_application.compute(dataset, "expressibility")
    ent = descriptor_application.compute(dataset, "entangling_capability")

    print("expr", expr)
    print("ent", ent)
