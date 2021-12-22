from sklearn import datasets
from sklearn.model_selection import train_test_split

import prevision_quantum_nn as qnn
import pennylane as qml
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
    num_q = 5
    num_layers = 2

    n = num_q

    variables_shape = (num_layers, 3 * n - 1)

    def ansatz(variables):
        wires = range(num_q)

        for var in variables:
            qml.broadcast(qml.RX, wires, "single", var[:n])
            qml.broadcast(qml.RZ, wires, "single", var[n:2 * n])
            ind = 2 * n
            qml.broadcast(qml.CRZ, wires, "double",
                          var[ind: ind + n // 2])
            ind += n // 2
            qml.broadcast(qml.CNOT, wires, "double_odd")


    model_params = {
        "architecture": "qubit",
        "num_q": num_q,
        "num_layers": num_layers,
        "layer_type": "custom",
        "ansatz": ansatz,
        "variables_shape": variables_shape,
    }

    # build application
    application = qnn.get_application(
        "classification",
        prefix="cust_atz",
        model_params=model_params)

    # solve application
    application.solve(dataset)
