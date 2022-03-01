from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np

import prevision_quantum_nn as qnn
import matplotlib.pyplot as plt
import time
from sklearn import datasets
import sys


def twospirals(turns, noise=0.7, random_state=0):
    """Returns the two spirals dataset."""
    np.random.seed(random_state)
    rng_sp = np.random
    n_points = int(200 * turns)
    n = np.sqrt(rng_sp.rand(n_points, 1)) * turns * (2 * np.pi)
    d1x = -np.cos(n) * n + (rng_sp.rand(n_points, 1) - 0.5) * noise
    d1y = np.sin(n) * n + (rng_sp.rand(n_points, 1) - 0.5) * noise
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack(
        (-1 * np.ones(n_points).astype(int), np.ones(n_points).astype(int)))
    return x, y


def main_function(variables_init_type, prefix, double_mode):
    # prepare data
    num_q = 3
    num_layers = 2
    circuit = "basic_circuit_5"
    dataset_name = "breast_cancer"

    if dataset_name == "spirals":
        nb_turns = float(sys.argv[1])
        prefix += f"_nb_turns_{nb_turns}"

    prefix = f"results/circuit{circuit.split('_')[-1]}_dataset_{dataset_name}" \
             f"_{prefix}_num_q_{num_q}"
    random_state = 3+1
    nb_turns = 0

    if dataset_name == "spirals":
        nb_turns = float(sys.argv[1])
        X, y = twospirals(nb_turns)
    elif dataset_name == "breast_cancer":
        X, y = datasets.load_breast_cancer(return_X_y=True)
        pca = PCA(n_components=num_q)
        X = pca.fit_transform(X=X)
        y = 2 * y - 1
    elif dataset_name == "moon":
        num_samples = 500
        X, y = datasets.make_moons(n_samples=num_samples,
                                   noise=0.05, random_state=0)
        # shift label from {0, 1} to {-1, 1}
        y = y * 2 - np.ones(len(y))
    else:
        raise ValueError("incorrect dataset_name")

    x_temp, x_test, y_temp, y_test = train_test_split(X, y, test_size=0.25,
                                                      random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp,
                                                      test_size=0.25,
                                                      random_state=42)

    dataset = qnn.get_dataset_from_numpy(x_train, y_train,
                                         val_features=x_val, val_labels=y_val)

    preprocessing_params = {
        # "polynomial_degree": 2
    }

    model_params = {
        "architecture": "qubit",
        "backend": "lightning.qubit",
        "num_q": num_q,
        "num_layers": num_layers,
        "layer_name": circuit,
        "variables_init_type": variables_init_type,
        "double_mode": double_mode,
        "variables_random_state": random_state,
        #"early_stopper_epsilon": 1e-4,
        "max_iterations": 300,
        "early_stopper_patience": 50,
    }

    # build application
    application = qnn.get_application(
        "classification",
        prefix=prefix,
        preprocessing_params=preprocessing_params,
        model_params=model_params)

    # solve application
    application.solve(dataset)

    def get_auc(file_name):
        file = open(file_name, "r")
        a = []
        for line in file.read().split('\n'):
            if line.find("auc") >= 0:
                auc = line[line.find("auc") + 5:]
                a.append(auc)
        y_vec = []
        for y in a:
            try:
                y_vec.append(float(y))
            except:
                pass
        y_vec = np.array(y_vec)
        return y_vec[-1]

    y_pred = application.predict(x_test)
    y_test = 0.5 + 0.5 * y_test

    auc = get_auc(prefix + ".listing")

    acc = 0
    for l_true, l_pred in zip(y_test, y_pred):
        if abs(l_true - l_pred) < 1e-5:
            acc = acc + 1
    accuracy = acc / len(y_test)

    if nb_turns > 0:
        str_details = f", nb_turns = {nb_turns}"
    else:
        str_details = f", dataset {dataset_name}"

    with open("results.txt", "a") as file:
        file.write(time.strftime("%m_%d_%H:%M:%S", time.gmtime()) + "\n")
        file.write(f"id_block_init{str_details}\n")
        file.write(f"auc = {auc},  accuracy = {accuracy}\n")
        file.write("\n")


if __name__ == "__main__":

    nb_turns = float(sys.argv[1])
    X, y = twospirals(nb_turns)

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1])
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1])
    plt.show()

"""
ARCHIVE STRING
custom circuit used for first results 'custom'

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

"""
