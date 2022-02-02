import prevision_quantum_nn as qnn
import pennylane.numpy as np
import matplotlib.pylab as plt


def get_metrics(backend, i):

    # customize model
    model_params = {
        "architecture": "qubit",
        "encoding": "no_encoding",
        "backend": backend,
        "layer_type": "template",
        "layer_name": "basic_circuit_6",
        "num_q": 1,
        "num_layers": 2,
        "double_mode": True,
    }

    descriptor_params = {
        "variables_sample_size": i,
    }

    prefix = f"results/ct{model_params['layer_name'].split('_')[-1]}-" \
             f"nq{model_params['num_q']}-nl{model_params['num_layers']}"

    # build application
    application = qnn.get_application("descriptor_computation",
                                      prefix=prefix,
                                      model_params=model_params,
                                      descriptor_params=descriptor_params)

    expr = application.compute(dataset, "expressibility")
    ent = application.compute(dataset, "entangling_capability")
    return expr, ent


if __name__ == "__main__":

    # prepare data
    data_sample_size = 1
    X = 2 * np.pi * np.random.random((data_sample_size, 2))
    dataset = qnn.get_dataset_from_numpy(X, np.zeros(data_sample_size))

    metrics_pennylane = []
    metrics = []

    for i in range(1, 500):
        metrics_pennylane.append(get_metrics("default.qubit.autograd", i)[0])
        metrics.append(get_metrics("damavand.qubit", i)[0])

    print(metrics_pennylane)
    print(metrics)
    plt.plot(metrics_pennylane, color="blueviolet", label="pennylane")
    plt.plot(metrics, color="lime", label="damavand")
    plt.legend()
    plt.savefig("results.png")
