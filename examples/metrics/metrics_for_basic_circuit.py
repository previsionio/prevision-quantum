import prevision_quantum_nn as qnn
import pennylane.numpy as np

if __name__ == "__main__":

    # prepare data
    data_sample_size = 1
    X = 2 * np.pi * np.random.random((data_sample_size, 2))
    dataset = qnn.get_dataset_from_numpy(X, np.zeros(data_sample_size))

    # todo: function get_model_params with all possible arguments,
    #  so that fields are automatically suggested
    # customize model
    model_params = {
        "architecture": "qubit",
        "backend": "lightning.qubit",
        "encoding": "no_encoding",
        "layer_type": "template",
        "layer_name": "basic_circuit_6",
        "num_q": 4,
        "num_layers": 2,
        "double_mode": True,
    }

    descriptor_params = {
        # "variables_sample_size": 100,
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

    print(prefix)
    print(expr)
    print(ent)
