""" get model module """

import os
import json

from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane_cv \
        import CVNeuralNetwork
from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane_qubit \
        import PennylaneQubitNeuralNetwork


def get_model(params):
    """Get a model according to parameters.

    Args:
        params (dictionnary):parameters of the model

    Returns:
        model: QuantumNeuralNetwork
            model to be constructed with these parameters
    """
    architecture = params.get("architecture", "qubit")

    if architecture == "qubit":
        model = PennylaneQubitNeuralNetwork(params)
    elif architecture == "cv":
        model = CVNeuralNetwork(params)
    else:
        raise ValueError("Invalid architecture. "
                         "Choices are qubit or cv")
    return model


def get_model_from_parameters_file(parameters_file_name):
    """Get a model from a parameters file.

    Args:
        parameters_file_name(str): name of the file containing the
            parameters of the model to be loaded

    Returns:
        model: QuantumNeuralNetwork
            model to be constructed with these parameters
    """
    if not os.path.isfile(parameters_file_name):
        raise ValueError("QNNFactory.load_parameters:"
                         "provided weights file does not exist "
                         f"{parameters_file_name}")
    if os.path.exists(parameters_file_name):
        with open(parameters_file_name, "rb") as parameters_file:
            params = json.load(parameters_file)
        print("utils.get_model_from_parameters_file:"
              "params loaded from file.")
    else:
        raise ValueError("params file cannot be found")
    model = get_model(params)
    return model
