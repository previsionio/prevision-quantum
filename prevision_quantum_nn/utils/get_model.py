""" get model module """

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
