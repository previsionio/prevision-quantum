""" get model module """

from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane_cv \
    import CVNeuralNetwork
from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane_qubit \
    import PennylaneQubitNeuralNetwork


def get_model(params):
    """Get a model according to parameters.

    Args:
        params (dictionary):parameters of the model

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


def get_model_params(
        running_mode="simulation",
        architecture="qubit",
        num_q=2,
        num_categories=None,
        num_actions=None,
        max_iterations=50000,
        num_layers=2,
        snapshot_frequency=0,
        type_problem=None,
        batch_size=1,
        use_early_stopper=True,
        early_stopper_patience=20,
        early_stopper_epsilon=1E-4,
        prefix="qnn",

        iteration=0,
        learning_rate=0.01,
        val_verbose_period=5,
        optimizer_name="Adam",
        interface="autograd",
        layer_type="template",
        optimizer=None,
        var=None,
        dev=None,
        neural_network=lambda *_, **__: None,

        architecture_type="qubit",
        encoding="angle",
        backend="default.qubit",
        layer_name="StronglyEntanglingLayers",
        ansatz=None,
        variables_shape=None,
        variables_init_type="default",
        double_mode=False,
        variables_random_state=0,
        **kwargs):

    params = {'running_mode': running_mode,
              'architecture': architecture,
              'num_q': num_q,
              'num_categories': num_categories,
              'num_actions': num_actions,
              'max_iterations': max_iterations,
              'num_layers': num_layers,
              'snapshot_frequency': snapshot_frequency,
              'type_problem': type_problem,
              'batch_size': batch_size,
              'val_verbose_period': val_verbose_period,
              'use_early_stopper': use_early_stopper,
              'early_stopper_patience': early_stopper_patience,
              'early_stopper_epsilon': early_stopper_epsilon,
              'prefix': prefix,
              'iteration': iteration,
              'learning_rate': learning_rate,
              'optimizer_name': optimizer_name,
              'interface': interface,
              'layer_type': layer_type,
              'encoding': encoding,
              'optimizer': optimizer,
              'var': var,
              'dev': dev,
              'neural_network': neural_network,
              'architecture_type': architecture_type,
              'backend': backend,
              'layer_name': layer_name,
              'ansatz': ansatz,
              'variables_shape': variables_shape,
              'variables_init_type': variables_init_type,
              'double_mode': double_mode,
              'variables_random_state': variables_random_state}

    params.update(kwargs)

    return params
