""" get descriptor module """

from prevision_quantum_nn.applications.metrics.expressibility_descriptor \
    import ExpressibilityDescriptor


def get_descriptor(params):
    """Get a descriptor according to parameters.

    Args:
        params (dictionnary):parameters of the descriptor

    Returns:
        descriptor: QuantumNeuralNetwork
            descriptor to be constructed with these parameters
    """
    descriptor_type = params.get("descriptor_type", "expressibility")

    if descriptor_type == "expressibility":
        descriptor = ExpressibilityDescriptor(params)
    else:
        raise ValueError("Invalid descriptor type. "
                         "Choices are expressibility, entangling capability "
                         "or concentratable entanglement")
    return descriptor


