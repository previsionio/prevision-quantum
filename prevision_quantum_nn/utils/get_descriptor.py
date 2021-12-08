""" get descriptor module """

from prevision_quantum_nn.applications.metrics.expressibility_descriptor \
    import ExpressibilityDescriptor
from prevision_quantum_nn.applications.metrics.entanglement_descriptor \
    import EntanglementDescriptor


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
    elif descriptor_type in EntanglementDescriptor.descriptor_types:
        descriptor = EntanglementDescriptor(params)
    else:
        raise ValueError("Invalid descriptor type. "
                         "Choices are expressibility, entangling_capability "
                         "or concentratable_entanglement")
    return descriptor


