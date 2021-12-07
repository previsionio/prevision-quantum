""" Base Descriptor module """
import numpy as np


class BaseDescriptor:
    """Base Descriptor Computer.

    Base class for further implementations of descriptors.

    Attributes:
    """
    def __init__(self, params):
        """constructor """
        self.params = params
        self.descriptor_type = params.get('descriptor_type')
        self.num_q = params.get("num_q", 2)
        self.variables_range = params.get("variables_range", [0, 2 * np.pi])
        self.variables_sample_size = params.get("variables_sample_size", 5000)
        self.variables_seed = params.get("variables_seed", 0)
        self.variables_generator = lambda *_, **__: []

    def compute(self, circuit) -> float:
        """ computes the descriptor

        to be implemented depending on the descriptor considered
        :param circuit:
        :return: descriptor_value
        """
        raise NotImplementedError("Implement this method in daughter class.")