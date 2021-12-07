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

    def compute(self, circuit) -> float:
        """ computes the descriptor

        to be implemented depending on the descriptor considered
        :param circuit:
        :return: descriptor_value
        """
        raise NotImplementedError("Implement this method in daughter class.")