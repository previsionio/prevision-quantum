""" Qubit module"""
from copy import deepcopy

import tensorflow as tf
import pennylane as qml
import pennylane.numpy as np
import math

from prevision_quantum_nn.models.pennylane_backend.pennylane_ansatz import \
    AnsatzBuilder
from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane \
    import PennylaneNeuralNetwork


class PennylaneQubitNeuralNetwork(PennylaneNeuralNetwork):
    """Class PennylaneQubitNeuralNetwork.

    Implements a neural network on a discrete qubit architecture

    Attributes:
        dev (qml.device):device to be used to train the model
    """

    def __init__(self, params):
        """Constructor.

        Args:
            params (dictionary):parameters of the model
        """
        super().__init__(params)
        self.architecture_type = "qubit"
        self.encoding = self.params.get("encoding", "angle")
        self.backend = self.params.get("backend", "default.qubit.tf")
        self.layer_name = self.params.get("layer_name",
                                          "StronglyEntanglingLayers")
        self.neural_network = self.params.get("neural_network", None)
        self.ansatz = self.params.get("ansatz", None)
        self.variables_shape = self.params.get("variables_shape", None)
        self.variables_init_type = self.params.get("variables_init_type",
                                                   "default")
        self.double_mode = self.params.get("double_mode", False)
        self.variables_random_state = self.params.get("variables_random_state",
                                                      0)
        self.ansatz_builder = None

        self.check_encoding()

    @staticmethod
    def get_params_attributes():
        """Attributes that can be set as a parameter"""
        cls = PennylaneQubitNeuralNetwork
        return super(cls, cls).get_params_attributes() + \
               ["encoding",
                "backend",
                "layer_name",
                "neural_network",
                "ansatz",
                "variables_shape",
                "variables_init_type",
                "double_mode",
                "variables_random_state"]

    def build_model(self):
        """ builds the device and the qnode"""
        self.ansatz_builder = AnsatzBuilder(self.num_q,
                                            self.num_layers,
                                            self.layer_name,
                                            self.layer_type,
                                            double_mode=self.double_mode)
        self.ansatz_builder.build(self.ansatz, self.variables_shape)

        def neural_network(var, features=None):
            """Neural_network, decorated by a pennylane qnode.

            Args:
                var (list):list of weights of the model
                features (array or tf.Tensor):observations to be passed
                    through the neural network

            Returns:
                list:predictions of the model
            """

            # encode data
            self.encode_data(features)

            # layers
            self.ansatz_builder.ansatz(var)

            return self.output_layer()

        self.neural_network = neural_network

    def build(self, weights_file=None):
        """ builds the backend and the device """
        super().build(weights_file=weights_file)
        # build backend
        self.check_backend()

        # build device
        self.dev = qml.device(self.backend, wires=self.num_q)

        self.neural_network = qml.QNode(self.neural_network,
                                        self.dev,
                                        interface=self.interface)

    def check_backend(self):
        """Checks backend consistency with interface """
        autograd_backends = ["default.qubit.autograd",
                             "default.qubit",
                             "lightning.qubit",
                             "damavand.qubit"]
        if self.interface == "autograd" and \
                self.backend not in autograd_backends:
            self.backend = "default.qubit.autograd"
        elif self.interface == "tf":
            self.backend = "default.qubit.tf"

    def check_encoding(self):
        """Checks encoding consistency.

        Raises:
            ValueError if invalid encoding for qubit calculation
        """
        valid_encoding = ["angle", "amplitude", "mottonen", "no_encoding"]
        if self.encoding not in valid_encoding:
            raise ValueError("Invalid encoding for qubit neural network. "
                             f"Valid encoding are: {', '.join(valid_encoding)}")

    def initialize_weights(self, weights_file=None):
        """Initializes weights.

        Args:
            weights_file (str):option, if None, the weights will be initialized
                randomly if not None, weights will be loaded from file
        """
        if weights_file is not None:
            self.load_weights(weights_file)
        else:
            low, high = self.ansatz_builder.variables_range
            var_shape = self.ansatz_builder.variables_shape

            # ideally, use np.random.RandomState(self.variables_random_state)
            np.random.seed(self.variables_random_state)

            # todo: should these conditions be moved to ansatz_builder?
            if self.variables_init_type == "default":
                var_init = np.random.uniform(low=low, high=high, size=var_shape)
            elif self.variables_init_type == "zeros":
                var_init = 0.01 * np.random.randn(*var_shape)
            else:   # identity block strategy
                var = 2 * np.pi * np.random.random(var_shape[:-1])
                var_init = np.array([[var[i], -var[i]]
                                     for i in np.ndindex(var.shape)])

                var_init = deepcopy(var_init)
                var_init.resize(var_shape)

                assert tuple(var_init.shape) == tuple(var_shape)

            if self.interface == "tf":
                var_init = tf.Variable(var_init)
            self.var = var_init

    def encode_data(self, features):
        """Encodes data according to encoding method."""

        wires = range(self.num_q)

        # amplitude encoding mode
        if self.encoding == "amplitude":
            qml.templates.embeddings.AmplitudeEmbedding(features,
                                                        wires=wires,
                                                        normalize=True)
        # angle encoding mode
        elif self.encoding == "angle":
            qml.templates.embeddings.AngleEmbedding(features,
                                                    wires=wires)
        elif self.encoding == "mottonen":
            norm = np.sum(np.abs(features) ** 2)
            features = features / math.sqrt(norm)
            qml.templates.state_preparations.MottonenStatePreparation(
                features, wires=wires)
        elif self.encoding == "no_encoding":
            pass

    def output_layer(self):
        """Output layer.

        Returns:
            list: quantum observables
        """
        expectations = None
        if self.type_problem == "classification" or \
                self.type_problem == "regression":
            expectations = qml.expval(qml.PauliZ(0))
        elif self.type_problem == "multiclassification" or \
                self.type_problem == "reinforcement_learning":
            expectations = [qml.expval(qml.PauliZ(i))
                            for i in range(self.num_categories)]
        elif self.type_problem == "descriptor_computation":
            # TODO: not a list of quantum observables... Doesn't work on QLACS
            #  for instance
            expectations = qml.state()
        return expectations
