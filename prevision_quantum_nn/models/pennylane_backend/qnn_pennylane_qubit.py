""" Qubit module"""

import tensorflow as tf
import pennylane as qml
import pennylane.numpy as np

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
            params (dictionnary):parameters of the model
        """
        super().__init__(params)
        self.architecture_type = "discrete"
        self.encoding = self.params.get("encoding", "angle")
        self.backend = self.params.get("backend", "default.qubit.tf")

        self.check_encoding()

    def build(self):
        """ builds the backend and the device """
        super().build()
        # build backend
        if self.interface == "autograd":
            self.backend = "default.qubit.autograd"
        elif self.interface == "tf":
            self.backend = "default.qubit.tf"
        # build device
        self.dev = qml.device(self.backend, wires=self.num_q)

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
            self.layers(var)

            return self.output_layer()

        self.neural_network = qml.QNode(neural_network,
                                        self.dev,
                                        interface=self.interface)

    def check_encoding(self):
        """Checks encoding consistency.

        Raises:
            ValueError if invalid encoding for qubit calculation
        """
        if self.encoding not in ["angle", "amplitude", "mottonen"]:
            raise ValueError("Invalid encoding for qubit neural network. "
                             "Valid encoding are: "
                             "angle, "
                             "amplitude, "
                             "mottonen")

    def initialize_weights(self, file_name=None):
        """Initializes weights.

        Args:
            file_name (str):option, if None, the weights will be initialized
                randomly if not None, weights will be loaded from file
        """
        if file_name is not None:
            var_init = self.load_weights(file_name)
        else:
            if self.num_q == 1:
                var_init = 0.05 * np.random.randn(self.num_layers,
                                                  3 * self.num_q)
            else:
                var_init = qml.init.strong_ent_layers_uniform(
                    n_layers=self.num_layers,
                    n_wires=self.num_q)

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
            qml.templates.state_preparations.MottonenStatePreparation(
                features, wires=wires)

    def layers(self, variables):
        """Layers of the model.

        Depending on layer_type, the layers will either be
        custom or template

        Args:
            variables (list):weights of the model
        """
        # custom layer
        if self.num_q == 1:
            for var in variables:
                for k in range(self.num_q):
                    qml.Rot(var[0], var[1], var[2], wires=k)
        # template layer
        else:
            qml.templates.layers.StronglyEntanglingLayers(
                variables,
                wires=range(self.num_q)
            )

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
        return expectations
