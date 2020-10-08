""" Continuous variable module """

import tensorflow as tf
import pennylane as qml
import pennylane.numpy as np

from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane \
        import PennylaneNeuralNetwork


class CVNeuralNetwork(PennylaneNeuralNetwork):
    """Class CVNeuralNetwork.
    Implements a neural network on Continuous Variable architecture

    Attributes:
        cutoff_dim (int):cutoff dimension of the strawberryfields backend
        dev (qml.device):device to be used to train the model
    """
    def __init__(self, params):
        """Constructor.

        Args:
            params (dictionnary):parameters of the model
        """
        super().__init__(params)
        self.architecture_type = "continuous_variable"
        self.cutoff_dim = self.params.get("cutoff_dim", 10)
        self.encoding = self.params.get("encoding", "displacement")
        self.measure_type = self.params.get("measure_type", "x")
        self.backend = self.params.get("backend", "strawberryfields.fock")

        self.check_encoding()

    def build(self):
        """ builds the backend and the device """
        super().build()
        # build backend
        # When strawberryfields.tf available in stable version,
        # will be used because faster
        # if self.interface == "tf":
        #    self.backend = "strawberryfields.tf"
        # elif self.interface == "autograd":
        if self.interface == "tf":
            raise ValueError("Tensorflow interface for CV calculations "
                             "will be available in prevision_quantum_nn "
                             "when the backend strawberryfields.tf "
                             "will be in the stable version of "
                             "strawberryfields")

        self.backend = "strawberryfields.fock"

        # build device
        self.dev = qml.device(self.backend,
                              wires=self.num_q,
                              cutoff_dim=self.cutoff_dim)

        def neural_network(var, features=None):
            """Neural_network, decorated by a pennylane qnode.

            Args:
                var (list):list of weights of the model
                x (array or tf.Tensor):observations to be passed
                    through the neural network

            Returns:
                _neural_network(var,x): array
                    predictions of the model
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
            ValueError if invalid encoding for CV calculation
        """
        if self.encoding not in ["displacement", "squeezing"]:
            raise ValueError("Invalid encoding for CV. Valid encoding are: "
                             "displacement, "
                             "squeezing")

    def initialize_weights(self, file_name=None):
        """Initializes weights.

        Args:
            file_name (str):option, if None, the weights will be initialized
                randomly if not None, weights will be loaded from file
        """
        # if file_name is provided, load weights from file
        if file_name is not None:
            self.load_weights(file_name)
        # otherwise, initalize new weights
        else:
            if self.layer_type == "custom":
                var_init = 0.05 * np.random.randn(self.num_layers,
                                                  9 * self.num_q
                                                  + 2 * (self.num_q-1))
                if self.interface == "tf":
                    var_init = tf.Variable(var_init)
            elif self.layer_type == "template":
                var_init = []
                var_init.append(qml.init.cvqnn_layers_theta_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_phi_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_varphi_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_r_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_phi_r_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_theta_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_phi_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_varphi_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_a_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_phi_a_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))
                var_init.append(qml.init.cvqnn_layers_kappa_normal(
                    n_layers=self.num_layers, n_wires=self.num_q))

                if self.interface == "tf":
                    var_init = [tf.Variable(v) for v in var_init]
            else:
                raise ValueError(f"Unknown layer type: {self.layer_type}")

        self.var = var_init

    def encode_data(self, features):
        """Encodes data according to encoding method.

        Args:
            x (array):Array of features to be embedded
        """
        wires = list(range(self.num_q))

        if self.encoding == "displacement":
            qml.templates.embeddings.DisplacementEmbedding(features,
                                                           wires=wires)
        elif self.encoding == "squeezing":
            qml.templates.embeddings.SqueezingEmbedding(features,
                                                        wires=wires)

    def layers(self, variables):
        """Layers of the model.

        Depending on layer_type, the layers will either be custom or template

        Args:
            variables (list):weights of the model
        """
        if self.num_q == 1 or self.layer_type == "custom":
            for var in variables:
                index = 0

                # entangle qumodes
                for wire in range(self.num_q - 1):
                    qml.Beamsplitter(var[index],
                                     var[index + 1],
                                     wires=[wire + 1, wire])
                    index += 2

                # Displacement
                for wire in range(self.num_q):
                    qml.Displacement(var[index],
                                     0.,
                                     wires=wire)
                    index += 1

                # Cubic
                for wire in range(self.num_q):
                    qml.CubicPhase(var[index], wires=wire)
                    index += 1

                # Quadratic
                for wire in range(self.num_q):
                    qml.QuadraticPhase(var[index], wires=wire)
                    index += 1
        elif self.layer_type == "template":
            qml.templates.layers.CVNeuralNetLayers(*variables,
                                                   wires=range(self.num_q))

    def output_layer(self):
        """Output layer.

        Returns:
            list: quantum observables
        """
        expectations = None

        if self.type_problem == "classification" or \
                self.type_problem == "regression":
            expectations = qml.expval(qml.X(0))
        elif self.type_problem == "multiclassification" or \
                self.type_problem == "reinforcement_learning":
            if self.measure_type == "x":
                expectations = [qml.expval(qml.X(i))
                                for i in range(self.num_categories)]
            elif self.measure_type == "photon_number":
                expectations = [qml.expval(qml.NumberOperator(wires=i))
                                for i in range(self.num_categories)]
        return expectations
