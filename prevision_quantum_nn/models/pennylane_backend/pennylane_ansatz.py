""" PennyLane ansatz module

contains the classes to handle ansatze
"""
import pennylane as qml
import pennylane.numpy as np


class AnsatzBuilder:
    """AnsatzBuilder.

    Abstraction to handle layers building

    Attributes:
        num_q (int):number of qubits
        num_q (int):number of qubits
        num_q (int):number of qubits
    """

    def __init__(self, num_q, num_layers, layer_name, layer_type,
                 variables_range=None, double_mode=False):
        """ constructor """
        self.num_q = num_q
        self.num_layers = num_layers
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.variables_shape = None
        self.ansatz = lambda *_, **__: None
        self.wires = range(num_q)
        if variables_range is None:
            variables_range = [0, 2 * np.pi]
        self.variables_range = variables_range
        self.double_mode = double_mode

        self.n_parameters = 0
        self.broadcast_list = []

        self.check_double_mode()

    def check_double_mode(self):
        if self.double_mode:
            if self.layer_type == "custom":
                if self.variables_shape[-1] != 2:
                    raise ValueError("custom mode and double mode are "
                                     "compatible only if "
                                     "variables_shape[-1] = 2")

            elif self.layer_type == "template":
                # todo: check double mode for template layers
                string_list = ["basic_circuit_" + str(i) for i in range(1, 20)]
                if self.layer_name in string_list:
                    pass

    def check_shape(self, variables):
        if np.prod(variables.shape) != np.prod(self.variables_shape):
            raise ValueError("variables shape is incorrect. "
                             f"Expected {self.variables_shape}, "
                             f"got {variables.shape}")

    def build(self, ansatz=None, variables_shape=None):

        if self.layer_type == "custom":
            if ansatz is None:
                raise ValueError("custom layer_type needs an ansatz")
            if variables_shape is None:
                raise ValueError("custom layer_type needs variables_shape")
            self.variables_shape = variables_shape
            self.ansatz = ansatz

        elif self.layer_type == "template":
            layers = self.get_layers()

            def ansatz(variables):
                self.check_shape(variables)
                variables = variables.reshape(self.variables_shape)
                layers(variables)

            self.ansatz = ansatz

        else:
            raise ValueError("Invalid layer_type for ansatz building. "
                             f"Valid layer_types are: custom, template")

    def broadcast(self, gate, wires, pattern, n_parameters=0, parameters=None):
        broadcast_params = {'gate': gate,
                            'wires': wires,
                            'pattern': pattern,
                            'n_parameters': n_parameters,
                            'parameters': parameters}
        if pattern == 'single' and n_parameters not in [0, len(wires)]:
            raise ValueError('Incorrect number of parameters')
        if pattern == 'double' and n_parameters not in [0, len(wires)//2]:
            raise ValueError('Incorrect number of parameters')
        if pattern == 'double_odd' and \
                n_parameters not in [0, (len(wires)-1)//2]:
            raise ValueError('Incorrect number of parameters')
        if pattern == 'chain' and n_parameters not in [0, len(wires) - 1]:
            raise ValueError('Incorrect number of parameters')
        if pattern == 'ring' and n_parameters not in [0, len(wires)]:
            raise ValueError('Incorrect number of parameters')

        if n_parameters > 0 and parameters is not None:
            raise ValueError("n_parameters and parameters cannot be defined"
                             "simultaneously.")
        self.broadcast_list.append(broadcast_params)
        self.n_parameters += n_parameters

    def unstack_layer(self):
        if self.variables_shape:
            print("Warning, variables_shape overwritten")
        self.variables_shape = (self.num_layers, self.n_parameters,
                                1 + self.double_mode)

        def layers(variables):
            for var in variables:  # each layer
                self.develop_broadcast_list(var[..., 0])

                if self.double_mode:
                    self.develop_broadcast_list(var[..., 1],
                                                double_mode=True)

        return layers

    def develop_broadcast_list(self, var, double_mode=False):
        if double_mode:
            var = var[::-1]
        ind = 0
        for broadcast_params in self.broadcast_list:
            gate, wires, pattern, n_var, parameters = \
                self.get_broadcast_params(**broadcast_params)
            if n_var > 0:
                weights = var[ind:ind + n_var]
            else:
                weights = parameters

            if double_mode:
                if pattern in ['single', 'double', 'double_odd']:
                    if n_var > 0:
                        weights = weights[::-1]
                    qml.broadcast(gate, wires, pattern, weights)
                elif pattern in ['chain']:
                    qml.broadcast(reverse(gate), wires[::-1], pattern, weights)
                elif pattern in ['ring', 'pyramid', 'all_to_all']:
                    print(f"Warning: double_mode not implemented "
                          f"for pattern '{pattern}'")
                elif type(pattern) is list:
                    qml.broadcast(gate, wires, pattern[::-1], weights)
                else:
                    print(f"Warning: unrecognized pattern '{pattern}'")
            else:
                qml.broadcast(gate, wires, pattern, weights)
            ind += n_var

    def get_broadcast_params(self, gate, wires, pattern,
                             n_parameters, parameters):
        return gate, wires, pattern, n_parameters, parameters

    def get_layers(self):
        if self.layer_name == "StronglyEntanglingLayers":
            if self.double_mode:
                print("Warning: StronglyEntanglingLayers is unfit "
                      "for double_mode")
            self.variables_shape = qml.templates.layers. \
                StronglyEntanglingLayers.shape(self.num_layers, self.num_q)

            def layers(variables):
                qml.templates.layers.StronglyEntanglingLayers(
                    variables,
                    wires=range(self.num_q)
                )
        else:
            self.stack_layer()
            layers = self.unstack_layer()

        return layers

    def stack_layer(self):
        n = self.num_q

        if self.layer_name == "basic_circuit_1":
            def layer():
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)

        elif self.layer_name == "basic_circuit_2":
            def layer():
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)
                self.broadcast(qml.CNOT, self.wires[::-1], "chain")

        elif self.layer_name in ["basic_circuit_3", "basic_circuit_4"]:
            if self.layer_name == "basic_circuit_3":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer():
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)
                self.broadcast(gate, self.wires[::-1], "chain", n-1)

        elif self.layer_name in ["basic_circuit_5", "basic_circuit_6"]:
            if self.layer_name == "basic_circuit_5":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            pattern = [[i, j] for i in range(n - 1, -1, -1)
                       for j in range(n - 1, -1, -1)
                       if i != j]

            def layer():
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)
                self.broadcast(gate, self.wires, pattern, len(pattern))
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)

        elif self.layer_name in ["basic_circuit_7", "basic_circuit_8"]:
            if self.layer_name == "basic_circuit_7":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer():
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)
                self.broadcast(gate, self.wires[::-1], "double", n // 2)
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)
                self.broadcast(gate, self.wires[::-1], "double_odd", (n - 1)//2)

        elif self.layer_name == "basic_circuit_9":
            def layer():
                self.broadcast(qml.Hadamard, self.wires, "single")
                self.broadcast(qml.CZ, self.wires[::-1], "chain")
                self.broadcast(qml.RX, self.wires, "single", n)

        elif self.layer_name == "basic_circuit_10":

            def layer():
                qml.broadcast(qml.RY, self.wires, "single", n)
                qml.broadcast(qml.CZ, self.wires[::-1], "ring")
                qml.broadcast(qml.RY, self.wires, "single", n)

        elif self.layer_name in ["basic_circuit_11", "basic_circuit_12"]:
            if self.layer_name == "basic_circuit_11":
                gate = qml.CNOT
            else:
                gate = qml.CZ

            def layer():
                qml.broadcast(qml.RY, self.wires, "single", n)
                qml.broadcast(qml.RZ, self.wires, "single", n)
                qml.broadcast(reverse(gate), self.wires, "double")
                qml.broadcast(qml.RY, self.wires[1:-1], "single", n - 2)
                qml.broadcast(qml.RZ, self.wires[1:-1], "single", n - 2)
                qml.broadcast(reverse(gate), self.wires, "double_odd")

        elif self.layer_name in ["basic_circuit_13", "basic_circuit_14"]:
            # Warning: the implementation might not be correct in the paper
            # TODO: write the original implementation from paper
            #                https://arxiv.org/pdf/1804.00633.pdf
            if self.layer_name == "basic_circuit_13":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            pattern1 = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]
            pattern2 = [[i % n, (i - 1) % n] for i in range(-1, n - 1)]

            def layer():
                self.broadcast(qml.RY, self.wires, "single", n)
                self.broadcast(gate, self.wires, pattern1, n)
                self.broadcast(qml.RY, self.wires, "single", n)
                self.broadcast(gate, self.wires, pattern2, n)

        elif self.layer_name == "basic_circuit_15":

            pattern1 = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]
            pattern2 = [[i % n, (i - 1) % n] for i in range(-1, n - 1)]

            def layer():
                self.broadcast(qml.RY, self.wires, "single", n)
                self.broadcast(qml.CNOT, self.wires, pattern1)
                self.broadcast(qml.RY, self.wires, "single", n)
                self.broadcast(qml.CNOT, self.wires, pattern2)

        elif self.layer_name in ["basic_circuit_16", "basic_circuit_17"]:
            if self.layer_name == "basic_circuit_16":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer():
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)
                self.broadcast(reverse(gate), self.wires, "double", n // 2)
                self.broadcast(reverse(gate), self.wires, "double_odd",
                               (n - 1)//2)

        elif self.layer_name in ["basic_circuit_18", "basic_circuit_19"]:
            if self.layer_name == "basic_circuit_18":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            pattern = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]

            def layer():
                self.broadcast(qml.RX, self.wires, "single", n)
                self.broadcast(qml.RZ, self.wires, "single", n)
                self.broadcast(gate, self.wires, pattern, n)

        else:
            raise ValueError(f"No ansatz corresponding to layer name: "
                             f"{self.layer_name}")

        layer()


def reverse(gate):
    if gate.num_params == 0:
        def reversed_gate(wires):
            gate(wires=[wires[1], wires[0]])
    else:
        def reversed_gate(weights, wires):
            gate(weights, wires=[wires[1], wires[0]])
    return reversed_gate
