""" PennyLane ansatz module

contains the classes to handle ansatze
"""
import pennylane as qml
import pennylane.numpy as np


def double_gate(gate):
    def new_gate(weights, wires):
        gate(weights[..., 0], wires=wires)
        gate(weights[..., 1], wires=wires)

    return new_gate


# todo: put the parametrised_gates_list somewhere else
#  maybe automatize it?
parametrised_gates_list = [qml.RX, qml.RY, qml.RZ,
                           qml.CRX, qml.CRY, qml.CRZ,
                           qml.Rot]


def double_broadcast(gate, wires, pattern, parameters=None, *args):
    if parameters is None:
        qml.broadcast(gate, wires, pattern, *args)

    elif gate in parametrised_gates_list:
        qml.broadcast(double_gate(gate), wires, pattern,
                      [[x] for x in parameters], *args)
    else:
        qml.broadcast(gate, wires, pattern, [[x] for x in parameters], *args)


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

        self.check_double_mode()

        if self.double_mode:
            self.broadcast = double_broadcast
        else:
            self.broadcast = qml.broadcast

    def check_double_mode(self):
        if self.double_mode:
            if self.layer_type == "custom":
                if self.variables_shape[-1] != 2:
                    raise ValueError("custom mode and double mode are "
                                     "compatible only if "
                                     "variables_shape[-1] = 2")

            elif self.layer_type == "template":
                # todo: check double mode for template layers
                string_list = ["basic_circuit_"+str(i) for i in range(1,20)]
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

    def get_layers(self):
        if self.layer_name == "StronglyEntanglingLayers":
            self.variables_shape = qml.templates.layers. \
                StronglyEntanglingLayers.shape(self.num_layers, self.num_q)

            def layers(variables):
                qml.templates.layers.StronglyEntanglingLayers(
                    variables,
                    wires=range(self.num_q)
                )
        else:
            layer = self.get_layer()

            def layers(variables):
                for var in variables:
                    layer(var)
        return layers

    def get_layer(self):
        n = self.num_q

        if self.layer_name == "basic_circuit_1":
            self.variables_shape = (self.num_layers, self.num_q, 2)

            def layer(var):
                self.broadcast(qml.RX, self.wires, "single",
                               parameters=var[:, 0])
                self.broadcast(qml.RZ, self.wires, "single",
                               parameters=var[:, 1])

        elif self.layer_name == "basic_circuit_2":
            self.variables_shape = (self.num_layers, self.num_q, 2)

            def layer(var):
                self.broadcast(qml.RX, self.wires, "single",
                               parameters=var[:, 0])
                self.broadcast(qml.RZ, self.wires, "single",
                               parameters=var[:, 1])
                self.broadcast(qml.CNOT, self.wires[::-1], "chain")

        elif self.layer_name in ["basic_circuit_3", "basic_circuit_4"]:
            self.variables_shape = (self.num_layers, 3 * self.num_q - 1)

            if self.layer_name == "basic_circuit_3":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer(var):
                self.broadcast(qml.RX, self.wires, "single",
                               parameters=var[:n])
                self.broadcast(qml.RZ, self.wires, "single",
                               parameters=var[n:2 * n])
                self.broadcast(gate, self.wires[::-1], "chain",
                               parameters=var[2 * n:])

        elif self.layer_name in ["basic_circuit_5", "basic_circuit_6"]:
            self.variables_shape = (self.num_layers, 3 * n + n * n)

            if self.layer_name == "basic_circuit_5":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer(var):
                self.broadcast(qml.RX, self.wires, "single",
                               parameters=var[:n])
                self.broadcast(qml.RZ, self.wires, "single",
                               parameters=var[n:2 * n])
                ind = 2 * n
                for i in range(n - 1, -1, -1):
                    for j in range(n - 1, -1, -1):
                        if i != j:
                            gate(var[ind], wires=[self.wires[i], self.wires[j]])
                            ind += 1

                self.broadcast(qml.RX, self.wires, "single", var[ind:ind + n])
                self.broadcast(qml.RZ, self.wires, "single",
                               var[ind + n:ind + 2 * n])

        elif self.layer_name in ["basic_circuit_7", "basic_circuit_8"]:
            self.variables_shape = (self.num_layers, 5 * n - 1)

            if self.layer_name == "basic_circuit_7":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer(var):
                self.broadcast(qml.RX, self.wires, "single", var[:n])
                self.broadcast(qml.RZ, self.wires, "single", var[n:2 * n])
                ind = 2 * n
                self.broadcast(gate, self.wires[::-1], "double",
                               var[ind: ind + n // 2])
                ind += n // 2
                self.broadcast(qml.RX, self.wires, "single", var[ind:ind + n])
                self.broadcast(qml.RZ, self.wires, "single",
                               var[ind + n:ind + 2 * n])
                ind += 2 * n
                self.broadcast(gate, self.wires[::-1], "double_odd",
                               var[ind: ind + (n - 1) // 2])

        elif self.layer_name == "basic_circuit_9":
            self.variables_shape = (self.num_layers, n)

            def layer(var):
                self.broadcast(qml.Hadamard, self.wires, "single")
                self.broadcast(qml.CZ, self.wires[::-1], "chain")
                self.broadcast(qml.RX, self.wires, "single", var)

        elif self.layer_name == "basic_circuit_10":
            self.variables_shape = (self.num_layers, 2 * n)

            def layer(var):
                self.broadcast(qml.RY, self.wires, "single", var[:n])
                self.broadcast(qml.CZ, self.wires[::-1], "ring")
                self.broadcast(qml.RY, self.wires, "single", var[n:])

        elif self.layer_name in ["basic_circuit_11", "basic_circuit_12"]:
            self.variables_shape = (self.num_layers, 4 * n - 4)

            if self.layer_name == "basic_circuit_11":
                gate = qml.CNOT
            else:
                gate = qml.CZ

            def layer(var):
                ind = n
                self.broadcast(qml.RY, self.wires, "single", var[:ind])
                self.broadcast(qml.RZ, self.wires, "single", var[ind: ind + n])
                ind += n
                self.broadcast(reverse(gate), self.wires, "double")
                self.broadcast(qml.RY, self.wires[1:-1], "single",
                               var[ind:ind + n - 2])
                ind += n - 2
                self.broadcast(qml.RZ, self.wires[1:-1], "single",
                               var[ind:ind + n - 2])
                ind += n - 2
                self.broadcast(reverse(gate), self.wires, "double_odd")

        elif self.layer_name in ["basic_circuit_13", "basic_circuit_14"]:
            # Warning: the implementation might not be correct in the paper
            # TODO: write the original implementation from paper
            #                https://arxiv.org/pdf/1804.00633.pdf
            self.variables_shape = (self.num_layers, 4 * n)
            if self.layer_name == "basic_circuit_13":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer(var):
                n = len(self.wires)
                pattern1 = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]
                pattern2 = [[i % n, (i - 1) % n] for i in range(-1, n - 1)]

                ind = 0
                self.broadcast(qml.RY, self.wires, "single", var[ind:ind + n])
                ind += n
                self.broadcast(gate, self.wires, pattern1, var[ind: ind + n])
                ind += n
                self.broadcast(qml.RY, self.wires, "single", var[ind:ind + n])
                ind += n
                self.broadcast(gate, self.wires, pattern2, var[ind:ind + n])

        elif self.layer_name == "basic_circuit_15":
            self.variables_shape = (self.num_layers, 2 * n)

            def layer(var):
                pattern1 = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]
                pattern2 = [[i % n, (i - 1) % n] for i in range(-1, n - 1)]

                self.broadcast(qml.RY, self.wires, "single", var[:n])
                self.broadcast(qml.CNOT, self.wires, pattern1)
                self.broadcast(qml.RY, self.wires, "single", var[n:])
                self.broadcast(qml.CNOT, self.wires, pattern2)

        elif self.layer_name in ["basic_circuit_16", "basic_circuit_17"]:
            self.variables_shape = (self.num_layers, 3 * n - 1)
            if self.layer_name == "basic_circuit_16":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer(var):
                self.broadcast(qml.RX, self.wires, "single", var[:n])
                self.broadcast(qml.RZ, self.wires, "single", var[n:2 * n])
                ind = 2 * n
                self.broadcast(reverse(gate), self.wires, "double",
                               var[ind: ind + n // 2])
                ind += n // 2
                self.broadcast(reverse(gate), self.wires, "double_odd",
                               var[ind: ind + (n - 1) // 2])

        elif self.layer_name in ["basic_circuit_18", "basic_circuit_19"]:
            self.variables_shape = (self.num_layers, 3 * n)
            if self.layer_name == "basic_circuit_18":
                gate = qml.CRZ
            else:
                gate = qml.CRX

            def layer(var):
                pattern = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]

                self.broadcast(qml.RX, self.wires, "single", var[:n])
                self.broadcast(qml.RZ, self.wires, "single", var[n:2 * n])
                self.broadcast(gate, self.wires, pattern, var[2 * n:])

        else:
            raise ValueError(f"No ansatz corresponding to layer name: "
                             f"{self.layer_name}")

        if self.double_mode:
            self.variables_shape = [*self.variables_shape, 2]

        return layer


def reverse(gate):
    if gate.num_params == 0:
        def reversed_gate(wires):
            gate(wires=[wires[1], wires[0]])
    else:
        def reversed_gate(weights, wires):
            gate(weights, wires=[wires[1], wires[0]])
    return reversed_gate
