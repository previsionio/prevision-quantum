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

    def __init__(self, num_q, num_layers, layer_name, variables_range=None):
        """ constructor """
        self.num_q = num_q
        self.num_layers = num_layers
        self.layer_name = layer_name
        self.variables_shape = None
        self.ansatz = lambda *_, **__: None
        self.wires = range(num_q)
        if variables_range is None:
            variables_range = [0, 2 * np.pi]
        self.variables_range = variables_range

    def check_shape(self, variables):
        if np.prod(variables.shape) != np.prod(self.variables_shape):
            raise ValueError("variables shape is incorrect. "
                             f"Expected {self.variables_shape}, "
                             f"got {variables.shape}")

    def build(self):
        layer = None
        layers = None
        n = self.num_q
        if self.layer_name == "basic_circuit_1":
            self.variables_shape = (self.num_layers, self.num_q, 2)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:, 0])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[:, 1])

        elif self.layer_name == "basic_circuit_2":
            self.variables_shape = (self.num_layers, self.num_q, 2)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:, 0])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[:, 1])
                qml.broadcast(qml.CNOT, self.wires[::-1], "chain")

        elif self.layer_name == "basic_circuit_3":
            self.variables_shape = (self.num_layers, 3 * self.num_q - 1)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2*n])
                qml.broadcast(qml.CRZ, self.wires[::-1], "chain",
                              parameters=var[2*n:])

        elif self.layer_name == "basic_circuit_4":
            self.variables_shape = (self.num_layers, 3 * n - 1)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2*n])
                qml.broadcast(qml.CRX, self.wires[::-1], "chain",
                              parameters=var[2*n:])

        elif self.layer_name == "basic_circuit_5":
            self.variables_shape = (self.num_layers,3 * n + n*n)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2*n])
                ind = 2 * n
                for i in range(n - 1, -1, -1):
                    for j in range(n - 1, -1, -1):
                        if i != j:
                            qml.CRZ(var[ind],
                                    wires=[self.wires[i], self.wires[j]])
                            ind += 1

                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[ind:ind+n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[ind+n:ind+2*n])

        elif self.layer_name == "basic_circuit_6":
            self.variables_shape = (self.num_layers, 3 * n + n*n)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2*n])
                ind = 2 * n
                for i in range(n - 1, -1, -1):
                    for j in range(n - 1, -1, -1):
                        if i != j:
                            qml.CRX(var[ind],
                                    wires=[self.wires[i], self.wires[j]])
                            ind += 1

                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[ind:ind+n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[ind+n:ind+2*n])

        elif self.layer_name == "basic_circuit_7":
            self.variables_shape = (self.num_layers, 5 * n - 1)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2 * n])
                ind = 2 * n
                qml.broadcast(qml.CRZ, self.wires[::-1], "double",
                              var[ind: ind + n // 2])
                ind += n // 2
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[ind:ind + n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[ind + n:ind + 2 * n])
                ind += 2 * n
                qml.broadcast(qml.CRZ, self.wires[::-1], "double_odd",
                              var[ind: ind + (n - 1) // 2])

        elif self.layer_name == "basic_circuit_8":
            self.variables_shape = (self.num_layers, 5 * n - 1)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2 * n])
                ind = 2 * n
                qml.broadcast(qml.CRX, self.wires[::-1], "double",
                              var[ind: ind + n // 2])
                ind += n // 2
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[ind:ind + n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[ind + n:ind + 2 * n])
                ind += 2 * n
                qml.broadcast(qml.CRX, self.wires[::-1], "double_odd",
                              var[ind: ind + (n - 1) // 2])

        elif self.layer_name == "basic_circuit_9":
            self.variables_shape = (self.num_layers, n)

            def layer(var):
                qml.broadcast(qml.Hadamard, self.wires, "single")
                qml.broadcast(qml.CZ, self.wires[::-1], "chain")
                qml.broadcast(qml.RX, self.wires, "single", parameters=var)

        elif self.layer_name == "basic_circuit_10":
            self.variables_shape = (self.num_layers, 2*n)

            def layer(var):
                qml.broadcast(qml.RY, self.wires, "single", parameters=var[:n])
                qml.broadcast(qml.CZ, self.wires[::-1], "ring")
                qml.broadcast(qml.RY, self.wires, "single", parameters=var[n:])

        elif self.layer_name == "basic_circuit_11":
            self.variables_shape = (self.num_layers, 4 * n - 4)

            def layer(var):
                ind = n
                qml.broadcast(qml.RY, self.wires, "single",
                              parameters=var[:ind])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[ind: ind + n])
                ind += n
                qml.broadcast(reverse(qml.CNOT), self.wires, "double")
                qml.broadcast(qml.RY, self.wires[1:-1], "single",
                              var[ind:ind + n - 2])
                ind += n - 2
                qml.broadcast(qml.RZ, self.wires[1:-1], "single",
                              var[ind:ind + n - 2])
                ind += n - 2
                qml.broadcast(reverse(qml.CNOT), self.wires, "double_odd")

        elif self.layer_name == "basic_circuit_12":
            self.variables_shape = (self.num_layers, 4 * n - 4)

            def layer(var):
                ind = n
                qml.broadcast(qml.RY, self.wires, "single",
                              parameters=var[:ind])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[ind: ind + n])
                ind += n
                qml.broadcast(reverse(qml.CZ), self.wires, "double")
                qml.broadcast(qml.RY, self.wires[1:-1], "single",
                              var[ind:ind + n - 2])
                ind += n - 2
                qml.broadcast(qml.RZ, self.wires[1:-1], "single",
                              var[ind:ind + n - 2])
                ind += n - 2
                qml.broadcast(reverse(qml.CZ), self.wires, "double_odd")

        elif self.layer_name == "basic_circuit_13":
            # Warning: the implementation might not be correct in the paper
            # TODO: write the original implementation from paper
            #                https://arxiv.org/pdf/1804.00633.pdf
            self.variables_shape = (self.num_layers, 4 * n)

            def layer(var):
                n = len(self.wires)
                pattern1 = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]
                pattern2 = [[i % n, (i - 1) % n] for i in range(-1, n - 1)]

                ind = 0
                qml.broadcast(qml.RY, self.wires, "single", var[ind:ind + n])
                ind += n
                qml.broadcast(qml.CRZ, self.wires, pattern1, var[ind: ind + n])
                ind += n
                qml.broadcast(qml.RY, self.wires, "single", var[ind:ind + n])
                ind += n
                qml.broadcast(qml.CRZ, self.wires, pattern2, var[ind:ind + n])

        elif self.layer_name == "basic_circuit_14":
            # Warning: the implementation might not be correct in the paper
            # TODO: write the original implementation from paper
            #                https://arxiv.org/pdf/1804.00633.pdf
            self.variables_shape = (self.num_layers, 4 * n)

            def layer(var):
                n = len(self.wires)
                pattern1 = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]
                pattern2 = [[i % n, (i - 1) % n] for i in range(-1, n - 1)]

                ind = 0
                qml.broadcast(qml.RY, self.wires, "single", var[ind:ind + n])
                ind += n
                qml.broadcast(qml.CRX, self.wires, pattern1, var[ind: ind + n])
                ind += n
                qml.broadcast(qml.RY, self.wires, "single", var[ind:ind + n])
                ind += n
                qml.broadcast(qml.CRX, self.wires, pattern2, var[ind:ind + n])

        elif self.layer_name == "basic_circuit_15":
            self.variables_shape = (self.num_layers, 2 * n)

            def layer(var):
                pattern1 = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]
                pattern2 = [[i % n, (i - 1) % n] for i in range(-1, n - 1)]

                qml.broadcast(qml.RY, self.wires, "single", parameters=var[:n])
                qml.broadcast(qml.CNOT, self.wires, pattern1)
                qml.broadcast(qml.RY, self.wires, "single", var[n:])
                qml.broadcast(qml.CNOT, self.wires, pattern2)

        elif self.layer_name == "basic_circuit_16":
            self.variables_shape = (self.num_layers, 3 * n - 1)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2 * n])
                ind = 2 * n
                qml.broadcast(reverse(qml.CRZ), self.wires, "double",
                              var[ind: ind + n // 2])
                ind += n // 2
                qml.broadcast(reverse(qml.CRZ), self.wires, "double_odd",
                              var[ind: ind + (n - 1) // 2])

        elif self.layer_name == "basic_circuit_17":
            self.variables_shape = (self.num_layers, 3 * n - 1)

            def layer(var):
                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2 * n])
                ind = 2 * n
                qml.broadcast(reverse(qml.CRX), self.wires, "double",
                              var[ind: ind + n // 2])
                ind += n // 2
                qml.broadcast(reverse(qml.CRX), self.wires, "double_odd",
                              var[ind: ind + (n - 1) // 2])

        elif self.layer_name == "basic_circuit_18":
            self.variables_shape = (self.num_layers, 3 * n)

            def layer(var):
                pattern = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]

                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2 * n])
                qml.broadcast(qml.CRZ, self.wires, pattern,
                              parameters=var[2 * n:])

        elif self.layer_name == "basic_circuit_19":
            self.variables_shape = (self.num_layers, 3 * n)

            def layer(var):
                pattern = [[i % n, (i + 1) % n] for i in range(n - 1, -1, -1)]

                qml.broadcast(qml.RX, self.wires, "single",
                              parameters=var[:n])
                qml.broadcast(qml.RZ, self.wires, "single",
                              parameters=var[n:2 * n])
                qml.broadcast(qml.CRX, self.wires, pattern,
                              parameters=var[2 * n:])

        elif self.layer_name == "StronglyEntanglingLayers":
            self.variables_shape = qml.templates.layers.\
                StronglyEntanglingLayers.shape(self.num_layers, self.num_q)

            def layers(variables):
                qml.templates.layers.StronglyEntanglingLayers(
                    variables,
                    wires=range(self.num_q)
                )

        if layer is not None and layers is None:
            def layers(variables):
                for var in variables:
                    layer(var)

        def ansatz(variables):
            self.check_shape(variables)
            variables = variables.reshape(self.variables_shape)
            layers(variables)

        self.ansatz = ansatz


def reverse(gate):
    if gate.num_params == 0:
        def reversed_gate(wires):
            gate(wires=[wires[1], wires[0]])
    else:
        def reversed_gate(weights, wires):
            gate(weights, wires=[wires[1], wires[0]])
    return reversed_gate