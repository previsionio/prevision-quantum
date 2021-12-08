from prevision_quantum_nn.applications.metrics.base_descriptor import \
    BaseDescriptor
import numpy as np
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def reduced_density_matrix(state, alpha, dim):
    traced_system = [x for x in range(0, dim) if x not in alpha]

    state = state.reshape([2] * dim)
    # Return the reduced density matrix by using numpy tensor product
    rho_alpha = np.tensordot(
        state, np.conj(state), axes=(traced_system, traced_system)
    )
    rho_alpha = rho_alpha.reshape((2**len(alpha), 2**len(alpha)))

    return rho_alpha


def entangling_capability_state(state, dim):
    alpha_set = [[k] for k in range(dim)]
    concent = concentratable_entanglement_state(state, dim, alpha_set)
    ent = 2 * concent
    return ent


def concentratable_entanglement_state(state, dim, alpha_set=None):
    sum_purity = 0
    if alpha_set is None:
        alpha_set = powerset(range(dim))

    for alpha in alpha_set:
        rho_alpha = reduced_density_matrix(state, alpha, dim)
        sum_purity += np.trace(rho_alpha @ rho_alpha)

    concent = 1 - sum_purity / len(alpha_set)

    return float(concent)


class EntanglementDescriptor(BaseDescriptor):
    """EntanglementDescriptor.

    Base class for further implementations of Deep Q Learners

    Attributes:
        params (dictionary): contains the parameters of the model
        input_size (int):the size of the state space
        model (tf.keras.model):the model itself
        optimizer_name (str):the name of the optimizer, can be
            adam for example
    """
    descriptor_types = {
        'entangling_capability': entangling_capability_state,
        'concentratable_entanglement': concentratable_entanglement_state
    }

    def __init__(self, params=None):
        """Constructor.

        Args:
            params (dictionary):contains the general parameters
                of the DeepQLearner
        """
        super().__init__(params=params)
        self.descriptor_type = params.get("descriptor_type",
                                          "entangling_capability")

        if self.descriptor_type not in self.descriptor_types:
            raise ValueError("Incorrect descriptor type")

        self.measurement_on_state = \
            self.descriptor_types[self.descriptor_type]

    def build_for_model(self, variables_shape, num_q):
        self.num_q = num_q
        v_min, v_max = self.variables_range

        def variables_generator():
            return np.random.uniform(low=v_min, high=v_max,
                                     size=variables_shape)

        self.variables_generator = variables_generator

    def measurement_on_circuit(self, circuit, data):
        entanglement_measures = []
        np.random.seed(self.variables_seed)
        for i in range(self.variables_sample_size):
            var = self.variables_generator()

            state_output = circuit(var, features=data)

            ent = self.measurement_on_state(state_output, self.num_q)
            entanglement_measures.append(ent)

        return np.mean(entanglement_measures)

    def compute(self, circuit, dataset=None, data=None):

        if dataset is None:
            if data is None:
                dataset = []
            else:
                dataset = [data]

        entanglement_measures = []
        for data in dataset:
            ent = self.measurement_on_circuit(circuit, data)

            entanglement_measures.append(ent)

        return np.mean(entanglement_measures)


