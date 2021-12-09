""" Base Descriptor module """
import numpy as np
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def reduced_density_matrix(state, alpha, dim):
    traced_system = [x for x in range(0, dim) if x not in alpha]

    state = state.reshape([2] * dim)
    # Return the reduced density matrix by using numpy tensor product
    rho_alpha = np.tensordot(
        state, np.conj(state), axes=(traced_system, traced_system)
    )
    rho_alpha = rho_alpha.reshape((2 ** len(alpha), 2 ** len(alpha)))

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

    return concent.real


class DescriptorComputer:
    """Descriptor Computer.

    Attributes:
    """
    entanglement_types = {
        'entangling_capability': entangling_capability_state,
        'concentratable_entanglement': concentratable_entanglement_state
    }

    def __init__(self, params):
        """constructor """
        self.params = params
        self.descriptor_type = params.get('descriptor_type', "expressibility")
        self.num_q = params.get("num_q", 2)
        self.variables_range = params.get("variables_range", [0, 2 * np.pi])
        self.variables_sample_size = params.get("variables_sample_size", 5000)
        self.variables_seed = params.get("variables_seed", 0)
        self.variables_generator = lambda *_, **__: []
        self.measurement_on_state = lambda *_, **__: None

        # expressibility params
        self.n_bins = params.get("n_bins", 75)

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

    def compute_entanglement(self, circuit, dataset,
                             entanglement_type="entangling_capability"):

        self.measurement_on_state = self.entanglement_types[entanglement_type]

        entanglement_measures = []
        for data in dataset:
            ent = self.measurement_on_circuit(circuit, data)

            entanglement_measures.append(ent)

        return np.mean(entanglement_measures)

    def fidelities(self, circuit, data):
        fids = []
        np.random.seed(self.variables_seed)

        for i in range(self.variables_sample_size):
            var_theta = self.variables_generator()
            var_phi = self.variables_generator()

            state_theta = circuit(var_theta, features=data)
            state_phi = circuit(var_phi, features=data)

            fid = fidelity(state_theta, state_phi)

            fids.append(fid)

        return np.array(fids)

    def compute_expressibility(self, circuit, dataset):
        expressibilities = []
        for data in dataset:
            fids = self.fidelities(circuit, data)
            P = density_probability(fids, self.n_bins)

            N = 2 ** self.num_q

            hist_haar = np.linspace(0, 1, self.n_bins + 1)
            hist_haar = (1 - hist_haar) ** (N - 1) - np.roll(
                (1 - hist_haar) ** (N - 1), -1)
            Q = hist_haar[:-1]

            expr = kl_divergence(P, Q)
            expressibilities.append(expr)

        return np.mean(expressibilities)

    def compute(self, circuit, dataset=None, descriptor_type="expressibility"):
        """ computes the descriptor

        to be implemented depending on the descriptor considered
        :param circuit:
        :param dataset:
        :param descriptor_type:
        :return: descriptor_value
        """
        if dataset is None:
            dataset = [None]

        if descriptor_type == "expressibility":
            return self.compute_expressibility(circuit, dataset)
        if descriptor_type in self.entanglement_types:
            return self.compute_entanglement(circuit, dataset,
                                             entanglement_type=descriptor_type)


def kl_divergence(P, Q):
    P = np.clip(P, 1e-16, 1)
    Q = np.clip(Q, 1e-16, 1)

    return np.sum(P * np.log(P / Q))


def fidelity(state1, state2):
    # states must be vectors
    return np.abs(state1.conjugate().dot(state2)) ** 2


def density_probability(fids, n_bins):
    hist_fids, bin_edges = np.histogram(fids, bins=n_bins, range=[0, 1])

    return hist_fids / np.sum(hist_fids)
