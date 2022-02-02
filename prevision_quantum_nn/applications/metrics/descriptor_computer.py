""" Base Descriptor module """
import numpy as np
from itertools import chain, combinations


def powerset(iterable_set):
    """
    Returns all the subset of a set.

    Arguments:
        iterable_set: a set of N elements

    Returns:
        power_set: a set of 2^N elements

    Example:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable_set)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def reduced_density_matrix(state, alpha, dim):
    """
    Computes the reduced density matrix of state on the subsystem alpha.
    The main system being the system of dim qubits

    Arguments:
        state (numpy array): complex array of the state vector
        alpha (set[int]): subset of [0, dim-1]
        dim (int): number of qubits
    Returns:
        rho_alpha (numpy array): reduced density matrix
    """
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
        params (dictionary):parameters of the model
        descriptor_type (str): name of the descriptor to be computed
        num_q (int): number of qubits
        variables_range (array of size 2):
            lower and upper bound of each parameter
        variables_sample_size (int): number of parameter vectors used to
            compute the descriptor
        variables_seed (int): the random state for sampling the parameter space
        variables_generator (function): function that generates a random
            parameter vector
        measurement_on_state (function): middle-ground function for computation
        n_bins (int): number of bins for density of probability
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
        self.backend = None

        # variables params
        self.variables_range = params.get("variables_range", [0, 2 * np.pi])
        self.variables_sample_size = params.get("variables_sample_size", 5000)
        self.variables_seed = params.get("variables_seed", 0)
        self.variables_generator = lambda *_, **__: []

        # output-based descriptors params
        self.measurement_on_state = lambda *_, **__: 0

        # expressibility params
        self.n_bins = params.get("n_bins", 75)

    def build_for_model(self, variables_shape, num_q, backend=None):
        """
        Adapts the attributes to the model parameters

        Arguments:
            variables_shape (tuple): shape of the desired parameter vector
            num_q (int): number of qubits
        """
        self.num_q = num_q
        v_min, v_max = self.variables_range
        self.backend = backend

        def variables_generator():
            return np.random.uniform(low=v_min, high=v_max,
                                     size=variables_shape)

        self.variables_generator = variables_generator
        self.backend = backend

    def measurement_on_circuit(self, circuit, data):
        entanglement_measures = []

        np.random.seed(self.variables_seed)
        for i in range(self.variables_sample_size):
            var = self.variables_generator()

            # TODO: change implementation so that we don't use the state vector
            #  (use fidelity)
            state_output = circuit(var, features=data)

            ent = self.measurement_on_state(state_output, self.num_q)
            entanglement_measures.append(ent)

        return np.mean(entanglement_measures)

    def compute_entanglement(self, circuit, dataset,
                             entanglement_type="entangling_capability"):
        """
        Computes entanglement descriptor

        Arguments:
            circuit: circuit for which descriptor is computed
            dataset (numpy array): list of data points
            entanglement_type (str): type of entanglement
                (entangling_capability or concentratable_entanglement)
        """
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

            if self.backend == "damavand.qubit":
                circuit(var_theta)
                fid = circuit.device.get_fidelity_between_two_states_with_parameters(
                        var_theta.flatten(),
                        var_phi.flatten())
            else:
                state_theta = circuit(var_theta)
                state_phi = circuit(var_phi)

                fid = fidelity(state_theta, state_phi)

            fids.append(fid)

        return np.array(fids)

    def compute_expressibility(self, circuit, dataset):
        """
        Computes expressibility

        Arguments:
            circuit: circuit for which descriptor is computed
            dataset (numpy array): list of data points
        """
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
    """
    Computes the Kullback-Leibler divergence

    Arguments:
        P (numpy array): histogram of a density of probability
        Q (numpy array): histogram of another density of probability
    Returns:
        kl_divergence (float): kullback-leibler divergence
    """
    P = np.clip(P, 1e-16, 1)
    Q = np.clip(Q, 1e-16, 1)

    return np.sum(P * np.log(P / Q))


def fidelity(state1, state2):
    """
    computes the fidelity between two state vectors

    Arguments:
        state1 (numpy array): a state vector
        state2 (numpy array): another state vector
    Returns:
        fid (float): the fidelity between the two states
    """
    # states must be vectors
    return np.abs(state1.conjugate().dot(state2)) ** 2


def density_probability(fids, n_bins):
    """
    Returns the density of probability of a set of fidelities

    Arguments:
        fids (list of float): fidelities
        n_bins (int): number of bins for the sampling
    """
    # TODO feat: variable sizes of bins,
    #  give more importance to values close to 0
    hist_fids, bin_edges = np.histogram(fids, bins=n_bins, range=[0, 1])

    return hist_fids / np.sum(hist_fids)
