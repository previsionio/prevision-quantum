from prevision_quantum_nn.applications.metrics.base_descriptor import \
    BaseDescriptor
import numpy as np


class ExpressibilityDescriptor(BaseDescriptor):
    """ExpressibilityDescriptor.

    Base class for further implementations of Deep Q Learners

    Attributes:
        params (dictionary): contains the parameters of the model
        input_size (int):the size of the state space
        model (tf.keras.model):the model itself
        optimizer_name (str):the name of the optimizer, can be
            adam for example
    """
    def __init__(self, params=None):
        """Constructor.

        Args:
            params (dictionary):contains the general parameters
                of the DeepQLearner
        """
        super().__init__(params=params)
        # todo: put a lot of these things in super() method
        #  and maybe add some intermediate classes
        self.model = None
        self.input_size = self.params.get("input_size", None)
        self.type = "deep"
        self.variables_range = params.get("variables_range", [0, 2 * np.pi])
        self.variables_sample_size = params.get("variables_sample_size", 5000)
        self.variables_seed = params.get("variables_seed", 0)
        self.n_bins = params.get("n_bins", 75)
        self.variables_generator = lambda *_, **__: []

    def build_for_model(self, variables_shape, num_q):
        self.num_q = num_q
        v_min, v_max = self.variables_range

        def variables_generator():
            return np.random.uniform(low=v_min, high=v_max,
                                     size=variables_shape)

        self.variables_generator = variables_generator

    def fidelities(self, circuit, data, **kwargs):
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

    def compute(self, circuit, dataset=None, data=None,
                **kwargs):

        if dataset is None:
            if data is None:
                dataset = []
            else:
                dataset = [data]

        expressibilities = []
        for data in dataset:
            fids = self.fidelities(circuit, data, **kwargs)

            P = density_probability(fids, self.n_bins)

            N = 2 ** self.num_q

            hist_haar = np.linspace(0, 1, self.n_bins + 1)
            hist_haar = (1 - hist_haar) ** (N - 1) - np.roll(
                (1 - hist_haar) ** (N - 1), -1)
            Q = hist_haar[:-1]

            expr = kl_divergence(P, Q)

            expressibilities.append(expr)

        return np.mean(expressibilities)


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

