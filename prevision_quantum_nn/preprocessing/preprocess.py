""" preprocessing module """
import logging

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from prevision_quantum_nn.preprocessing.polynomial_expansion \
        import PolynomialExpander
from prevision_quantum_nn.preprocessing.feature_engineering \
        import FeatureEngineer
from prevision_quantum_nn.preprocessing.dimension_reduction.wrapper \
        import Wrapper
from prevision_quantum_nn.preprocessing.dimension_reduction.pca\
        import PrincipalComponentAnalysis


class Preprocessor:
    """Class Preprocessor, preprocesses features for quantum models.

    Attributes:
        polynomial_expander (PolynomialExpander):the polynomial expander
        polynomial_expansion_type (str):can be either polynomial_features
            or kronecker
        feature_engineer (FeatureEngineer):feature engineer object
        dimension_reduction_fitter (str):can be either wrapper or pca
        force_dimension_reduction (bool):if True, dimension reduction will be
            forced if False, errors will prompt if the data does not fit into
            the quantum computer
    """
    def __init__(self, params):
        """Constructor.

        Args:
            params (dictionnary):parameters of the preprocessor
        """
        self.params = params
        self.polynomial_expander = None
        self.feature_engineer = None
        self.wrapper = None
        self.pca = None
        self.reduce_dimension = False
        self.num_components = None
        self.type_problem = None
        self.architecture_type = None
        self.encoding = None
        self.num_q = None
        self.rl_bounds = None
        self.rl_tanh_mask = None
        self.rl_discrete_depth = None

        self.verbose = self.params.get("verbose", True)
        self.polynomial_degree = self.params.get("polynomial_degree", 1)
        self.polynomial_expansion_type = self.params.get(
            "polynomial_expansion_type", "polynomial_features")
        self.feature_engineering = self.params.get(
            "feature_engineering", False)
        self.force_dimension_reduction = self.params.get(
            "force_dimension_reduction", False)
        self.dimension_reduction_fitter = self.params.get(
            "dimension_reduction_fitter", "wrapper")
        self.padding_parameter = self.params.get("padding", 0.)

        if self.polynomial_degree > 1:
            self.polynomial_expander = \
                PolynomialExpander(
                    degree=self.polynomial_degree,
                    expansion_type=self.polynomial_expansion_type
                )

        if self.feature_engineering:
            self.feature_engineer = FeatureEngineer()

        if self.dimension_reduction_fitter == "wrapper":
            self.wrapper = Wrapper()
        elif self.dimension_reduction_fitter == "pca":
            self.pca = PrincipalComponentAnalysis()

        self.scaler = MinMaxScaler((0, np.pi))

        self.check_preprocessor()
        self.logger = logging.getLogger("preprocessing")

    def check_preprocessor(self):
        """Checks the preprocessor consistency.

        Raises:
            ValueError if parameters are not consistent
        """
        if not isinstance(self.force_dimension_reduction, bool):
            raise ValueError("force_dimension_reduction must be True or False")

        if not isinstance(self.polynomial_degree, int) or \
                self.polynomial_degree < 1:
            raise ValueError("polynomial_degree must be a positive integer")

        if not isinstance(self.padding_parameter, float):
            raise ValueError("padding must be a number")

        if not isinstance(self.feature_engineering, bool):
            raise ValueError("feature_engineering must be True or False")

        if self.polynomial_expansion_type not in ['polynomial_features',
                                                  'kronecker']:
            raise ValueError("Undefined polynomial expansion type. "
                             "Please use polynomial_features or kronecker")

    def build_for_model(self,
                        architecture_type,
                        encoding,
                        num_q,
                        type_problem,
                        rl_bounds=None,
                        rl_tanh_mask=None,
                        rl_discrete_depth=None):
        """Builds the preprocessor for a given model.

        Args:
            model (QuantumNeuralNetwork):model that needs to be used with
                this preprocessing
        """
        self.type_problem = type_problem
        self.architecture_type = architecture_type
        self.encoding = encoding
        self.num_q = num_q

        if self.wrapper:
            self.wrapper.build(type_problem)

        self.rl_bounds = rl_bounds
        self.rl_tanh_mask = rl_tanh_mask
        self.rl_discrete_depth = rl_discrete_depth

        if self.rl_bounds is not None:
            self.scaler.fit_transform(rl_bounds)

    def fit_transform(self, features, labels):
        """Fit and transforms features.

        Args:
            features (numpy array):input features
            labels (numpy array): data label

        Returns:
            features: numpy array
                transformed features
        """
        # polynomial expansion
        if self.polynomial_expander:
            if self.verbose:
                self.logger.info("performing polynomial expansion with degree "
                                 f"{self.polynomial_degree}.")
            features = self.polynomial_expander.fit_transform(features)

        # feature engineering
        if self.feature_engineer:
            if self.verbose:
                self.logger.info("performing prevision.io feature engineering")
            features = self.feature_engineer.fit_transform(features, labels)

        # reduce dimension
        self.compute_dimension_reduction_params(features.shape[1])

        if self.reduce_dimension:
            if self.verbose:
                self.logger.info("performing dimension reduction with "
                                 f"{self.dimension_reduction_fitter}.")
            if self.dimension_reduction_fitter == "pca":
                features = self.pca.fit_transform(features,
                                                  self.num_components)
            elif self.dimension_reduction_fitter == "wrapper":
                features = self.wrapper.fit_transform(features,
                                                      labels,
                                                      self.num_components)
            else:
                raise ValueError("Invalid dimension reduction fitter."
                                 "Please use wrapper or pca")
        # scale
        if self.encoding == "angle":
            features = self.scaler.fit_transform(features)

        # padding
        features = self.apply_padding(features)

        return features

    def transform(self, features):
        """Transforms features.

        Args:
            features (numpy array):input features

        Returns:
            features: numpy array
                transformed features
        """
        # apply tanh for features which can be infinite in RL
        if self.rl_tanh_mask is not None:
            features[:, self.rl_tanh_mask] = \
                tf.tanh(features[:, self.rl_tanh_mask])

        # polynomial expansion
        if self.polynomial_expander:
            features = self.polynomial_expander.transform(features)

        # feature engineering
        if self.feature_engineer:
            features = self.feature_engineer.transform(features)

        # dimension reduction
        if self.reduce_dimension:
            if self.dimension_reduction_fitter == "pca":
                features = self.pca.transform(features)
            elif self.dimension_reduction_fitter == "wrapper":
                features = self.wrapper.transform(features)
            else:
                raise ValueError("Invalid dimension reduction fitter."
                                 "Please use wrapper or pca")
        # scale
        if self.encoding == "angle":
            if self.type_problem == "reinforcement_learning" and \
                    self.rl_bounds is None:
                pass
            else:
                features = self.scaler.transform(features)

        # padding
        features = self.apply_padding(features)

        return features

    def compute_dimension_reduction_params(self, data_dim):
        """Compute dimensions reduction parameters.

        Args:
            data_dim (int):dimension of the data
        """

        reduce_dimension = False
        num_components = None

        # Continuous Variable quantum computing
        if self.architecture_type == "continuous_variable":
            if self.encoding == "amplitude":
                raise ValueError("Amplitude encoding incompatible with "
                                 "Continuous Variable model")
            # perform pca if data dimension is greater than number of qubits
            if data_dim > self.num_q:
                if self.force_dimension_reduction:
                    reduce_dimension = True
                    num_components = self.num_q
                else:
                    raise ValueError("Number of qubits "
                                     f"required ({data_dim}) is greater "
                                     "than the number of available qubits "
                                     f"({self.num_q}). "
                                     "Force dimension reduction to fit "
                                     "data into available qumodes")
        # Discrete quantum computing
        elif self.architecture_type == "discrete":
            # amplitude or mottonen encoding
            if self.encoding == "amplitude" or self.encoding == "mottonen":
                num_amplitudes = 2 ** self.num_q
                # check if data fits into
                # the number of available amplitudes
                if data_dim > num_amplitudes:
                    if self.force_dimension_reduction:
                        reduce_dimension = True
                        num_components = num_amplitudes
                        if num_components < 1:
                            raise ValueError("You have forced pca "
                                             "and polynomial expansion. "
                                             "But this decreases the data "
                                             "dimension down to 0 "
                                             "components. "
                                             "Reduce the degree of "
                                             "the polynomial expansion")
                    else:
                        raise ValueError("Number of amplitudes "
                                         f"required ({data_dim}) is greater "
                                         "than the number of available "
                                         "amplitudes "
                                         f"({num_amplitudes}). "
                                         "Force dimension reduction"
                                         "to fit data into "
                                         "available amplitudes")
            # angle encoding
            elif self.encoding == "angle":
                if data_dim > self.num_q:
                    if self.force_dimension_reduction:
                        reduce_dimension = True
                        num_components = self.num_q
                    else:
                        raise ValueError("Number of qubits "
                                         f"required ({data_dim}) is greater "
                                         "than the number of "
                                         f"available qubits ({self.num_q}). "
                                         "Force dimension reduction to fit "
                                         "data into available amplitudes")

        self.reduce_dimension = reduce_dimension
        self.num_components = num_components

    def apply_padding(self, features):
        """Apply padding to data

        Args:
            features (numpy array):features to be padded
        """
        num_features = features.shape[1]
        # check if padding is necessary
        if self.encoding in ["amplitude", "mottonen"]:
            padding_dim = 2 ** self.num_q - num_features
        elif self.encoding in ["angle", "displacement", "squeezing"]:
            padding_dim = self.num_q - num_features

        # apply padding
        if padding_dim > 0:
            obs_shape = np.shape(features)[0]
            if self.verbose:
                self.logger.info("padding input array "
                                 f"with {padding_dim} "
                                 "fake feature(s) equal to "
                                 f"{self.padding_parameter}.")
            padding = self.padding_parameter*np.ones((obs_shape, padding_dim))
            features = np.hstack([features, padding])
        return features
