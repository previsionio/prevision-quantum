""" polynomial expansion module """

import functools

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class PolynomialExpander():
    """Class PolynomialExpander: expands features.

    Attributes:
        degree (int):polynomial degree of the expansion
        expansion_type (str):expansion type to be applied:
            1. polynomial_features
            2. kronecker
        poly (sklearn.preprocessiong.PolynomialFeatures):
            polynomial feature engineer
    """

    def __init__(self, degree=2, expansion_type="polynomial_features"):
        """Constructor.

        Args:
            degree (int):degree of the polynomial expansion, default 2
            expansion_type (str):type of the polynomial expansion,
                default "polynomial_features"
        """
        self.degree = degree
        self.expansion_type = expansion_type
        self.poly = None

        if self.expansion_type == "polynomial_features":
            self.poly = PolynomialFeatures(degree,
                                           interaction_only=False,
                                           include_bias=False)

    def fit(self, features):
        """Fits the expander.

        Args:
            features (numpy array):input features
        """
        if self.expansion_type == "polynomial_features":
            self.poly.fit(features)

    def transform(self, features):
        """Transforms the expander.

        Args:
            features (numpy array):input features

        Returns:
            numpy array:
                transformed features
        """
        # polynomial features engine
        if self.expansion_type == "polynomial_features":
            features = self.poly.transform(features)
        # kronecker engine
        elif self.expansion_type == "kronecker":
            features_ = []
            for obs in range(features):
                obs = np.reshape(obs, (obs.shape[0], 1))
                to_kron = [np.ones(obs.shape)]
                for _ in range(self.degree):
                    to_kron.append(obs)
                features_.append(np.array(
                    functools.reduce(np.kron, to_kron)).ravel())
            features = np.array(features_)
        return features

    def fit_transform(self, features):
        """Fits and then transforms the input.

        Args:
            features (numpy array):input features

        Returns:
            numpy array:
                fit and transformed features
        """
        self.fit(features)
        return self.transform(features)
