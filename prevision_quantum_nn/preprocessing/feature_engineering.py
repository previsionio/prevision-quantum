""" feature engineering module """
import pandas as pd
import numpy as np

has_prevision = False

try:
    import prevision
    has_prevision = True
except ImportError:
    pass


class FeatureEngineer:
    """Class Feature Engineer.

    Attributes:
        feature_engineer (prevision.FeatureEngineer):prevision
            feature engineer
    """
    def __init__(self):
        """  constructor """
        if not has_prevision:
            raise ValueError("Prevision library not loaded")
        self.feature_engineer = prevision.FeatureEngineer()

    def fit_transform(self, features, labels):
        """Fit transform: calls fit and transform afterwards.

        Args:
            features (numpy array):features
            labels (numpy array): labels

        Returns:
            numpy array:
                transformed features
        """
        self.fit(features, labels)
        return self.transform(features)

    def fit(self, features, labels):
        """Fit.

        Args:
            x (numpy array):features
            y (numpy array):labels
        """
        features_df = pd.DataFrame(data=features)
        self.feature_engineer.fit(features_df, labels)

    def transform(self, features):
        """Transform: calls transform.

        Args:
            features (numpy array):features

        Returns:
            numpy array:
                transformed features
        """
        features_df = pd.DataFrame(data=features)
        return np.array(self.feature_engineer.transform(features_df))
