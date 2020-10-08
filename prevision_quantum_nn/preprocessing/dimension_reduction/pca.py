""" Principal Component Analysis module """
from sklearn.decomposition import PCA


class PrincipalComponentAnalysis:
    """Class PrincipalComponentAnalysis.

    Attributes:
        pca (sklearn.preprocessing.PCA):sklearn dimension
            reduction fitter
    """
    def __init__(self):
        """ constructor """
        self.pca = None

    def fit_transform(self, features, num_components):
        """Fits and transforms data with PCA.

        Args:
            features (numpy array):input features
            num_components (int):number of components to which the data
                should be decreased
        """
        self.fit(features, num_components)
        return self.transform(features)

    def fit(self, features, num_components):
        """Fits data to num_components.

        Args:
            features (numpy array):input features
            num_components (int):number of components to which the data
                should be decreased
        """
        self.pca = PCA(n_components=num_components)
        features = self.pca.fit(features)

    def transform(self, features):
        """Transforms input features with PCA.

        Args:
            features (numpy array):input features

        Returns:
            numpy array:
                transformed features
        """
        return self.pca.transform(features)
