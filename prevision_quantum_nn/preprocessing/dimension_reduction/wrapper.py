""" wrapper module """
import numpy as np
import lightgbm as lgbm


class Wrapper:
    """Class Wrapper.

    Attributes:
        type_problem (str):can be either: classification,
            multiclassification or regression
        features_indexes (list):list of features indexes to
            be retained
    """
    def __init__(self):
        """ constructor """
        self.type_problem = None
        self.features_indexes = None
        self.num_components = None

    def build(self, type_problem):
        """Builds the model.

        Args:
            type_problem (str):type problem
        """
        self.type_problem = type_problem

    def fit(self, features, labels, num_components):
        """Fits observations.

        Args:
            features (numpy array):input features
            labels (numpy array):input labels
            num_components (int):number of components to downscale
                the data to
        """
        self.num_components = num_components
        if labels is None:
            raise ValueError("You have requested dimension reduction "
                             "with lgb wrapper, "
                             "but have not provided with the labels "
                             "y in the parameters of "
                             "preprocess_data")
        num_classes = len(np.unique(labels))
        train_data = lgbm.Dataset(data=features,
                                  label=labels,
                                  free_raw_data=False)
        if self.type_problem == "regression":
            metric = "mse"
            application = "regression"
        elif self.type_problem == "classification":
            metric = "binary_logloss"
            application = "binary"
        elif self.type_problem == "multiclassification":
            metric = "multi_logloss"
            application = "multiclass"
        if self.type_problem == "multiclassification":
            lgbm_params = {
                'boosting': 'dart',
                'application': "multiclass",
                'learning_rate': 0.05,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.7,
                'num_leaves': 41,
                'num_classes': num_classes,
                'metric': metric,
                'drop_rate': 0.15
            }
        else:
            lgbm_params = {
                'boosting': 'dart',
                'application': application,
                'learning_rate': 0.05,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.7,
                'boost_from_average': False,
                'num_leaves': 41,
                'metric': metric,
                'drop_rate': 0.15
            }
        evaluation_results = {}
        clf = lgbm.train(train_set=train_data,
                         params=lgbm_params,
                         evals_result=evaluation_results,
                         num_boost_round=500,
                         early_stopping_rounds=100,
                         verbose_eval=20
                         )
        features_importance = clf.feature_importance()
        self.features_indexes = np.argsort(
            features_importance)[-num_components:]

    def transform(self, features):
        """Transforms the input features to the retained ones.

        Args:
            features (numpy array):input features

        Returns:
            features: numpy array
                output features
        """
        if self.num_components is None:
            raise ValueError("Wrapper.transform: "
                             "run fit on Wrapper before running transform")
        return features[:, self.features_indexes]

    def fit_transform(self, features, labels, num_components):
        """Fit and transforms the input features to the retained ones.

        Args:
            features (numpy array):input features
            labels (numpy array):input labels
            num_components (int):number of components to scale the data

        Returns:
            features: numpy array
                output features
        """
        self.fit(features, labels, num_components)
        return self.transform(features)
