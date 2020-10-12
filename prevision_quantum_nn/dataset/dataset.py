""" Dataset module

contains the classes to handle datasets
"""

import numpy as np


class DataSet:
    """DataSet.

    Abstraction to handle user data

    Attributes:
        num_features (int):number of features in the dataset
        num_categories (int):number of categories in the case
            of multiclassification
    """
    def __init__(self):
        """ constructor """
        self.num_features = None
        self.num_categories = None
        self.train_features = None
        self.train_labels = None
        self.val_features = None
        self.val_labels = None

    def from_numpy(self,
                   train_features,
                   train_labels,
                   val_features=None,
                   val_labels=None):
        """Creates a DataSet from a set of numpy arrays.

        Args:
            train_features (numpy array):array of features for
                the training phase
            train_labels (numpy array):array of labels for
                the training phase
            val_features (numpy array):array of features for
                the validation phase
            val_labels (numpy array):array of labels for
                the validation phase

        Returns:
            a dataset: DataSet
        """
        # train
        self.train_features = train_features
        self.train_labels = train_labels

        # val
        self.val_features = val_features
        self.val_labels = val_labels

        self.num_features = self.train_features.shape[1]

        if val_labels is not None:
            labels = np.hstack((train_labels, val_labels))
            self.num_categories = len(np.unique(labels))
        else:
            self.num_categories = len(np.unique(train_labels))

        return self

    def from_pandas(self, train_data_frame, targets, val_data_frame):
        """Creates a DataSet from a set of pandas dataframes.

        Args:
            train_data_frame (pandas DataFrame):dataframe for the
                training phase
            targets (list):list of targets columns in input
                data frames
            val_data_frame (pandas DataFrame):dataframe for
                the validation phase

        Returns:
            a dataset: DataSet
        """
        # train
        self.train_features = train_data_frame.drop(targets).to_numpy()
        self.train_labels = train_data_frame["targets"].to_numpy()

        # val
        self.val_features = train_data_frame.drop(targets).to_numpy()
        self.val_labels = val_data_frame["targets"].to_numpy()

        self.num_features = self.train_features.shape[1]

        if val_labels is not None:
            labels = np.hstack((train_labels, val_labels))
            self.num_categories = len(np.unique(labels))
        else:
            self.num_categories = len(np.unique(train_labels))

        return self

    def to_numpy(self):
        """Returns the dataset in the numpy format.

        Returns:
            self.train_features: numpy array
                training features
            self.train_labels: numpy array
                training labels
            self.val_features: numpy array
                validation features
            self.val_labels: numpy array
                validation labels
        """
        return self.train_features, self.train_labels, \
            self.val_features, self.val_labels
