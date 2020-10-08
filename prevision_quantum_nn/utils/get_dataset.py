""" get dataset module """

from prevision_quantum_nn.dataset.dataset import DataSet


def get_dataset_from_numpy(train_features,
                           train_labels,
                           val_features=None,
                           val_labels=None):
    """ get dataset from numpy
    """
    return DataSet().from_numpy(train_features,
                                train_labels,
                                val_features=val_features,
                                val_labels=val_labels)


def get_dataset_from_pandas(train_data_frame, targets, val_data_frame=None):
    """ get dataset from pandas
    """
    return DataSet().from_pandas(train_data_frame,
                                 targets,
                                 val_data_frame=val_data_frame)
