from sklearn.model_selection import train_test_split
import openml as oml

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    """
    OpenML dataset_id: 1462

    num_features: TODO
        angle encoding with 4 qubits
    num_categories: TODO (classification)
    """

    dataset_id = 1462

    # retrieve data
    dataset = oml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format='array',
                                  target=dataset.default_target_attribute)

    # split train and validation
    train_features, val_features, train_labels, val_labels = \
        train_test_split(X, y, test_size=0.33, random_state=42)

    model_params = {
        "architecture": "qubit",
        "TYPE_problem": "classification",
        "num_q": 4,
        "encoding": "angle",
        "interface": "autograd",
        "num_layers": 3,
        "use_early_stopper": False,
        "val_verbose_period": 1,
    }

    dataset = qnn.get_dataset_from_numpy(train_features,
                                         train_labels,
                                         val_features=val_features,
                                         val_labels=val_labels)
    application = qnn.get_application("classification",
                                      prefix=f"openml_{dataset_id}",
                                      model_params=model_params)
    application.solve(dataset)
