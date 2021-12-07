from sklearn import datasets
from sklearn.model_selection import train_test_split

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    # prepare data
    X, y = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)
    x_train, x_val, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                       random_state=42)

    dataset = qnn.get_dataset_from_numpy(x_train, y_train,
                                         val_features=x_val, val_labels=y_test)
    application = qnn.get_application("classification")
    application.solve(dataset)
