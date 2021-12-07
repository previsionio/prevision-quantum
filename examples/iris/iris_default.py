import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    # retrieve data and build dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    X = df.to_numpy()
    y = iris.target

    train_features, val_features, train_labels, val_labels = train_test_split(
        X, y, test_size=0.33, random_state=42)

    dataset = qnn.get_dataset_from_numpy(train_features,
                                         train_labels,
                                         val_features=val_features,
                                         val_labels=val_labels)

    application = qnn.get_application("multiclassification",
                                      prefix="iris")
    application.solve(dataset)
