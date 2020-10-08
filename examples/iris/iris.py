import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

import prevision_quantum_nn as qnn

if __name__=="__main__":

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

    # customize application
    model_params = {
        "num_q": 4, 
        "max_iterations": 100000,
        "verbose": True,
        "num_layers": 4,
        "interface": "autograd",
        "learning_rate": 0.05,
        "encoding": "angle",
        "early_stopper_patience": 50,
    }

    preprocessing_params = {
        "polynomial_degree": 1
    }
                          
    # build application
    application = qnn.get_application("multiclassification",
                                       prefix="iris",
                                       preprocessing_params=preprocessing_params,
                                       model_params=model_params)

    # before solving the application, save the parameters
    # in order to be able to reload them in case of 
    # interruption of the solve method
    # you will still have the weights file genereated by the snapshot_frequency keyword
    # and will be able to reload the application, with the weights you want
    application.save_params()

    application.solve(dataset)
