from sklearn import datasets
import numpy as np

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    application_params = "sinusoid_params.json"
    model_weights = "sinusoid_weights_10.npz"
    preprocessor_file = "sinusoid_preprocessor.obj"
    application = qnn.load_application(application_params,
                                       model_weights=model_weights,
                                       preprocessor_file=preprocessor_file)
    train_features = np.linspace(-1. * np.pi, 1. * np.pi, 50)
    train_features = train_features / np.pi
    train_features = train_features.reshape((len(train_features), 1))
    print(application.predict(train_features))
