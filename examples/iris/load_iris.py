import numpy as np
import pandas as pd

from sklearn import datasets

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    application_params = "iris_params.json"
    model_weights = "iris_weights_10.npz"
    preprocessor_file = "iris_preprocessor.obj"
    application = qnn.load_application(application_params,
                                       model_weights=model_weights,
                                       preprocessor_file=preprocessor_file)
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data)
    X = df.to_numpy()
    print(application.predict(X))
