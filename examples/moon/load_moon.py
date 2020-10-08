
from sklearn import datasets

import prevision_quantum_nn as qnn

if __name__ == "__main__":
    application_params = "moon_params.json"
    model_weights = "moon_weights_100.npz"
    application = qnn.load_application(application_params,
                                       model_weights=model_weights)
    num_samples = 200
    X, y = datasets.make_moons(n_samples=num_samples,
                               noise=0.05, random_state=0)
    print(application.predict(X))
