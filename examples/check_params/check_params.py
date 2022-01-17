import prevision_quantum_nn as qnn

if __name__ == "__main__":

    # customize preprocessing
    preprocessing_params = {
        "polynomial_degree": 2,
        "num_q": 4,
        "early_stopper": None,
    }

    # customize model
    model_params = {
        "architecture": "qubit",
        "num_q": 5,
        "encoding": "angle",
        "use_early_stopper": True,
        "early_stopper_patience": 20,
        "max_iterations": 10000,
        "interface": "autograd",
        "layer_type": "template",
        "snapshot_frequency": 5,
        "verbose": True,
        "prefix": "moon",
        "num_layers": 4,
        "optimizer_name": "Adam",
        "learning_rate": 0.05,

        "early_stopper": True,
        "nb_iterations": 2000,
        "optimizer": "GradientDescent",

    }

    # build application
    application = qnn.get_application(
        "classification",
        prefix="moon",
        preprocessing_params=preprocessing_params,
        model_params=model_params,
    )

    application.check_params()
