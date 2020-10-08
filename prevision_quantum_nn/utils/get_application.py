""" get application module """
import os
import json

from prevision_quantum_nn.applications.classification_application \
        import ClassificationApplication
from prevision_quantum_nn.applications.multiclassification_application \
        import MultiClassificationApplication
from prevision_quantum_nn.applications.regression_application \
        import RegressionApplication
from prevision_quantum_nn.applications.reinforcement_learning_application \
        import ReinforcementLearningApplication
from prevision_quantum_nn.utils.get_model import get_model
from prevision_quantum_nn.preprocessing.preprocess import Preprocessor
from prevision_quantum_nn.postprocessing.postprocess import Postprocessor


def get_application(application_type,
                    prefix="qnn",
                    preprocessing_params=None,
                    model_params=None,
                    postprocessing_params=None,
                    rl_learner_type="quantum"):
    """Get application.

    Args:
        application_type (str): application type can be
             1. classification
             2. multiclassification
             3. regression
             4. reinforcement_learning

    Returns:
        application: Application
            application according to application type
    """
    application = None

    if application_type == "classification":
        application = ClassificationApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params)
    elif application_type == "multiclassification":
        application = MultiClassificationApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params)
    elif application_type == "regression":
        application = RegressionApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params)
    elif application_type == "reinforcement_learning":
        application = ReinforcementLearningApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params,
            rl_learner_type=rl_learner_type)
    else:
        raise ValueError(f"No such type of application {application_type}")

    return application


def load_application(application_params, model_weights):
    """ loads application from files """

    if os.path.exists(application_params):
        with open(application_params, "rb") as parameters_file:
            params = json.load(parameters_file)
        print("utils.get_model_from_parameters_file:"
              "params loaded from file.")
    else:
        raise ValueError(f"Application params file cannot be found: "
                         f"{application_params}")

    application = get_application(params.get("model_params").get("TYPE_problem"))

    # get preprocessor
    application.preprocessor = Preprocessor(params.get("preprocessing_params"))

    # get model
    application.model = get_model(params.get("model_params"))

    # get postprocessor
    application.postprocessor = Postprocessor(params.get("postprocessing_params"))

    application.model.load(model_weights)
    application.model.build()
    application.preprocessor.build_for_model(application.model)

    return application
