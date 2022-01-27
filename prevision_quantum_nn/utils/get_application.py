""" get application module """
import os
import json
import pickle

from prevision_quantum_nn.applications.classification_application \
        import ClassificationApplication
from prevision_quantum_nn.applications.multiclassification_application \
        import MultiClassificationApplication
from prevision_quantum_nn.applications.regression_application \
        import RegressionApplication
from prevision_quantum_nn.applications.reinforcement_learning_application \
        import ReinforcementLearningApplication
from prevision_quantum_nn.applications.descriptor_application \
        import DescriptorApplication
from prevision_quantum_nn.utils.get_model import get_model
from prevision_quantum_nn.preprocessing.preprocess import Preprocessor
from prevision_quantum_nn.postprocessing.postprocess import Postprocessor


def get_application(application_type,
                    prefix="qnn",
                    preprocessing_params=None,
                    model_params=None,
                    postprocessing_params=None,
                    descriptor_params=None,
                    rl_learner_type="quantum"):
    """Get application.

    Args:
        application_type (str): application type can be
             1. classification
             2. multiclassification
             3. regression
             4. reinforcement_learning
             5. descriptor computation
        prefix (str): prefix to each filename
        preprocessing_params (dict): parameters of the preprocessor
        model_params (dict): parameters of the model
        postprocessing_params (dict): parameters of the postprocessor
        descriptor_params (dict): parameters of the descriptor computer
        rl_learner_type (str): type of learner of reinforcement learning

    Returns:
        application: Application
            application according to application type
    """
    application = None

    if application_type == "classification":
        ClassificationApplication.check_params(
            preprocessing_params,
            model_params,
            postprocessing_params)
        application = ClassificationApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params)

    elif application_type == "multiclassification":
        MultiClassificationApplication.check_params(
            preprocessing_params,
            model_params,
            postprocessing_params)
        application = MultiClassificationApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params)

    elif application_type == "regression":
        RegressionApplication.check_params(
            preprocessing_params,
            model_params,
            postprocessing_params)
        application = RegressionApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params)

    elif application_type == "reinforcement_learning":
        ReinforcementLearningApplication.check_params(
            preprocessing_params,
            model_params,
            postprocessing_params)
        application = ReinforcementLearningApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params,
            rl_learner_type=rl_learner_type)

    elif application_type == "descriptor_computation":
        DescriptorApplication.check_params(
            preprocessing_params,
            model_params,
            postprocessing_params)
        application = DescriptorApplication(
            prefix,
            preprocessing_params,
            model_params,
            postprocessing_params,
            descriptor_params=descriptor_params)

    else:
        raise ValueError(f"No such type of application {application_type}.\n"
                         f"Try classification, multiclassification, "
                         f"regression, reinforcement_learning "
                         f"or descriptor_computation")

    return application


def load_application(application_params, model_weights, preprocessor_file):
    """ loads application from files """

    if os.path.exists(application_params):
        with open(application_params, "rb") as parameters_file:
            params = json.load(parameters_file)
        print("params loaded from file.")
    else:
        raise ValueError(f"Application params file cannot be found: "
                         f"{application_params}")
                         
    if os.path.exists(preprocessor_file):
        with open(preprocessor_file, "rb") as prepros_file:
            loaded_preprocessor = pickle.load(prepros_file)
        print("Preprocessor loaded from file.")
    else:
        print(f"Preprocessor file cannot be found: {preprocessor_file}")

    application = get_application(
        params.get("model_params").get("type_problem"))

    # get preprocessor
    application.preprocessor = loaded_preprocessor

    # get model
    application.model = get_model(params.get("model_params"))

    # get postprocessor
    application.postprocessor = Postprocessor(
        params.get("postprocessing_params"))
    application.postprocessor.build(application.preprocessor)

    application.model.build(weights_file=model_weights)
    return application
