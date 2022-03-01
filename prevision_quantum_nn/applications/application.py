""" Application module:
    contains base class for applications
"""

import logging
import json
import dill as pickle

from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane_cv import \
    CVNeuralNetwork
from prevision_quantum_nn.models.pennylane_backend.qnn_pennylane_qubit import \
    PennylaneQubitNeuralNetwork
from prevision_quantum_nn.postprocessing.postprocess import Postprocessor
from prevision_quantum_nn.preprocessing.preprocess import Preprocessor
import sys

if "Levenshtein" in sys.modules:
    from Levenshtein import distance as levenshtein_distance


class Application:
    """Base Application class. Applications will inherite from this class.

    Attributes:
        dataset (DataSet):the dataset to be solved
        max_num_q (int):maximum number of qubits/qumodes
    """

    def __init__(self, prefix="qnn"):
        """ Constructor.

        Args:
            prefix (string):name used for filenames
        """
        self.prefix = prefix
        self.dataset = None
        self.max_num_q = 20
        self.preprocessor = None
        self.model = None
        self.postprocessor = None
        self.descriptor_computer = None

        # Logging configuration
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        date_strftime_format = "%d-%b-%y %H:%M:%S"
        self.output_file_name = prefix + ".listing"
        logging.basicConfig(filename=self.output_file_name,
                            filemode='w',
                            datefmt=date_strftime_format,
                            level=logging.INFO,
                            format='%(asctime)s | %(name)s | %(message)s')

        self.logger = logging.getLogger('application')

    def save_params(self):
        """ save parameters into file """
        params = {
            "preprocessing_params": self.preprocessor.params,
            "model_params": self.model.params,
            "postprocessing_params": self.postprocessor.params
        }

        params_file_name = self.prefix + "_params.json"
        with open(params_file_name, 'w') as params_file:
            json.dump(params, params_file, indent=4)

    def save_preprocessor(self):
        """ save preprocessor into file """
        if self.preprocessor:
            preprocessor_file_name = self.prefix + "_preprocessor.obj"
            with open(preprocessor_file_name, 'wb') as f:
                pickle.dump(self.preprocessor, f)

    def predict(self, val_features):
        """Predict. Returns prediction of the model

        Args:
            val_features (numpy array): validation features
        """
        # todo: something's wrong here
        if self.preprocessor:
            preprocessed_features = self.preprocessor.transform(val_features)
        return self.model.predict(preprocessed_features)

    def log_params(self):
        """ logs parameters of the application
            at the begining of the output file
        """

        # log preprocesssing params
        self.logger.info("Preprocessing parameters:")
        self.logger.info("\n" + json.dumps(self.preprocessor.params,
                                           indent=4,
                                           sort_keys=True))

        # log model params
        self.logger.info("Model parameters:")
        to_dump = self.model.params
        for key, value in list(to_dump.items()):
            if callable(value):
                del to_dump[key]
        self.logger.info("\n" + json.dumps(to_dump,
                                           indent=4,
                                           sort_keys=True))

        # log postprocesssing params
        self.logger.info("Postprocessing parameters:")
        self.logger.info("\n" + json.dumps(self.postprocessor.params,
                                           indent=4,
                                           sort_keys=True))

    @classmethod
    def check_params(cls, preprocessing_params,
                     model_params,
                     postprocessing_params):
        if preprocessing_params:
            print("-- Check preprocessor params --")
            valid_params = cls.get_valid_preprocessing_params(
                preprocessing_params)
            cls._check_valid_params(preprocessing_params, valid_params)

        if model_params:
            print("-- Check model params --")
            valid_params = cls.get_valid_model_params(model_params)
            cls._check_valid_params(model_params, valid_params)

        if postprocessing_params:
            print("-- Check postprocessor params --")
            valid_params = cls.get_valid_postprocessing_params(
                postprocessing_params)
            cls._check_valid_params(postprocessing_params, valid_params)

    @classmethod
    def get_valid_preprocessing_params(cls, preprocessing_params):
        return Preprocessor.get_params_attributes()

    @classmethod
    def get_valid_model_params(cls, model_params):
        architecture = model_params.get("architecture", "qubit")

        if architecture == "qubit":
            valid_params = PennylaneQubitNeuralNetwork.get_params_attributes()
        elif architecture == "cv":
            valid_params = CVNeuralNetwork.get_params_attributes()
        else:
            raise ValueError("Invalid architecture. "
                             "Choices are qubit or cv")
        return valid_params

    @classmethod
    def get_valid_postprocessing_params(cls, postprocessing_params):
        return Postprocessor.get_params_attributes()

    @classmethod
    def _check_valid_params(cls, params, valid_params):
        valid_params = set(valid_params)
        for param in params:
            if param not in valid_params:
                suggestions_message = ""
                if "Levenshtein" in sys.modules:
                    candidate_params = []
                    cmp = lambda x: levenshtein_distance(x, param)
                    for valid_param in valid_params:
                        if cmp(valid_param) <= 5 or valid_param.find(param) >= 0:
                            candidate_params.append(valid_param)
                    candidate_params.sort(key=cmp)
                    if candidate_params:
                        suggestions_message = f"Try with:\n" + \
                                              "\n".join(candidate_params)
                raise ValueError(f"Parameter {param} is not valid. " +
                                 suggestions_message)
