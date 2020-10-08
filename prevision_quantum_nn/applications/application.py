""" Application module:
    contains base class for applications
"""

import logging
import json


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

    def predict(self, val_features):
        """Predict. Returns prediction of the mode

        Args:
            x_val (numpy array): weights
        """
        return self.model.predict(val_features)

    def log_params(self):
        """ logs parameters of the application
            at the begining of the output file
        """

        # log preprocesssing params
        self.logger.info("Preprocessing parameters:")
        self.logger.info("\n"+json.dumps(self.preprocessor.params,
                                         indent=4,
                                         sort_keys=True))

        # log model params
        self.logger.info("Model parameters:")
        self.logger.info("\n"+json.dumps(self.model.params,
                                         indent=4,
                                         sort_keys=True))

        # log postprocesssing params
        self.logger.info("Postprocessing parameters:")
        self.logger.info("\n"+json.dumps(self.postprocessor.params,
                                         indent=4,
                                         sort_keys=True))
