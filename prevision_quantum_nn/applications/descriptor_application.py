""" classification application module
    provides with the class for classification applications
"""

from prevision_quantum_nn.applications.application import Application
from prevision_quantum_nn.preprocessing.preprocess import Preprocessor
from prevision_quantum_nn.postprocessing.postprocess import Postprocessor
from prevision_quantum_nn.utils.get_descriptor import get_descriptor
from prevision_quantum_nn.utils.get_model import get_model


class DescriptorApplication(Application):
    """Classification Application.

    Attributes:
        model (QuantumNeuralNetwork):model to be trained to
            solve the application
        preprocessor (Preprocessor):preprocessor to be used to
            solve the application
        postprocessor (Postprocessor):postprocessor to be used to
            solve the application
    """

    def __init__(self,
                 prefix="qnn",
                 preprocessing_params=None,
                 model_params=None,
                 postprocessing_params=None,
                 descriptor_params=None):
        """ constructor """
        super().__init__(prefix)

        # default params
        default_preprocessing_params = {"force_dimension_reduction": True}
        default_model_params = {
            "type_problem": "descriptor_computation",
            "encoding": "no_encoding",
        }
        default_descriptor_params = {
            "descriptor_type": "expressibility",

        }

        default_postprocessing_params = dict()

        # combine preprocessing params
        if preprocessing_params:
            preprocessing_params = {
                **default_preprocessing_params,
                **preprocessing_params
            }
        else:
            preprocessing_params = default_preprocessing_params

        # combine model params
        if model_params:
            model_params = {
                **default_model_params,
                **model_params
            }
        else:
            model_params = default_model_params

        # combine postprocessing params
        if postprocessing_params:
            postprocessing_params = {
                **default_postprocessing_params,
                **postprocessing_params
            }
        else:
            postprocessing_params = default_postprocessing_params

        # combine descriptor params
        if descriptor_params:
            descriptor_params = {
                **default_descriptor_params,
                **descriptor_params
            }
        else:
            descriptor_params = default_descriptor_params

        # init attributes
        self.preprocessor = Preprocessor(preprocessing_params)
        self.model = get_model(model_params)
        self.postprocessor = Postprocessor(postprocessing_params)
        self.descriptor = get_descriptor(descriptor_params)

    def build(self):
        """ builds the application according to dataset characteristics """

        # get dataset information
        if self.dataset:
            unused_variables = []
            if self.dataset.train_labels is not None:
                unused_variables.append("train_labels")
            if self.dataset.val_features is not None:
                unused_variables.append("val_features")
            if self.dataset.val_labels is not None:
                unused_variables.append("val_labels")
            if unused_variables:
                unused_variables = ", ".join(unused_variables)
                raise Warning(f"Unused variables: {unused_variables}. "
                              f"Not needed for descriptor computation")

        # build model
        self.model.build()

        # build preprocessing (mandatory)
        self.preprocessor.build_for_model(self.model.architecture_type,
                                          self.model.encoding,
                                          self.model.num_q,
                                          self.model.type_problem)

        self.descriptor.build_for_model(
            self.model.ansatz_builder.variables_shape,
            self.model.num_q)

        if self.model.num_q > self.max_num_q:
            raise ValueError("Too many qubits required for "
                             "this calculation: "
                             f"{self.model.num_q} > {self.max_num_q} "
                             "(maximum)")

        self.logger.info("successfully built")
        self.log_params()

    def compute(self, dataset=None):
        """Computes the descriptor given a certain dataset.

        Args:
            dataset (DataSet):dataset to be solved with this application
        """

        self.dataset = dataset

        # build application
        self.build()

        if self.model.architecture_type != 'discrete':
            raise ValueError("Invalid architecture. "
                             "Descriptors can only be computed with "
                             "discrete architecture for now")

        # preprocess data
        # features = self.preprocessor.fit_transform(dataset, [])
        features = None
        if self.dataset:
            features = self.dataset.train_features

        # build postprocessor
        # self.postprocessor.build(self.preprocessor)

        # save params and preprocessor before computing descriptor
        # self.save_params()
        # self.save_preprocessor()

        # compute descriptor

        descriptor_value = self.descriptor.compute(
            self.model.neural_network,
            dataset=features
        )

        return descriptor_value
