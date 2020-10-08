""" Deep Q Learning module """

from .base_learner import BaseLearner
from prevision_quantum_nn.utils.get_model import get_model
from prevision_quantum_nn.preprocessing.preprocess import Preprocessor
from prevision_quantum_nn.postprocessing.postprocess import Postprocessor


class QNNQLearner(BaseLearner):
    """DeepQLearner.

    Base class for further implementations of Deep Q Learners

    Attributes:
        params (dictionary):contains the parameters of the model
        input_size (int):he size of the state space
        num_layers (int):the number of layers in the deep model
        model (tf.keras.model):the model itself
        optimizer_name (str):the name of the optimizer, can be
            adam for example
    """
    def __init__(self,
                 params,
                 preprocessing_params=None,
                 model_params=None,
                 postprocessing_params=None):
        """Constructor.

        Args:
            params (dictionary):contains the general parameters of
                the DeepQLearner
        """
        super().__init__(params)
        self.num_layers = 5
        self.model = None
        max_num_q = 20

        self.type = "quantum"
        self.num_features = params.get("num_features", None)
        self.num_actions = params.get("num_actions", None)

        if self.num_actions > max_num_q:
            raise ValueError("You requested a quantum circuit with "
                             f"{self.num_actions} qubits/qumodes. "
                             "which is a bit too much for now "
                             "try something with less than 20 actions")

        num_q = max(self.num_actions, self.num_features)
        if num_q > max_num_q:
            raise ValueError("Too many qubits required for this "
                             f"calculation: {num_q}")

        # default params
        default_preprocessing_params = {
            "force_dimension_reduction": True,
            "verbose": False
        }
        default_model_params = {
            "type_problem": "reinforcement_learning",
            "num_actions": self.num_actions,
            "num_q": num_q,
            "max_iterations": 1,
            "interface": "autograd",
            "num_layers": 5
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

        # combine preprocessing params
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

        # init attributes
        self.preprocessor = Preprocessor(preprocessing_params)
        self.model = get_model(model_params)
        self.postprocessor = Postprocessor(postprocessing_params)

        self.model.build()

        self.preprocessor.build_for_model(self.model.architecture_type,
                                          self.model.encoding,
                                          self.model.num_q,
                                          self.model.type_problem,
                                          rl_bounds=self.tanh_bounds,
                                          rl_tanh_mask=self.tanh_mask)

    def fit(self, x_train, y_train):
        """Fit the model.

        Args:
            x_train (numpy array):contains the features of the observations
            y_train (numpy array):contains the targets of the observations
        """
        # do not use fit_transform because x_train is not coming from a dataset
        x_train = self.preprocessor.transform(x_train)
        self.model.fit(x_train, y_train, verbose=False, plotter_callback=None)

    def forward(self, state):
        """
        """
        state = self.preprocessor.transform(state)
        predictions = self.model.predict(state)
        return predictions
