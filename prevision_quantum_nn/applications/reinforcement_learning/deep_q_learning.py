""" Deep Q Learning module """

import tensorflow as tf

from .base_learner import BaseLearner


class DeepQLearner(BaseLearner):
    """DeepQLearner.

    Base class for further implementations of Deep Q Learners

    Attributes:
        params (dictionary): contains the parameters of the model
        input_size (int):the size of the state space
        model (tf.keras.model):the model itself
        optimizer_name (str):the name of the optimizer, can be
            adam for example
    """
    def __init__(self, params=None):
        """Constructor.

        Args:
            params (dictionary):contains the general parameters
                of the DeepQLearner
        """
        super().__init__(params=params)
        self.model = None
        self.input_size = self.params.get("input_size", None)
        self.type = "deep"

    def fit(self, x_train, y_train):
        """Fit the model.

        Args:
            x_train (numpy array):contains the features of the observations
            y_train (numpy array):contains the targets of the observations
        """
        self.model.fit(x_train, y_train, epochs=1, verbose=False)


class DeepFullyConnectedLearner(DeepQLearner):
    """Deep Fully Connected Learner.

    This deep model is based on a fully connected neural network structure.
    """
    def __init__(self, params=None):
        """Constructor.

        Args:
            params (dictionary):the parameters of the model
        """
        super().__init__(params=params)
        self.build()

    def build(self):
        """ builds the keras model given a fixed strucutre """
        input_features = tf.keras.layers.Input(shape=(self.num_features, ))

        output = tf.keras.layers.Dense(128, activation="relu")(input_features)
        output = tf.keras.layers.Dense(64, activation="relu")(output)
        output = tf.keras.layers.Dense(self.num_actions,
                                       activation="linear")(output)

        self.model = tf.keras.models.Model(input_features, output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                           loss=tf.keras.losses.MSE)

    def forward(self, state):
        """Forwards a state through the neural network.

        Args:
            state (numpy array):state at which the model needs to predict
                the Q-value
        """
        predictions = self.model(state)
        return predictions.numpy()
