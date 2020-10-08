""" Pennylane module
    contains the base class of quantum neural networks
    based on pennylane
"""
import time
import linecache

import pennylane as qml
import pennylane.numpy as np
import tensorflow as tf

from prevision_quantum_nn.models.qnn import QuantumNeuralNetwork
from prevision_quantum_nn.models.utilities.losses \
    import square_loss, cross_entropy
from prevision_quantum_nn.models.utilities.to_categorical import to_categorical

OPTIMIZER_NAMES = ["SGD", "Adagrad", "Adam", "RMSProp"]


class PennylaneNeuralNetwork(QuantumNeuralNetwork):
    """Class PennylaneNeuralNetwork.

    Attributes:
        params (dict):dictionary containing the main parameters of the model
        optimizer (Optimizer):Optimizer of the quantum circuit.
            Can be AdamOptimizer or NesterovMomentumOptimizer
        batch_size (int):size of the batch with which the training
            should be perfomed
        verbose (bool):sets the verbosity to on if True and off if False
        interface (str):interface of the pennylane backend. Can be tf
            or autograd
        learning_rate: learning rate at which the fitting phase needs to
            be performed

    Methods:
        encode_data(self, x)
            needs to be overridden by child class
        layer(self, v)
            needs to be overridden by child class
        output_layer(self, v)
            needs to be overridden by child class
        neural_network(self, var, features=None)
            Main method that is decorated by the qml.qnode decorator.
            This will set the structure of the neural network
        cost(self, var, features, labels)
            cost function to be optimized
    """
    def __init__(self, params):
        """ constructor
        Args:
            params: dictionnary
                parameters of the model
        """
        super().__init__(params)
        self.iteration = 0

        # retrieve model parameters
        self.learning_rate = self.params.get("learning_rate", 0.01)
        self.val_verbose_period = self.params.get("val_verbose_period", 5)
        self.optimizer_name = self.params.get("optimizer_name", "Adam")
        self.interface = self.params.get("interface", "autograd")
        self.layer_type = self.params.get("layer_type", "template")
        self.encoding = self.params.get("encoding", None)
        self.optimizer = None
        self.var = None
        self.dev = None
        self.neural_network = lambda *_, **__: None
        self.backend = None

    def initialize_weights(self, file_name=None):
        """ initialize weights

        to be implemented depending on the architecture used
        """
        raise NotImplementedError("Implement this method in doughter classe.")

    def build(self):
        """ builds the optimizer and initializes weights """
        super().build()

        self.build_optimizer()
        self.initialize_weights()
        self.built = True

    def build_optimizer(self):
        """Builds the optimizer according to its name and to the interface used.
        """
        if self.optimizer_name not in OPTIMIZER_NAMES:
            raise ValueError("Optimizer name not recognized: "
                             f"{self.optimizer_name}")
        # autograd interface
        if self.interface == "autograd":
            if self.optimizer_name == "SGD":
                self.optimizer = \
                    qml.optimize.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer_name == "Adagrad":
                self.optimizer = \
                    qml.optimize.AdagradOptimizer(self.learning_rate)
            elif self.optimizer_name == "Adam":
                self.optimizer = \
                    qml.optimize.AdamOptimizer(self.learning_rate)
            elif self.optimizer_name == "RMSProp":
                self.optimizer = \
                    qml.optimize.RMSPropOptimizer(self.learning_rate)
        # interface
        elif self.interface == "tf":
            if self.optimizer_name == "SGD":
                self.optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate)
            elif self.optimizer_name == "Adagrad":
                self.optimizer = tf.keras.optimizers.Adagrad(
                    learning_rate=self.learning_rate)
            elif self.optimizer_name == "Adam":
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate)
            elif self.optimizer_name == "RMSProp":
                self.optimizer = tf.keras.optimizers.RMSProp(
                    learning_rate=self.learning_rate)

    def snapshot(self):
        """Snapshots the model to a file."""
        current_file = self.prefix + "_weights_" + \
            str(self.iteration) + ".npz"

        if self.interface == "tf":
            tosave = [v.numpy() for v in self.var]
        elif self.interface == "autograd":
            tosave = self.var

        np.savez(current_file, *tosave)

    def load_weights(self, weights_file):
        """Loads weights from file.

        Args:
            weights_file (string):file name containing the weights
        """
        weights_dict = np.load(weights_file)
        weights_list = []
        for _, value in weights_dict.iteritems():
            weights_list.append(value)
        self.var = weights_list

        if self.interface == "tf":
            self.var = [tf.Variable(v) for v in self.var]

    def cost(self, var, features, labels):
        """Cost to be optimized during training.

        Args:
            var (list):weights of the model
            features (array):obervations to be evalueated by the model
            labels (array):labels associated to features

        Returns:
            loss: float
                loss of the model given x
        """
        preds = [self.neural_network(var, features=x_) for x_ in features]
        if self.interface == "autograd":
            if self.type_problem == "regression":
                loss = square_loss(labels, preds)
            elif self.type_problem == "classification":
                loss = square_loss(labels, preds)
            elif self.type_problem == "multiclassification":
                preds = np.array(preds)
                preds = np.exp(preds)/np.sum(np.exp(preds), axis=1)[:, None]
                loss = cross_entropy(labels, preds)
            elif self.type_problem == "reinforcement_learning":
                loss = np.mean(square_loss(labels, preds))
        elif self.interface == "tf":
            if self.type_problem == "regression":
                loss = square_loss(labels, preds)
            elif self.type_problem == "classification":
                loss = square_loss(labels, preds)
            elif self.type_problem == "multiclassification":
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels, preds),
                    keepdims=True
                )
            elif self.type_problem == "reinforcement_learning":
                loss = np.mean(square_loss(labels, preds))
        return loss

    def step(self, features, labels, var):
        """Performs one step of training.

        Args:
            features(array):observations
            labels(array):labels
            var (array):weights of the model
        """
        if self.interface == "autograd":
            var = self.optimizer.step(lambda v:
                                      self.cost(v, features, labels), var)
        elif self.interface == "tf":
            with tf.GradientTape() as tape:
                loss = self.cost(var, features, labels)
                # FIXME
                # due to pennylane layer templating
                # in CV, we got lists of tf.Variable
                # in qubit, we got tf.Variables, which are not iterable
                if isinstance(var, tf.Variable):
                    gradients = tape.gradient(loss, [var])
                    self.optimizer.apply_gradients(zip(gradients, [var]))
                else:
                    gradients = tape.gradient(loss, var)
                    self.optimizer.apply_gradients(zip(gradients, var))
        return var

    def fit(self,
            train_features,
            train_labels,
            plotter_callback,
            val_features=None,
            val_labels=None,
            verbose=True):
        """Fits data with model.

        Args:
            train_features (array):training features
            train_labels (array):training labels
            val_features (array):validation features
            val_labels (array):validation labels
            verbose (bool):if True, verbosity will be activated
        """
        if not self.built:
            raise ValueError("Build the model before fitting any data.")

        start_fit = time.time()

        # to categorical for multiclassification
        if self.type_problem == "multiclassification":
            train_labels, val_labels = to_categorical(train_labels, val_labels)

        if verbose:
            self.logger.info("starting iterations")

        self.iteration = 0

        var = self.var

        # iterate
        stopping_criterion = False
        while not stopping_criterion and self.iteration < self.max_iterations:

            if self.batch_size > 1:
                x_train, y_train = self.get_random_batch(train_features,
                                                         train_labels,
                                                         self.batch_size)
            else:
                x_train = train_features
                y_train = train_labels

            var = self.step(x_train, y_train, var)

            if self.backend == "strawberryfields.tf":
                linecache.clearcache()

            # early stopper
            val_loss = None
            if val_features is not None:
                val_loss = self.cost(var, val_features, val_labels)
                if self.early_stopper:
                    stopping_criterion = \
                            self.early_stopper.update(val_loss, var)
                    if stopping_criterion:
                        var = self.early_stopper.get_best_var()
                        best_iter = self.iteration-self.early_stopper_patience
                        self.logger.info("early stopper stopped - "
                                         "restoring best weights. "
                                         f"Best iter: {best_iter}")

            # dump output
            if verbose:
                train_loss = self.cost(var, x_train, y_train)
                self.logging_iteration(val_features,
                                       val_labels,
                                       train_loss,
                                       val_loss)

            # if snapshot enabled, save weights in file
            if self.snapshot_frequency > 0 and \
                    self.iteration % self.snapshot_frequency == 0:
                self.logger.info(f"iter: {self.iteration} "
                                 "snapshotting weights into file.")
                self.snapshot()

            # plotter
            if plotter_callback is not None:
                plotter_callback(self)

            if not stopping_criterion:
                self.iteration += 1

            # for prediction
            self.var = var

        # print elapsed time
        elapsed_time = time.time() - start_fit
        if verbose:
            self.logger.info(f"elapsed time (s): {elapsed_time:.3e}")

    def predict(self, features):
        """Predicts certain obervations.

        Args:
            features (array):observations to be predicted

        Returns:
            preds: float or int
                prediction of the model
        """
        preds = np.array([self.neural_network(self.var, features=x_)
                          for x_ in features])
        if self.type_problem == "classification":
            preds = np.where(preds > 0.5, 1, 0)
        elif self.type_problem == "multiclassification":
            preds = np.exp(preds)/np.sum(np.exp(preds), axis=1)[:, None]
            preds = np.argmax(preds, axis=1)
        return preds
