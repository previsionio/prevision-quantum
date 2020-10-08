""" Quantum Neural Network module
    provides with the base class of Quantum Neural Networks from which all
    models should inherit
"""
from sklearn.utils import resample
import numpy as np
from sklearn import metrics
import tensorflow as tf
import pennylane as qml
import logging

from prevision_quantum_nn.models.utilities.early_stopper import EarlyStopper


class QuantumNeuralNetwork:
    """Quantum Neural Network
        base class for all quantum neural networks

    Attributes:
        params (dictionnary):contains the parameters of the model
        use_early_stopper (bool):if True, early stopping will be activated,
            default: True
        early_stopper (EarlyStopper):early stopper that stops the run when
            the validation loss increases
        early_stopper_patience (int):number of iterations during which the
            early stopper module will save the weights and restore them
            when triggered
        postprocessor (Postprocessor):postprocessor to be called as a
            callback during the run
        built (bool):True if the model has been built, it False,
            fit won't start
        running_mode (str):default, "simulation", but could also be
            "computation"once the access to quantum computers will be
            effective
        architecture (str):architecture of the quantum computer - can be:
            1. qubit
            2. cv
        num_q (int):number of qubits/qumodes
        num_categories (int):number of categories/classes/labels of the problem
            only the first num_categories qubits/qumodes will be measured
        num_actions (int):number of actions in the case of a reinforcement
            learning mode
        max_iterations (int):maximum number of iteration that the fitting phase
            needs to perform
        num_layers (int):number of layers of the quantum neural network
        snapshot_frequency (int):frequency in number of iterations at which the
            model needs to snapshot
        type_problem (str):problem that is being solved, can be:
            1. classifiation
            2. multiclassificaiton
            3. regression
            4. reinforcement_learning
        batch_size (int):batch size to be used for one fitting iteration
        save (bool):if True, the model will save at the end of fit
        prefix (str):name of the file to which the output should go to
    """
    def __init__(self, params):
        """
        Args:
            params (dictionnary):Parameters of the QuantumNeuralNetwork
                Required
        """
        self.params = params

        self.architecture_type = None
        self.params_file = None
        self.early_stopper = None
        self.postprocessor = None
        self.built = False
        self.logger = logging.getLogger('model')
        self.iteration = 0
        self.val_verbose_period = 1

    def build(self):
        """ build the model: initializes the weights """

        self.running_mode = self.params.get("running_mode", "simulation")
        self.architecture = self.params.get("architecture", "qubit")
        self.num_q = self.params.get("num_q", 2)
        self.num_categories = self.params.get("num_categories", None)
        self.num_actions = self.params.get("num_actions", None)
        self.max_iterations = self.params.get("max_iterations", 50000)
        self.num_layers = self.params.get("num_layers", 2)
        self.snapshot_frequency = self.params.get("snapshot_frequency", 0)
        self.type_problem = self.params.get("type_problem", None)
        self.batch_size = self.params.get("batch_size", 1)
        self.use_early_stopper = self.params.get("use_early_stopper", True)
        self.val_verbose_period = self.params.get("val_verbose_period", 1)
        self.early_stopper_patience = self.params.get(
            "early_stopper_patience", 20)
        self.prefix = self.params.get("prefix", "qnn")

        if self.use_early_stopper:
            self.build_early_stopper()

        self.check_model()

        self.built = True

    def check_model(self):
        """Checks the model's parameters consistency.

        Raises:
            ValueError when needed
        """
        if not isinstance(self.num_q, int) or self.num_q < 1:
            raise ValueError("num_q ({}) must be a positive integer",
                             self.num_q)

        if self.type_problem == "classification":
            self.num_categories = 2

        if self.type_problem == "reinforcement_learning":
            self.num_categories = self.num_actions

        if self.type_problem != "regression":
            if not isinstance(self.num_categories, int) or \
               self.num_categories < 1:
                raise ValueError("num_categories must be a positive integer")

        if self.type_problem == "regression" and \
                self.num_categories is not None:
            raise ValueError("remove the option num_categories "
                             "for regression tasks")

        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        if self.type_problem is None:
            raise ValueError("type_problem must be defined")

        if not isinstance(self.prefix, str):
            raise ValueError("prefix must be a string")

        if not isinstance(self.use_early_stopper, bool):
            raise ValueError("use_early_stopper must be True or False")

        if self.type_problem not in ["classification", "multiclassification",
                                     "regression", "reinforcement_learning"]:
            raise ValueError("Non valid problem type. Possible options are: "
                             "classification, "
                             "multiclassification, "
                             "regression, "
                             "reinforcement_learning")

    def build_early_stopper(self):
        """Builds the early stoppe."""
        if not isinstance(self.early_stopper_patience, int) or \
                self.early_stopper_patience < 1:
            raise ValueError("early_stopper_patience must be "
                             "a positive integer")
        self.early_stopper = EarlyStopper(window=self.early_stopper_patience)

    def get_random_batch(self, features, labels, batch_size):
        """Get random batch.
        Args:
            features (numpy array):features to be randomly selected
            labels (numpy array):labels to be randomly selected

        Returns:
            features: numpy array
                randomized batched features
            labels: numpy array
                randomized batched labels
        """
        features, labels = resample(features, labels, n_samples=batch_size)
        return features, labels

    def logging_iteration(self,
                          val_features,
                          val_labels,
                          train_loss,
                          val_loss):
        """Dumps information during training.

        Args:
            val_features (array):validation features
            val_labels (array):validation labels
            train_loss (float):loss of the current iteration
            val_loss (float):loss of the current iteration
        """
        if isinstance(train_loss, qml.numpy.tensor):
            train_loss = train_loss[0]
        if isinstance(val_loss, qml.numpy.tensor):
            val_loss = val_loss[0]
        if isinstance(train_loss, tf.Tensor):
            train_loss = train_loss.numpy()[0]
        if isinstance(val_loss, tf.Tensor):
            val_loss = val_loss.numpy()[0]

        if val_features is not None and \
           self.iteration % self.val_verbose_period == 0:
            # classification
            if self.type_problem == "classification":
                preds = self.predict(val_features)
                fpr, tpr, _ = metrics.roc_curve(val_labels, preds)
                auc = metrics.auc(fpr, tpr)
                self.logger.info(f"iter: {self.iteration} "
                                 f"train_loss: {train_loss:.3e} "
                                 f"val_loss: {val_loss:.3e} "
                                 f"auc: {auc:.5f}")

            # multiclassification
            elif self.type_problem == "multiclassification":
                preds = self.predict(val_features)
                accuracy = metrics.accuracy_score(
                        np.argmax(val_labels, axis=1), preds)
                self.logger.info(f"iter: {self.iteration} "
                                 f"train_loss: {train_loss:.3e} "
                                 f"val_loss: {val_loss:.3e} "
                                 f"accuracy: {accuracy:.5f}")

            # regression or reinforcement_learning:
            else:
                self.logger.info(f"iter: {self.iteration} "
                                 f"train_loss: {train_loss:.3e} "
                                 f"val_loss: {val_loss:.3e}")
        else:
            self.logger.info(f"iter: {self.iteration} "
                             f"train_loss: {train_loss:.3e}")
