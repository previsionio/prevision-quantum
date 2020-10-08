""" early stopper module """
import queue
import numpy as np


class EarlyStopper():
    """Class EarlyStopper.

    Used to early stop the convergence of the quantum neural network

    Attributes:
        window (int):early stopper window or patience
        val_losses (list):list of validation losses to be kept
            in memory
        best_var (list):the best set of weights that yielded the
            minimum loss in the window
        best_val_loss (float):best validation loss so far in the window
        buffer (queue):keeps the weights of the models until
            early stopper criterion is met
    """
    def __init__(self, window=10):
        """Constructor.

        Args:
            window (int):default 10, is the early stopper window during
                which the early stopper will keep in memory the variables
        """
        self.window = window
        self.val_losses = []
        self.best_var = None
        self.best_val_loss = np.Infinity
        self.epsilon = 1E-4
        self.buffer = queue.Queue(self.window)

    def add_validation_loss(self, loss):
        """Adds validation loss to loss buffer.

        Also stores the best loss reached so far.

        Args:
            loss (float):loss of the current iteration
        """
        self.best_val_loss = min(loss, self.best_val_loss)
        self.val_losses.append(loss)

    def get_stopping_criterion(self):
        """Stopping_criterions.

        Returns:
            stopping_criterion: bool
                True is algorithm should stop
                False if algorithm should continue
        """
        window_losses = self.val_losses[-self.window:]

        # Check early stop only when the algorithm has
        # at least self.window size
        if len(window_losses) == self.window:

            # Check if last best loss is in still in current window
            if all(w > self.best_val_loss for w in window_losses):
                return True
            # Check if last value has moved more than 1E-4
            # from the last iteration
            if abs(max(window_losses) - min(window_losses)) \
               / max(window_losses) < self.epsilon:
                return True
        return False

    def update(self, val_loss, var):
        """Updates the early stopper at a given iteration.

        Args:
            val_loss (float):validation loss to be stored
            var: (list):list of weights that yielded this loss

        Returns:
            stopping_criterion: bool
                if True, the calculatino will stop
        """
        self.add_validation_loss(val_loss)
        stopping_criterion = self.get_stopping_criterion()
        if stopping_criterion:
            self.best_var = self.buffer.get()
        else:
            if self.buffer.full():
                self.buffer.get()
            self.buffer.put(var)

        return stopping_criterion

    def get_best_var(self):
        """Returns best weights.

        Returns:
            best_var: list
                list of best weights
        """
        return self.best_var
