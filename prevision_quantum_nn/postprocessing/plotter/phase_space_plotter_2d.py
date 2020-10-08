""" phase space plotter 2D module """
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from prevision_quantum_nn.postprocessing.plotter.phase_space_plotter \
        import PhaseSpacePlotter


class PhaseSpacePlotter2D(PhaseSpacePlotter):
    """Class PhaseSpacePlotter2D.

    Plotter for a phase space of dimension 2

    Attributes:
        x_val (numpy array):validation features to be plotted with
            the phase space
        y_val (numpy array):validation labels to be plotted with
            the phase space
        x_plot (numpy array):array of features to map the phase space
        has_validation (bool):tells if validation is set to on or not
    """
    def __init__(self, plotting_params):
        """ constructor """
        super().__init__(plotting_params)
        self.xxp = None
        self.yyp = None

    def set_validation_data(self, features, labels):
        """Sets validation data.

        Args:
            features (numpy array):validation features to be plotted with
                the phase space
            labels (numpy array):validation labels to be plotted with
                the phase space

        Raises:
            ValueError if dimensions do not match 2
        """
        if features.shape[1] != 2:
            ValueError("X must have dimension [num_samples, 2], "
                       f"currently have shape: {np.shape(features)}")
        self.x_val = features
        self.y_val = labels
        self.has_validation = True

    def prepare_phase_space(self):
        """Prepare phase space of PhaseSpacePlotter2D.

        Checks consistency of min and max
        creates xxp, yyp for drawing
        and x_plot for predicting
        """
        if np.any(self.min) is None or np.any(self.max) is None:
            raise ValueError("use first to define")
        if np.shape(self.min) != (2,) or np.shape(self.max) != (2,):
            raise ValueError("min and max should have shapes(2,). "
                             f"provided {np.shape(self.min)} and "
                             f"{np.shape(self.max)}")
        xp_ = np.linspace(self.min[0], self.max[0],
                          self.num_samples_per_axis)
        yp_ = np.linspace(self.min[1], self.max[1],
                          self.num_samples_per_axis)
        self.xxp, self.yyp = np.meshgrid(xp_, yp_)
        self.x_plot = np.vstack((self.xxp.flatten(), self.yyp.flatten())).T
        self.x_predict = self.preprocessor.transform(self.x_plot)

    def plot(self, model):
        """Plot the phase space of a model predictions
            in 2D at current_iteration.

        Args:
            model (QuantumNeuralNetwork):model to be used to plot phase space
            cuttent_iteration (int):iteration at which the model is
            force_plot (bool):if current iteration does not
                match verbose_period, you can also force the plot
                with this option set to True
        """
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(7, 7)
        phase_space_predictions = model.predict(self.x_predict)

        if model.type_problem == "classification" or \
           model.type_problem == "multiclassification":
            axes.contourf(
                self.x_plot[:, 0].reshape((
                    self.num_samples_per_axis,
                    self.num_samples_per_axis)),
                self.x_plot[:, 1].reshape((
                    self.num_samples_per_axis,
                    self.num_samples_per_axis)),
                phase_space_predictions.reshape((
                    self.num_samples_per_axis,
                    self.num_samples_per_axis)),
                cmap=self.cmap
            )
            if self.has_validation:
                axes.scatter(self.x_val[:, 0],
                             self.x_val[:, 1],
                             c=self.y_val,
                             cmap=cm.binary)
        elif model.type_problem == "regression" or \
                model.type_problem == "reinforcement_learning":
            axes.contourf(phase_space_predictions
                          .reshape(self.num_samples_per_axis,
                                   self.num_samples_per_axis))
        axes.set_xlim([self.min[0], self.max[0]])
        axes.set_ylim([self.min[1], self.max[1]])
        plt.savefig(self.prefix + "_" + str(model.iteration) + ".png")
        plt.close()
