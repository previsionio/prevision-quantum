""" phase space plotter 1D module """
import numpy as np
import matplotlib.pylab as plt
from prevision_quantum_nn.postprocessing.plotter.phase_space_plotter \
        import PhaseSpacePlotter


class PhaseSpacePlotter1D(PhaseSpacePlotter):
    """Class PhaseSpacePlotter1D.

    Plotter for a phase space of dimension 1

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

    def set_validation_data(self, features, labels):
        """Sets validation data.

        Args:
            features (numpy array):validation features to be plotted with
                the phase space
            labels (numpy array):validation labels to be plotted with
                the phase space

        Raises:
            ValueError if dimensions do not match 1
        """
        if features.shape[1] != 1:
            ValueError("X must have dimension [num_samples, 1], "
                       f"currently have shape: {np.shape(features)}")
        self.x_val = features
        self.y_val = labels
        self.has_validation = True

    def prepare_phase_space(self):
        """Prepare phase space of PhaseSpacePlotter1D.

        Checks consistency of min and max
        creates x_plot for predicting and drawing
        """
        if self.min is None or self.max is None:
            raise ValueError("use set_phase_space first to define min and max")
        if np.shape(self.min) != (1,) or np.shape(self.max) != (1,):
            raise ValueError("min and max should have shapes (1,). "
                             f"provided {np.shape(self.min)} and "
                             f"{np.shape(self.max)}")
        self.x_plot = np.linspace(self.min[0], self.max[0],
                                  self.num_samples_per_axis)
        self.x_predict = self.preprocessor.transform(
            self.x_plot.reshape((len(self.x_plot), 1))
        )

    def plot(self, model):
        """Plot the phase space of a model predictions
            in 1D at current_iteration

        Args:
            model (QuantumNeuralNetwork):model to be used to plot phase space
            cuttent_iteration (int):iteration at which the model is
            force_plot (bool):if current iteration does not match
                verbose_period, you can also force the plot with
                this option set to True
        """
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(7, 7)
        phase_space_predictions = model.predict(self.x_predict)
        colors = [(120/255, 131/255, 212/255),
                  (134/255, 212/255, 166/255)]
        if model.type_problem == "classification" or \
           model.type_problem == "multiclassification":
            axes.scatter(self.x_plot,
                         phase_space_predictions, axis=1)
        elif model.type_problem == "regression":
            axes.scatter(self.x_plot,
                         phase_space_predictions,
                         color=colors[1])
            if self.has_validation:
                axes.scatter(self.x_val,
                             self.y_val,
                             color=colors[0])

        axes.set_xlim([self.min[0], self.max[0]])
        plt.savefig(self.prefix + "_" + str(model.iteration)+".png")
        plt.close()
