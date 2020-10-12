""" Phase space plotter module """
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class PhaseSpacePlotter:
    """Class PhaseSpacePlotter.

    Attributes:
        dim (int):dimension of the phase space
    """
    def __init__(self, params):
        """ Initialization of PhaseSpacePlotter """
        self.params = params
        self.x_val = None
        self.y_val = None
        self.has_validation = False
        self.x_plot = None
        self.x_predict = None
        self.preprocessor = None

        self.dim = self.params.get("dim", None)
        self.verbose_period = self.params.get("verbose_period", 10)
        self.prefix = self.params.get("prefix", "qnn_phase_space")
        self.min_max_array = np.array(self.params.get("min_max_array", None))
        self.num_samples_per_axis = self.params.get("num_samples_per_axis", 20)

        if self.dim is None:
            raise KeyError("dim should be provided in the plotter parameters")

        if self.min_max_array is None:
            raise KeyError("min_max_array should be provided in the plotter "
                           "parameters")
        self.min = self.min_max_array[:, 0]
        self.max = self.min_max_array[:, 1]

        if np.any(self.min >= self.max):
            raise ValueError("all components of min should be lower than the "
                             "associated max components. "
                             f"provided min:{self.min} and max {self.max}")
        cmap_name = "prevision_quantum_colors"
        colors = [(120/255, 131/255, 212/255),
                  (134/255, 212/255, 166/255)]
        n_bins = 100
        self.cmap = LinearSegmentedColormap.from_list(cmap_name,
                                                      colors,
                                                      N=n_bins)

    def build(self, preprocessor):
        """ build the plotter
        """
        self.preprocessor = preprocessor
        self.prepare_phase_space()

    def prepare_phase_space(self):
        """ prepares phase space

        to be overridden
        """
        raise NotImplementedError("Implement in doughter class.")
