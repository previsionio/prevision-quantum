""" postprocessing module """
import logging
from prevision_quantum_nn.utils.get_plotter import get_plotter


class Postprocessor:
    """Postprocessor.

    Attributes:
        plotter (PhaseSpacePlotter):plotter to be used to plot phase space
    """
    def __init__(self, params):
        """Constructor.

        Args:
            params (dictionnary):postprocessing parameters
        """
        self.params = params
        self.plotter = None
        self.logger = logging.getLogger('postprocessing')

    @classmethod
    def get_params_attributes(cls):
        """Attributes that can be set as a parameter"""
        return ["phase_space_plotter"]

    def build(self, preprocessor):
        """Builds the postprocessor using preprocessor.

        Args:
            preprocessor (Preprocessor):preprocessor identical to
                the one used to build the model
        """
        # todo: add a check_params function for plotting_params
        plotting_params = self.params.get("phase_space_plotter")
        if plotting_params:
            self.plotter = get_plotter(plotting_params)
            self.plotter.build(preprocessor)

    def get_plotter(self):
        """Gets plotter.

        Returns:
            plotter: PhaseSpacePlotter
                built plotter
        """
        return self.plotter

    def callback(self, model):
        """Callback.

        Args:
            model (QuantumNeuralNetwork):model that needs to be used with
                this preprocessing
        """
        if self.plotter and model.iteration % self.plotter.verbose_period == 0:
            self.logger.info("generating phase space plot.")
            self.plotter.plot(model)

    def check_params(self):
        """
        This first implementation requires that we respect the underscore
        naming convention.
        Another implementation would be to write every 'correct arguments' in
        a list. They are the attributes xxx defined as:
            self.xxx = self.params.get('xxx', 'yyy')
        """
        for key in self.params.keys():
            if key not in self.__dict__:
                print(f"Incorrect key '{key}'")
                for self_key in self.__dict__:
                    if self_key.find(key) >= 0 and self_key[0] != "_":
                        print(f"    You can try '{self_key}'")
