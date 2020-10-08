""" get plotter module """
from prevision_quantum_nn.postprocessing.plotter.phase_space_plotter_1d \
        import PhaseSpacePlotter1D
from prevision_quantum_nn.postprocessing.plotter.phase_space_plotter_2d \
        import PhaseSpacePlotter2D


def get_plotter(params):
    """Returns a plotter given plotting parameters.

    Args:
        params (dictionnary):parameters of the plotter

    Returns:
        plotter: PhaseSpacePlotter
            phase space plotter
    """
    dim = params.get("dim", 2)
    if dim == 1:
        plotter = PhaseSpacePlotter1D(params)
    elif dim == 2:
        plotter = PhaseSpacePlotter2D(params)
    else:
        raise ValueError("Plotting dimension "
                         "should be 1 or 2. "
                         f"Currently set to {dim}")
    return plotter
