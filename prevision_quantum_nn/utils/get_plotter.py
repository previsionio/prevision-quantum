""" get plotter module """
from prevision_quantum_nn.postprocessing.plotter.phase_space_plotter_1d \
    import PhaseSpacePlotter1D
from prevision_quantum_nn.postprocessing.plotter.phase_space_plotter_2d \
    import PhaseSpacePlotter2D


def get_plotter(params):
    """Returns a plotter given plotting parameters.

    Args:
        params (dictionary):parameters of the plotter

    Returns:
        plotter: PhaseSpacePlotter
            phase space plotter
    """
    dim = params.get("dim", 2)
    if dim == 1:
        plotter = PhaseSpacePlotter1D(params)
    elif dim == 2:
        plotter = PhaseSpacePlotter2D(params)
    elif dim > 2:
        plotter = PhaseSpacePlotter2D(params)
        print("The dimension is bigger than 2. A PCA will be applied. "
              f"Current dim: {dim}")
    else:
        raise ValueError("Plotting dimension "
                         "should be 1, 2 or more. "
                         f"Currently set to {dim}")
    return plotter
