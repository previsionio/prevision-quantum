def isjupyternotebook():
    """Check if the library is called from a Jupyter notebook environment.
    
    Returns:
        boolean: True if code is used in a Jupyter notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False      # Standard Python interpreter