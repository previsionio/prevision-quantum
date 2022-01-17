import sys
from common_file import main_function


if __name__ == "__main__":

    nb_turns = float(sys.argv[1])
    prefix = f"zero_init_double_nb_turns_{nb_turns}"
    variables_init_type = "zeros"

    double_mode = True

    main_function(variables_init_type, prefix, nb_turns, double_mode)
