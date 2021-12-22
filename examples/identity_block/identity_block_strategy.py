import sys
from common_file import main_function


if __name__ == "__main__":

    nb_turns = 0.9 #float(sys.argv[1])
    prefix = f"id_blk_nb_turns_{nb_turns}"
    variables_init_type = "identity_block"

    double_mode = True

    main_function(variables_init_type, prefix, nb_turns, double_mode)
