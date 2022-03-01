from common_file import main_function


if __name__ == "__main__":

    prefix = "global_zero"

    training_type = "default"
    variables_init_type = "zeros"
    double_mode = False

    main_function(training_type, prefix, double_mode, variables_init_type)
