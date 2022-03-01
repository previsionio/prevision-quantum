from common_file import main_function


if __name__ == "__main__":

    prefix = "layerwise_id_blk"

    training_type = "layerwise"
    variables_init_type = "identity_block"
    double_mode = True

    main_function(training_type, prefix, double_mode, variables_init_type)
