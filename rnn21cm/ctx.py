def init():
    global inputs, HP, Data, model
    global filepath, logdir, main_process
    global fit_options, compile_options, generators
    global test_data
    global load_model
    global database, formatting

    inputs = None  # argparse.Namespace object, with all input variables
    HP = None  # py21cnn.utilities.AuxiliaryHyperparameters object
    Data = None  # py21cnn.utilities.Data or py21cnn.utilities.LargeData object
    model = None  # keras model

    filepath = None  # data_location + filename for model, summary, logs
    logdir = None  # location of Tensorboard logs
    main_process = None  # bool, in multi-gpu case one of the processes that saves files

    fit_options = None  # dictionary passed to model.fit, model.fit_generator
    compile_options = None  # dictionary passed to model.compile
    generators = None  # in large_run, dictionary of train, validation, test generators

    test_data = (
        []
    )  # in large_run, saving the actual filenames and "labels" is needed for prediction

    load_model = None  # if we are loading the model or have training from beginning - important for warmup callback

    database = {
        "default": [
            "clipped_-250_+50",
            "NaN_removed",
            "TVT_parameterwise",
            "boxcar444",
            "sliced22",
        ],
        "ska": [
            "clipped_-250_+50",
            "NaN_removed",
            "TVT_parameterwise",
            "boxcar444",
            "sliced22",
            "tools21cm",
            "SKA1000",
        ],
        "horizon_wedge": [
            "clipped_-250_+50",
            "NaN_removed",
            "TVT_parameterwise",
            "boxcar444",
            "sliced22",
            "tools21cm",
            "SKA1000",
            "horizon_wedge",
        ],
    }  # different databases descriptors
    formatting = None  # contains a database descriptor
