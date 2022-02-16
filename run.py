###############################################################################
# define context, module with important variables
###############################################################################
from rnn21cm import ctx

ctx.init()
###############################################################################
# parsing inputs
###############################################################################
import argparse

parser = argparse.ArgumentParser(prog="Model Fit")

# database
parser.add_argument(
    "--database",
    type=str,
    choices=ctx.database.keys(),
    default="default",
    help="Database to be used for the run. (default: default)",
)
parser.add_argument(
    "--X_fstring",
    type=str,
    default="{}_realization_{:d}_{:03d}_of_{:03d}.tfrecord",
    help="f-string defining filename for the database.",
)
parser.add_argument(
    "--normalize_X",
    type=int,
    choices=[0, 1],
    default=1,
    help="Normalize input or not. (default: 1)",
)
parser.add_argument(
    "--N_realizations",
    type=int,
    default=10,
    help="Number of noise realizations to train on. (default: 10)",
)

# model and hyperparameters
parser.add_argument(
    "--model",
    type=str,
    default="RNN.SummarySpace3D",
    help="Keras model to be used for training. (default: RNN.SummarySpace3D)",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="",
    help="RNN or CNN model, inferred from `model` by default.",
)
parser.add_argument(
    "--HyperparameterIndex",
    type=int,
    default=0,
    help="Combination index in the list of all possible hyperparameters. (default: 0)",
)
parser.add_argument(
    "--simple_run",
    type=int,
    choices=[0, 1],
    default=0,
    help=(
        "If 1, runs hyperparameters defined in `py21cnn.hyperparamters.HP_simple`. "
        "Else, runs hyperparameters defined with `HyperparameterIndex`. "
        "(default: 0)"
    ),
)

# training setup
parser.add_argument(
    "--seed", type=int, default=-1, help="By default -1, which picks a seed randomly."
)
parser.add_argument(
    "--warmup",
    type=int,
    default=0,
    help="In the case of multi-gpu training, for how many epochs to linearly increase learining rate? (default: 0)",
)
parser.add_argument(
    "--LR_correction",
    type=int,
    choices=[0, 1],
    default=1,
    help="In the case of multi-gpu training, should learning rate be multiplied by number of GPUs? (default: 0)",
)
parser.add_argument(
    "--mixed_precision",
    type=int,
    choices=[0, 1],
    default=0,
    help="Either to use mixed precision or not.",
)
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs. (default: 1)")
parser.add_argument(
    "--workers",
    type=int,
    default=24,
    help="Number of cpu workers for data loading. (default: 24)",
)
parser.add_argument(
    "--eager",
    type=int,
    choices=[0, 1],
    default=1,
    help="Run the code eagerly? (default: 1)",
)

parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train. (default: 200)"
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=-1,
    help=(
        "In the case of re-training, sets an upper bound on the total number of epochs."
        "By default, the same as `epochs`."
    ),
)

# saving and output
parser.add_argument(
    "--data_location", type=str, default="data/", help="Defaults to `data/`"
)
parser.add_argument(
    "--saving_location", type=str, default="models/", help="Defaults to `models/`"
)
parser.add_argument(
    "--tensorboard",
    type=int,
    choices=[0, 1],
    default=1,
    help="Creating TensorBoard logs during the run? (default: 1)",
)
parser.add_argument(
    "--logs_location", type=str, default="logs/", help="Defaults to `logs/`"
)
parser.add_argument(
    "--file_prefix",
    type=str,
    default="",
    help="File prefix to all outputs of the program - saved models, logs. (default: `None`)",
)
parser.add_argument(
    "--verbose",
    type=int,
    choices=[0, 1, 2],
    default=2,
    help="Verbosity of the stdout. (default: 2)",
)


inputs = parser.parse_args()
ctx.formatting = ctx.database[inputs.database]
inputs.normalize_X = bool(inputs.normalize_X)
inputs.LR_correction = bool(inputs.LR_correction)
inputs.simple_run = bool(inputs.simple_run)
inputs.mixed_precision = bool(inputs.mixed_precision)
inputs.tensorboard = bool(inputs.tensorboard)
inputs.eager = bool(inputs.eager)
inputs.model = inputs.model.split(".")
if len(inputs.model_type) == 0:
    inputs.model_type = inputs.model[0]
if inputs.max_epochs == -1:
    inputs.max_epochs = inputs.epochs
elif inputs.max_epochs < inputs.epochs:
    raise ValueError("epochs shouldn't be larger than max_epochs")

ctx.inputs = inputs

###############################################################################
# seting up GPUs
###############################################################################
import tensorflow as tf

if inputs.eager == False:
    tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices("GPU")
if ctx.inputs.gpus > 1:
    import horovod.tensorflow.keras as hvd

    hvd.init()
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

# importing keras at the end, I had some issues if I import it before setting GPUs
from tensorflow import keras

keras.backend.set_image_data_format("channels_last")

if ctx.inputs.gpus > 1:
    ctx.main_process = True if hvd.rank() == 0 else False
else:
    ctx.main_process = True

# Configure to use XLA compiler
USE_XLA = True
tf.config.optimizer.set_jit(USE_XLA)

###############################################################################
# Eager mode and logging
###############################################################################
DEBUG = False
tf.debugging.set_log_device_placement(DEBUG)
# tf.executing_eagerly()
# tf.config.experimental_run_functions_eagerly(True)
# this one should be equivalent to experimental_run_tf_function=False in model.compile, might work better

###############################################################################
# Determinism
###############################################################################

import os, random
import numpy as np

if ctx.inputs.seed >= 0:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    SEED = ctx.inputs.seed
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

###############################################################################
# seting hyperparameters
###############################################################################
import copy
import itertools
import sys
from rnn21cm import utilities
from rnn21cm import hyperparameters

if ctx.inputs.simple_run == True:
    HP_dict = hyperparameters.HP_simple()
else:
    HP = hyperparameters.HP()
    HP_list = list(itertools.product(*HP.values()))
    HP_dict = dict(zip(HP.keys(), HP_list[ctx.inputs.HyperparameterIndex]))

# correct learning rate for multigpu run
if ctx.inputs.LR_correction == True:
    HP_dict["LearningRate"] *= ctx.inputs.gpus

HP = utilities.AuxiliaryHyperparameters(
    model_name=f"{ctx.inputs.model[0]}_{ctx.inputs.model[1]}",
    Epochs=ctx.inputs.epochs,
    MaxEpochs=ctx.inputs.max_epochs,
    **HP_dict,
)

ctx.HP = HP

###############################################################################
# constructing data and filepaths
###############################################################################
Data = utilities.Data()

ctx.Data = Data

ctx.filepath = (
    f"{ctx.inputs.saving_location}{ctx.inputs.file_prefix}"
    f"{ctx.inputs.model[0]}_{ctx.inputs.model[1]}_"
    f"{ctx.HP.hash()}_{ctx.Data.hash()}"
)
ctx.logdir = (
    f"{ctx.inputs.logs_location}{ctx.inputs.file_prefix}"
    f"{ctx.inputs.model[0]}/{ctx.inputs.model[1]}/"
    f"{ctx.Data.hash()}/{ctx.HP.hash()}"
)

if ctx.inputs.seed >= 0:
    ctx.filepath += f"_seed_{ctx.inputs.seed}"
    ctx.logdir += f"_seed_{ctx.inputs.seed}"
###############################################################################
# building the model
###############################################################################
import importlib

ModelClassObject = getattr(
    importlib.import_module(f"rnn21cm.architectures.{ctx.inputs.model[0]}"),
    ctx.inputs.model[1],
)
ModelClass = ModelClassObject(ctx.Data.shape, HP)
ModelClass.build()

ctx.model = ModelClass.model

###############################################################################
# print status before training
###############################################################################
if ctx.main_process:
    print()
    print("TF_VERSION:", tf.__version__)
    print("INPUTS:", ctx.inputs)
    if ctx.inputs.gpus > 1:
        print("HVD.SIZE", hvd.size())
    print("HYPERPARAMETERS:", str(ctx.HP))
    print("DATA:", str(ctx.Data))
    print("FILEPATH:", ctx.filepath)
    print("MODEL:\n", ctx.model.summary())

###############################################################################
# fit the model
###############################################################################
utilities.fit()
