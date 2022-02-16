###############################################################################
# define context, module with important variables
###############################################################################
from rnn21cm import ctx

ctx.init()
###############################################################################
# parsing inputs
###############################################################################
import argparse

parser = argparse.ArgumentParser(prog="Calculate Saliency Map for input")

# database
parser.add_argument(
    "--database", type=str, choices=ctx.database.keys(), default="default"
)
parser.add_argument(
    "--X_fstring", type=str, default="{}_realization_{:d}_{:03d}_of_{:03d}.tfrecord"
)
parser.add_argument("--normalize_X", type=int, choices=[0, 1], default=1)
# parser.add_argument('--N_realizations', type=int, default=10)

# model and hyperparameters
parser.add_argument("--model", type=str, default="RNN.SummarySpace3D")
parser.add_argument("--model_type", type=str, default="")
parser.add_argument("--HyperparameterIndex", type=int, default=0)
parser.add_argument("--simple_run", type=int, choices=[0, 1], default=0)

# training setup
# parser.add_argument('--seed', type = int, default = -1)
# parser.add_argument('--warmup', type=int, default=0)
parser.add_argument("--LR_correction", type=int, choices=[0, 1], default=1)
# parser.add_argument('--mixed_precision', type = int, choices = [0, 1], default = 0)
# parser.add_argument('--gpus', type=int, default=1)
# parser.add_argument('--workers', type=int, default=24)
# parser.add_argument('--eager', type=int, choices=[0, 1], default=1)

parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--max_epochs", type=int, default=-1)

# saving and output
# parser.add_argument('--data_location', type=str, default="data/")
parser.add_argument("--saving_location", type=str, default="./")
# parser.add_argument('--tensorboard', type=int, choices=[0, 1], default=1)
# parser.add_argument('--logs_location', type=str, default="logs/")
parser.add_argument("--file_prefix", type=str, default="")
# parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=2)

# input to compute saliency on
parser.add_argument("--image", type=str)
parser.add_argument("--params", type=str)


inputs = parser.parse_args()
ctx.formatting = ctx.database[inputs.database]
inputs.normalize_X = bool(inputs.normalize_X)
inputs.LR_correction = bool(inputs.LR_correction)
inputs.simple_run = bool(inputs.simple_run)
# inputs.mixed_precision = bool(inputs.mixed_precision)
# inputs.tensorboard = bool(inputs.tensorboard)
# inputs.eager = bool(inputs.eager)
inputs.model = inputs.model.split(".")
if len(inputs.model_type) == 0:
    inputs.model_type = inputs.model[0]
if inputs.max_epochs == -1:
    inputs.max_epochs = inputs.epochs

ctx.inputs = inputs

###############################################################################
# seting up GPUs
###############################################################################
import tensorflow as tf

# importing keras at the end, I had some issues if I import it before setting GPUs
from tensorflow import keras

keras.backend.set_image_data_format("channels_last")

ctx.main_process = True

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
    print("HYPERPARAMETERS:", str(ctx.HP))
    print("DATA:", str(ctx.Data))
    print("FILEPATH:", ctx.filepath)
    print("MODEL:\n", ctx.model.summary())

###############################################################################
# saliency calculation
###############################################################################
import numpy as np

image, true = np.load(ctx.inputs.image), np.load(ctx.inputs.params)
image = tf.convert_to_tensor(image, dtype=tf.float32)
true = tf.convert_to_tensor(true, dtype=tf.float32)

with tf.GradientTape as tape:
    pred = ctx.model(image, train=False)
    loss = tf.losses.mse(pred, true)
    per_param_loss = (pred - true) ** 2

grads = tape.gradient(loss, image)
per_param_grads = tape.gradient(per_param_loss, image)
