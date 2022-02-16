from . import ctx

import os
import time
import copy
import hashlib
import numpy as np
import io
import tensorflow as tf
from tensorflow import keras

if ctx.inputs.gpus > 1:
    import horovod.tensorflow.keras as hvd


class TimeHistory(keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.file = open(self.filename, "a")

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.file.write(f"{time.time() - self.epoch_time_start}\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def on_train_end(self, logs={}):
        self.file.close()


class LR_tracer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        lr = keras.backend.eval(self.model.optimizer.lr)
        print(f"LR: {lr:.10f}")


class LR_scheduler:
    def __init__(self, total_epochs, inital_LR, multi_gpu_run=False, reduce_factor=0.1):
        self.total_epochs = total_epochs
        self.initial_LR = inital_LR
        self.multi_gpu_run = multi_gpu_run
        self.reduce_factor = reduce_factor

    def scheduler(self, epoch):
        """
        Returns learning rate at a given epoch.
        Recieves total number of epochs and initial learning rate
        """
        # print(f"IN LR_scheduler, initLR {self.initial_LR}, epoch {epoch}, frac. {(epoch + 1) / self.total_epochs}")
        if (epoch + 1) / self.total_epochs < 0.5:
            return self.initial_LR
        elif (epoch + 1) / self.total_epochs < 0.75:
            return self.initial_LR * self.reduce_factor
        else:
            return self.initial_LR * self.reduce_factor**2

    def callback(self):
        if self.multi_gpu_run == True:
            return hvd.callbacks.LearningRateScheduleCallback(self.scheduler)
        else:
            return tf.keras.callbacks.LearningRateScheduler(self.scheduler)


def define_callbacks():
    if ctx.inputs.gpus == 1:
        saving_callbacks = True
        horovod_callbacks = False
    else:
        saving_callbacks = True if hvd.rank() == 0 else False
        horovod_callbacks = True

    if saving_callbacks == True:
        saving_callbacks = [
            # hp.KerasCallback(logdir, HP_TensorBoard),
            TimeHistory(f"{ctx.filepath}_time.txt"),
            keras.callbacks.ModelCheckpoint(
                f"{ctx.filepath}_best.hdf5",
                monitor="val_loss",
                save_best_only=True,
                verbose=True,
            ),
            keras.callbacks.ModelCheckpoint(
                f"{ctx.filepath}_last.hdf5",
                monitor="val_loss",
                save_best_only=False,
                verbose=True,
            ),
            keras.callbacks.CSVLogger(
                f"{ctx.filepath}.log", separator=",", append=True
            ),
            # LR_tracer(),
        ]
        if ctx.inputs.tensorboard == True:
            saving_callbacks.append(
                keras.callbacks.TensorBoard(ctx.logdir, update_freq="epoch")
            )

    else:
        saving_callbacks = []
    if horovod_callbacks == True:
        horovod_callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ]
        if ctx.load_model == False:
            horovod_callbacks.append(
                hvd.callbacks.LearningRateWarmupCallback(
                    warmup_epochs=ctx.inputs.warmup
                )
            )
    else:
        horovod_callbacks = []

    important_callbacks = [
        keras.callbacks.TerminateOnNaN(),
    ]
    if ctx.HP.ReducingLR == True:
        scheduler = LR_scheduler(
            ctx.HP.MaxEpochs,
            ctx.HP.LearningRate,
            multi_gpu_run=(ctx.inputs.gpus > 1),
            reduce_factor=0.1,
        )
        important_callbacks.append(scheduler.callback())

    # not saving into ctx as it might screw up things during broadcast of variables
    return horovod_callbacks + saving_callbacks + important_callbacks
