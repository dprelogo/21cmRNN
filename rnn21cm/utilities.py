from . import ctx
from . import callbacks

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


class Data:
    def __init__(
        self,
        removed_average=True,
        Zmax=30,
        base_shape=(25, 25, 526),
        X_std=18.130475997924805,
        Y_mean=[131.08347, 4.9945736, 39.99807, 799.07794],
        Y_std=[69.06536, 0.57125527, 1.1579071, 403.83325],
    ):
        self.removed_average = removed_average
        self.Zmax = Zmax
        if ctx.inputs.model_type == "RNN":
            self.shape = base_shape[::-1] + (1,)
        else:
            self.shape = base_shape + (1,)
        self.X_std = X_std
        self.Y_mean = np.array(Y_mean, dtype=np.float32)[np.newaxis, :]
        self.Y_std = np.array(Y_std, dtype=np.float32)[np.newaxis, :]
        self.load()

    def __str__(self):
        S = (
            f"removed_average:{self.removed_average}__"
            f"normalized:{ctx.inputs.normalize_X}__"
            f"Zmax:{self.Zmax}"
        )
        S += "".join([f"__{i}" for i in ctx.formatting])
        S += f"__realizations_{ctx.inputs.N_realizations}"
        return S

    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()

    def Example(self, serialized_example):
        return tf.io.parse_single_example(
            serialized_example,
            features={
                "Xx": tf.io.FixedLenFeature([], tf.int64),
                "Xy": tf.io.FixedLenFeature([], tf.int64),
                "Xz": tf.io.FixedLenFeature([], tf.int64),
                "Xstd": tf.io.FixedLenFeature([], tf.float32),
                "X": tf.io.FixedLenFeature([], tf.string),
                "Yx": tf.io.FixedLenFeature([], tf.int64),
                "Ymean": tf.io.FixedLenFeature([], tf.string),
                "Ystd": tf.io.FixedLenFeature([], tf.string),
                "Y": tf.io.FixedLenFeature([], tf.string),
            },
        )

    def decode(self, serialized_example):
        """Parses an image and label from the given `serialized_example`."""
        example = self.Example(serialized_example)
        xx = tf.cast(example["Xx"], tf.int64)
        xy = tf.cast(example["Xy"], tf.int64)
        xz = tf.cast(example["Xz"], tf.int64)
        x_std = tf.cast(example["Xstd"], tf.float32)
        x = tf.io.decode_raw(example["X"], tf.float32)
        x = tf.reshape(x, (xx, xy, xz))
        if ctx.inputs.model_type == "RNN":
            x = tf.transpose(x)
        x = tf.expand_dims(x, -1)
        if ctx.inputs.normalize_X == True:
            x /= x_std

        yx = tf.cast(example["Yx"], tf.int64)
        y_mean = tf.io.decode_raw(example["Ymean"], tf.float32)
        y_mean = tf.reshape(y_mean, (yx,))
        y_std = tf.io.decode_raw(example["Ystd"], tf.float32)
        y_std = tf.reshape(y_std, (yx,))
        y = tf.io.decode_raw(example["Y"], tf.float32)
        y = tf.reshape(y, (yx,))
        y = (y - y_mean) / y_std
        return x, y

    def create_shards(self, filenames, ds_type):
        loading_sequence = np.arange(len(filenames))
        if ctx.inputs.gpus > 1:
            loading_sequence = np.roll(loading_sequence, -hvd.rank())

        shards = tf.data.Dataset.from_tensor_slices(filenames[loading_sequence[0]])
        if ds_type == "train" or ds_type == "validation":
            shards = shards.shuffle(len(filenames[loading_sequence[0]]))
        for i in loading_sequence[1:]:
            t_shards = tf.data.Dataset.from_tensor_slices(filenames[i])
            if ds_type == "train" or ds_type == "validation":
                t_shards = t_shards.shuffle(len(filenames[i]))
            shards = shards.concatenate(t_shards)
        # in the case of test database repeat only once, else repeat indefinitely
        if ds_type == "test":
            shards = shards.repeat(1)
        else:
            shards = shards.repeat()
        return shards

    def get_dataset(self, ds_type, filenames, buffer_size):
        """Read TFRecords files and turn them into a TFRecordDataset."""
        shards = self.create_shards(filenames, ds_type)
        if ds_type == "train" or ds_type == "validation":
            dataset = shards.interleave(
                tf.data.TFRecordDataset, cycle_length=5, block_length=1
            )
        else:
            dataset = shards.interleave(
                tf.data.TFRecordDataset, cycle_length=1, block_length=1
            )
        #   dataset = dataset.shuffle(buffer_size=8192)
        dataset = dataset.map(
            map_func=lambda x: self.decode(x), num_parallel_calls=ctx.inputs.workers
        ).batch(batch_size=ctx.HP.BatchSize)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        return dataset

    def load(self):
        # creating .tfrecord filenames
        shardsTVT = {"train": 320, "validation": 40, "test": 40}
        self.filenames = {"train": [], "validation": [], "test": []}
        for key in self.filenames.keys():
            for seed in range(ctx.inputs.N_realizations):
                self.filenames[key].append(
                    [
                        (
                            f"{ctx.inputs.data_location}"
                            f"{ctx.inputs.X_fstring.format(key, seed, i, shardsTVT[key]-1)}"
                        )
                        for i in range(shardsTVT[key])
                    ]
                )

        # creating all datasets
        self.train_ds = self.get_dataset(
            "train",
            self.filenames["train"],
            buffer_size=16,
        )
        self.validation_ds = self.get_dataset(
            "validation",
            [self.filenames["validation"][0]],
            buffer_size=16,
        )
        self.test_ds = self.get_dataset(
            "test",
            self.filenames["test"],
            buffer_size=16,
        )

        self.steps_per_epoch = (
            shardsTVT["train"] * 100 // ctx.HP.BatchSize // ctx.inputs.gpus
        )
        self.validation_steps = (
            shardsTVT["validation"] * 100 // ctx.HP.BatchSize // ctx.inputs.gpus
        )


class AuxiliaryHyperparameters:
    def __init__(
        self,
        model_name,
        Epochs=200,
        MaxEpochs=200,
        Loss=[keras.losses.mean_squared_error, "mse"],
        Optimizer=[keras.optimizers.RMSprop, "RMSprop", {}],
        LearningRate=0.01,
        ActivationFunction=[
            "relu",
            {
                "activation": keras.activations.relu,
                "kernel_initializer": keras.initializers.he_uniform(),
            },
        ],
        BatchNormalization=True,
        Dropout=0.2,
        ReducingLR=False,
        BatchSize=20,
    ):
        self.model_name = model_name
        self.Loss = Loss
        self.Optimizer = Optimizer
        self.LearningRate = LearningRate
        self.Optimizer[2]["lr"] = self.LearningRate
        self.ActivationFunction = ActivationFunction
        self.BatchNormalization = BatchNormalization
        self.Dropout = Dropout
        self.ReducingLR = ReducingLR
        self.BatchSize = BatchSize
        self.Epochs = Epochs
        self.MaxEpochs = MaxEpochs
        self.TensorBoard = {
            "Model": self.model_name,
            "LearningRate": self.LearningRate,
            "Dropout": self.Dropout,
            "BatchSize": self.BatchSize,
            "BatchNormalization": self.BatchNormalization,
            "Optimizer": self.Optimizer[1],
            "ActivationFunction": self.ActivationFunction[0],
        }

    def __str__(self):
        return (
            f"Loss:{self.Loss[1]}__"
            f"Optimizer:{self.Optimizer[1]}__"
            f"LR:{self.LearningRate:.10f}__"
            f"Activation:{self.ActivationFunction[0]}__"
            f"BN:{self.BatchNormalization}__"
            f"dropout:{self.Dropout:.2f}__"
            f"reduceLR:{self.ReducingLR}__"
            f"Batch:{self.BatchSize:05d}__"
            f"Epochs:{self.MaxEpochs:05d}"
        )

    def hash(self):
        return hashlib.md5(self.__str__().encode()).hexdigest()


def R2(y_true, y_pred):
    SS_res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    SS_tot = keras.backend.sum(
        keras.backend.square(y_true - keras.backend.mean(y_true, axis=0))
    )
    return 1 - SS_res / (SS_tot + keras.backend.epsilon())


def R2_numpy(y_true, y_pred):
    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - SS_res / (SS_tot + 1e-7)


def R2_final(y_true, y_pred):
    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1 - SS_res / (SS_tot + 1e-7)


def define_model(restore_training):
    model_exists = os.path.exists(f"{ctx.filepath}_last.hdf5")
    # define in what case to load the model
    if model_exists == True and restore_training == True:
        load_model = True
        if ctx.inputs.gpus == 1:
            load_function = keras.models.load_model
        else:
            load_function = hvd.load_model
    else:
        load_model = False
    ctx.load_model = load_model

    # load the model
    if load_model == True:
        custom_obj = {}
        custom_obj["R2"] = R2
        if ctx.HP.ActivationFunction[0] == "leakyrelu":
            custom_obj[ctx.HP.ActivationFunction[0]] = ctx.HP.ActivationFunction[1][
                "activation"
            ]
        # if loading last model fails for some reason, load the best one
        try:
            ctx.model = load_function(
                f"{ctx.filepath}_last.hdf5", custom_objects=custom_obj
            )
        except:
            ctx.model = load_function(
                f"{ctx.filepath}_best.hdf5", custom_objects=custom_obj
            )

        with open(f"{ctx.filepath}.log") as f:
            number_of_epochs_trained = (
                len(f.readlines()) - 1
            )  # the first line is description
            if ctx.main_process:
                print("NUMBER_OF_EPOCHS_TRAINED", number_of_epochs_trained)
        if ctx.HP.Epochs + number_of_epochs_trained > ctx.HP.MaxEpochs:
            final_epochs = ctx.HP.MaxEpochs
        else:
            final_epochs = ctx.HP.Epochs + number_of_epochs_trained

        ctx.fit_options = {
            "epochs": final_epochs,
            "initial_epoch": number_of_epochs_trained,
            "steps_per_epoch": ctx.Data.steps_per_epoch,
            "validation_steps": ctx.Data.validation_steps,
        }
        ctx.compile_options = {}
    else:
        ctx.fit_options = {
            "epochs": ctx.HP.Epochs,
            "initial_epoch": 0,
            "steps_per_epoch": ctx.Data.steps_per_epoch,
            "validation_steps": ctx.Data.validation_steps,
        }
        ctx.compile_options = {
            "loss": ctx.HP.Loss[1],
            "optimizer": ctx.HP.Optimizer[0](**ctx.HP.Optimizer[2]),
            "metrics": [R2],
        }
        if ctx.inputs.mixed_precision == True:
            ctx.compile_options[
                "optimizer"
            ] = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                ctx.compile_options["optimizer"]
            )
        if ctx.inputs.gpus > 1:
            ctx.compile_options["optimizer"] = hvd.DistributedOptimizer(
                ctx.compile_options["optimizer"]
            )


def fit(restore_training=True):
    verbose = ctx.inputs.verbose if ctx.main_process == True else 0

    # build callbacks and model
    cb = callbacks.define_callbacks()
    define_model(restore_training)
    if ctx.main_process:
        print("COMPILE AND FIT OPTIONS:")
        print(ctx.compile_options)
        print(ctx.fit_options)
    if len(ctx.compile_options) > 0:
        ctx.model.compile(**ctx.compile_options, experimental_run_tf_function=False)

    ctx.model.fit(
        ctx.Data.train_ds,
        validation_data=ctx.Data.validation_ds,
        verbose=verbose,
        callbacks=cb,
        **ctx.fit_options,
    )


def predict(Type="last"):
    """Run full predict over test set.
    Args:
        Type: "best" or "last", used as a saving flag and for logs
    """
    custom_obj = {}
    custom_obj["R2"] = R2
    # if activation is leakyrelu add to custom_obj
    if ctx.HP.ActivationFunction[0] == "leakyrelu":
        custom_obj[ctx.HP.ActivationFunction[0]] = ctx.HP.ActivationFunction[1][
            "activation"
        ]
    ctx.model = keras.models.load_model(
        f"{ctx.filepath}_{Type}.hdf5", custom_objects=custom_obj
    )
    print(f"PREDICTING THE MODEL {Type}")

    # assumes eager execution
    true = []
    for x, y in ctx.Data.test_ds:
        true.append(y.numpy())
    true = np.concatenate(true, axis=0)

    pred = ctx.model.predict(
        ctx.Data.test_ds,
        verbose=False,
        workers=1,
        use_multiprocessing=False,
    )

    np.save(f"{ctx.filepath}_prediction_{Type}.npy", pred)
    np.save(f"{ctx.filepath}_true_{Type}.npy", true)

    R2_score = []
    R2_score.append(R2_final(true, pred))
    for i in range(4):
        R2_score.append(R2_numpy(true[:, i], pred[:, i]))

    if Type == "best":
        with open(f"{ctx.filepath}_summary.txt", "w") as f:
            f.write(f"DATA: {str(ctx.Data)}\n")
            f.write(f"HYPARAMETERS: {str(ctx.HP)}\n")
            f.write(f"R2_total: {R2_score[0]}\n")
            for i in range(4):
                print(f"R2: {R2_score[i+1]}")
                f.write(f"R2_{i}: {R2_score[i+1]}\n")
            f.write("\n")
            stringlist = []
            ctx.model.summary(print_fn=lambda x: stringlist.append(x))
            f.write("\n".join(stringlist))

    return true, pred, R2_score


def average_gradient_saliency(
    model,
    image,
    model_input_shape=(25, 25, 526),
    model_output_shape=(4,),
):
    """Computing average of the gradient saliency map.
    The function takes all possible slices of the input image and computes
    gradients for each one. Assumes CNN model. Iterations are taken only over
    first two (spatial) dimensions.

    Args:
        model: `keras.Model`
        image: input image, usually larger dimensions than the model input

    Returns:
        grads: averaged gradients
    """
    image_shape = image.shape
    if len(model_input_shape) != len(image_shape):
        raise AttributeError(
            "Model input and `image` should be of same dimensionality."
        )
    if (
        model_input_shape[0] > image_shape[0]
        or model_input_shape[1] > image_shape[1]
        or model_input_shape[-1] != image_shape[-1]
    ):
        raise AttributeError(
            "`image` should be larger or equal than `model_input_shape` "
            "in firs two, spatial dimensions and equal in last, redshift dimension."
        )
    per_param_grads = np.zeros(model_output_shape + model_input_shape, dtype=np.float64)

    for i in range(image_shape[0]):
        # print(i, end=" ")
        for j in range(image_shape[1]):
            im = np.roll(np.roll(image, i, axis=0), j, axis=1)[
                : model_input_shape[0], : model_input_shape[1], :
            ]
            # first axis in gradients are parameters
            per_param_grads = np.roll(np.roll(per_param_grads, i, axis=1), j, axis=2)
            ppg = _calculate_saliency_gradients(model, im)
            per_param_grads[: model_input_shape[0], : model_input_shape[1], :] += ppg
            per_param_grads = np.roll(np.roll(per_param_grads, -i, axis=1), -j, axis=2)

    return per_param_grads


def _calculate_saliency_gradients(model, image):
    """One iteration of the saliency computation."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, -1)
    image = tf.expand_dims(image, 0)

    with tf.GradientTape() as tape:
        tape.watch(image)
        pred_params = model(image, training=False)

    per_param_grads = tape.jacobian(pred_params, image)
    per_param_grads = np.squeeze(per_param_grads.numpy())

    return per_param_grads
