from tensorflow import keras


def leakyrelu(x):
    return keras.activations.relu(x, alpha=0.1)


HP = dict(
    Loss=[[None, "mse"]],
    BatchSize=[20],
    BatchNormalization=[True, False],
    LearningRate=[1e-3, 1e-4, 1e-5],
    Dropout=[0.2],
    ReducingLR=[True],
    Optimizer=[
        [keras.optimizers.Adam, "Adam", {}],
        [keras.optimizers.Nadam, "Nadam", {}],
    ],
    ActivationFunction=[
        [
            "relu",
            {
                "activation": keras.activations.relu,
                "kernel_initializer": keras.initializers.he_uniform(),
            },
        ],
        [
            "leakyrelu",
            {
                "activation": leakyrelu,
                "kernel_initializer": keras.initializers.he_uniform(),
            },
        ],
    ],
)

HP_large = dict(
    Loss=[[None, "mse"]],
    BatchSize=[20, 100],
    BatchNormalization=[True, False],
    LearningRate=[1e-2, 1e-3, 1e-4, 1e-5],
    Dropout=[0.2, 0.5],
    ReducingLR=[True],
    Optimizer=[
        [keras.optimizers.RMSprop, "RMSprop", {}],
        [keras.optimizers.SGD, "SGD", {}],
        [keras.optimizers.SGD, "Momentum", {"momentum": 0.9, "nesterov": True}],
        [keras.optimizers.Adam, "Adam", {}],
        [keras.optimizers.Adamax, "Adamax", {}],
        [keras.optimizers.Nadam, "Nadam", {}],
    ],
    ActivationFunction=[
        [
            "relu",
            {
                "activation": keras.activations.relu,
                "kernel_initializer": keras.initializers.he_uniform(),
            },
        ],
        [
            "leakyrelu",
            {
                "activation": leakyrelu,
                "kernel_initializer": keras.initializers.he_uniform(),
            },
        ],
        [
            "elu",
            {
                "activation": keras.activations.elu,
                "kernel_initializer": keras.initializers.he_uniform(),
            },
        ],
        [
            "selu",
            {
                "activation": keras.activations.selu,
                "kernel_initializer": keras.initializers.lecun_normal(),
            },
        ],
    ],
)

HP_simple = dict(
    Loss=[None, "mse"],
    BatchSize=20,
    BatchNormalization=True,
    LearningRate=1e-3,
    Dropout=0.2,
    ReducingLR=True,
    Optimizer=[keras.optimizers.Adam, "Adam", {}],
    ActivationFunction=[
        "relu",
        {
            "activation": keras.activations.relu,
            "kernel_initializer": keras.initializers.he_uniform(),
        },
    ],
)
