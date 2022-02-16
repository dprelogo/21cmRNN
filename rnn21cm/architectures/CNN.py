from tensorflow import keras


class basic3D:
    def __init__(
        self,
        InputShape,
        AuxiliaryHP,
    ):
        self.InputShape = InputShape
        # self.Data = Data
        self.AuxHP = AuxiliaryHP
        if self.AuxHP.ActivationFunction[0] == "selu":
            self.DropoutLayer = keras.layers.AlphaDropout
        else:
            self.DropoutLayer = keras.layers.Dropout

    def build(self):
        img_input = keras.layers.Input(shape=self.InputShape)
        x = keras.layers.Conv3D(128, (8, 8, 8), **self.AuxHP.ActivationFunction[1])(
            img_input
        )
        x = keras.layers.MaxPooling3D(pool_size=(2, 2, 4), strides=(2, 2, 4))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv3D(128, (4, 4, 1), **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv3D(128, (1, 1, 4), **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv3D(128, (3, 3, 1), **self.AuxHP.ActivationFunction[1])(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv3D(128, (1, 1, 4), **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv3D(128, (1, 1, 4), **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv3D(128, (1, 1, 4), **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Conv3D(256, (1, 1, 4), **self.AuxHP.ActivationFunction[1])(x)
        # x = keras.layers.MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)
        # if self.AuxHP.BatchNormalization == True:
        #     x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Flatten()(x)
        x = self.DropoutLayer(self.AuxHP.Dropout)(x)
        x = keras.layers.Dense(2**9, **self.AuxHP.ActivationFunction[1])(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(2**8, **self.AuxHP.ActivationFunction[1])(x)
        if self.AuxHP.BatchNormalization == True:
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Dense(2**6, **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.Dense(2**4, **self.AuxHP.ActivationFunction[1])(x)
        x = keras.layers.Dense(4)(x)

        self.model = keras.models.Model(
            inputs=img_input, outputs=x, name=self.AuxHP.model_name
        )
        # print(self.model.summary())
        return self.model
