import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

class DCGan:
    def __init__(self):
        self.disc = DCDiscriminator().disc
        self.gen = DCGenerator().gen
        self.gan = self.combine_models()

    def combine_models(self):
        gen_input = self.gen.input
        gen_output = self.gen.output
        disc_output = self.disc(gen_output)
        model_combined = Model(gen_input, disc_output)

        self.disc.trainable = False

        model_combined.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.0002, 0.5),
                  metrics=['accuracy'])

        return model_combined

    def gen_predict(self, x):
        return self.gen.predict(x)

    def disc_predict(self, x):
        return self.disc.predict(x)

    def disc_train(self, x, y):
        self.disc.train_on_batch(x, y)

    def gen_train(self, x, y):
        self.gan.train_on_batch(x, y)

    def save_imgs(self, imgs=None, path="results/gen_mnist.png"):
        r, c = 4, 4
        if imgs is None:
            noise = np.random.normal(0, 1, (r * c, self.gen.gen_input_shape))
            imgs = self.gen_predict(noise)
        else:
            imgs = imgs[:r * c]

        # Rescale images 0 - 1
        imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs[cnt].reshape(28, 28), cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(path)
        plt.close()


class Discriminator:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.chanels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.chanels)

    def get_model(self):
        model = Sequential()

        model.add(Conv2D(28, kernel_size=5, strides=2, padding='same',
                input_shape=self.img_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(56, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(112, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(1, kernel_size=5, strides=4, padding='same'))
        model.add(Activation('sigmoid'))
        model.add(Reshape((1,)))

        model.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.0002, 0.5),
                  metrics=['accuracy'])

        return model

class DCDiscriminator(Discriminator):
    def __init__(self):
        super().__init__()
        self.disc = self.get_model()


class DCGenerator:
    def __init__(self):
        self.gen_input_shape = 100
        self.channels = 1

        self.gen = self.get_model()

    def get_model(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation='relu',
        input_shape=(self.gen_input_shape,)))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=5, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        return model


class CycleGan:
    def __init__(self):
        self.cycle_disc = CycleDiscriminator()
        self.disc_x = self.cycle_disc.disc_x
        self.disc_y = self.cycle_disc.disc_y

        self.cycle_gen = CycleGenerator()
        self.gen_xy = self.cycle_gen.gen_xy
        self.gen_yx = self.cycle_gen.gen_yx

        self.gan_xy, self.gan_yx = self.combine_models()

    def define_composite_model(self, gen_1, disc, gen_2):
        gen_1.trainable = True
        disc.trainable = False
        gen_2.trainable = False

        gen_1_input = Input(shape=self.cycle_gen.img_shape)
        gen_1_output = gen_1(gen_1_input)
        disc_output = disc(gen_1_output)

        # identity element
        identity_input = Input(shape=self.cycle_gen.img_shape)
        identity_output = gen_1(identity_input)

        # forward cycle
        forward_output = gen_2(gen_1_output)

        # backward cycle
        gen_2_output = gen_2(identity_input)
        backward_output = gen_1(gen_2_output)

        model = Model([gen_1_input, identity_input], [disc_output, identity_output,
        forward_output, backward_output])

        model.compile(loss=['mse', 'mae', 'mae', 'mae'],
                    optimizer=Adam(0.0002, 0.5),
                    metrics=['accuracy'])

        return model

    def combine_models(self):
        model_combined_xy = self.define_composite_model(self.gen_xy, self.disc_y,
        self.gen_yx)
        model_combined_yx = self.define_composite_model(self.gen_yx, self.disc_x,
        self.gen_xy)
        return model_combined_xy, model_combined_yx

    def gen_xy_predict(self, x):
        return self.gen_xy.predict(x)

    def gen_yx_predict(self, x):
        return self.gen_yx.predict(x)

    def disc_x_predict(self, x):
        return self.disc_x.predict(x)

    def disc_y_predict(self, x):
        return self.disc_y.predict(x)

    def disc_x_train(self, x, y):
        return self.disc_x.train_on_batch(x, y)

    def disc_y_train(self, x, y):
        return self.disc_y.train_on_batch(x, y)

    def gen_xy_train(self, x, y):
        return self.gan_xy.train_on_batch(x, y)

    def gen_yx_train(self, x, y):
        return self.gan_yx.train_on_batch(x, y)

    def save_imgs(self, imgs, path="results/gen_usps.png"):
        r, c = 4, 4
        imgs = imgs[:r * c]

        # Rescale images 0 - 1
        imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(imgs[cnt].reshape(28, 28), cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(path)
        plt.close()


class CycleDiscriminator(Discriminator):
    def __init__(self):
        super().__init__()
        self.disc_x = self.get_model()
        self.disc_y = self.get_model()


class CycleGenerator:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.chanels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.chanels)

        self.gen_xy = self.get_model()
        self.gen_yx = self.get_model()

    def get_model(self):

        def conv2d(input_layer, filters, kernel_size, strides, padding):
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding)(input_layer)
            conv = BatchNormalization()(conv)
            conv = LeakyReLU()(conv)
            return conv

        def residual_block(input_layer, filters, kernel_size):
            res = Conv2D(filters=filters, kernel_size=kernel_size,
            padding='same')(input_layer)
            res = BatchNormalization()(res)
            res = Activation('relu')(res)

            res = Conv2D(filters=filters, kernel_size=kernel_size,
            padding='same')(res)
            res = BatchNormalization()(res)

            res = Concatenate()([res, input_layer])
            return res

        def deconv2d(input_layer, filters, kernel_size):
            deconv = UpSampling2D()(input_layer)
            deconv = Conv2D(filters=filters, kernel_size=kernel_size,
            padding='same')(deconv)
            deconv = BatchNormalization()(deconv)
            return deconv

        input_img = Input(shape=self.img_shape)

        # first stage : convolutional layers to encode the input
        x = conv2d(input_img, 28, 5, 2, 'same')
        x = conv2d(x, 56, 5, 2, 'same')

        # second stage : residual block to transform the features
        x = residual_block(x, 56, 5)

        # third stage : transpose convolutional layers to decode the transformed
        # features
        x = deconv2d(x, 28, 5)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = deconv2d(x, 1, 5)
        output_img = Activation('tanh')(x)

        model = Model(input_img, output_img)

        return model
