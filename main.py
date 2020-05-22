def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np

from data import X_imgs, Y_imgs
from models import DCGan, CycleGan

batch_size = 64
gen_input_shape = 100
epochs = 4000
verbose = 1
save_images = True
save_interval = 100

dc_gan = DCGan()
cycle_gan = CycleGan()

## Training Deep Convolutional GAN

for epoch in range(epochs+1):
    idx = np.random.randint(X_imgs.shape[0], size=batch_size)
    imgs = X_imgs[idx]
    noise = np.random.normal(0, 1, (batch_size, gen_input_shape))
    gen_imgs = dc_gan.gen_predict(noise)

    preds_real = dc_gan.disc_predict(imgs)
    preds_fake = dc_gan.disc_predict(gen_imgs)
    discriminator_loss = ((preds_real - 1)**2 + preds_fake**2).mean() / 2

    dc_gan.disc_train(imgs, np.ones(batch_size))
    dc_gan.disc_train(gen_imgs, np.zeros(batch_size))

    noise = np.random.normal(0, 1, (batch_size, gen_input_shape))
    gen_imgs = dc_gan.gen_predict(noise)

    preds_fake = dc_gan.disc_predict(gen_imgs)
    generator_loss = ((preds_fake - 1)**2).mean()

    dc_gan.gen_train(noise, np.ones(batch_size))

    if epoch % verbose == 0:
        print ("%d [D loss: %f] [G loss: %f]" % (epoch, discriminator_loss,
        generator_loss))

    if save_images:
        if epoch % save_interval == 0:
            dc_gan.save_imgs(imgs=gen_imgs, path="results/dc_gan/mnist_%d.png" % epoch)


## Training Cycle GAN

for epoch in range(epochs+1):
    idx_x = np.random.randint(X_imgs.shape[0], size=batch_size)
    imgs_x = X_imgs[idx_x]

    idx_y = np.random.randint(Y_imgs.shape[0], size=batch_size)
    imgs_y = Y_imgs[idx_y]

    gen_imgs_y = cycle_gan.gen_xy_predict(imgs_x)
    gen_imgs_x = cycle_gan.gen_yx_predict(imgs_y)


    disc_x_loss_real = cycle_gan.disc_x_train(imgs_x, np.ones(batch_size))[0]
    disc_x_loss_fake = cycle_gan.disc_x_train(gen_imgs_x, np.zeros(batch_size))[0]
    disc_x_loss = (disc_x_loss_real + disc_x_loss_fake) / 2

    disc_y_loss_real = cycle_gan.disc_y_train(imgs_y, np.ones(batch_size))[0]
    disc_y_loss_fake = cycle_gan.disc_y_train(gen_imgs_y, np.zeros(batch_size))[0]
    disc_y_loss = (disc_y_loss_real + disc_y_loss_fake) / 2

    gen_xy_loss = cycle_gan.gen_xy_train([imgs_x, imgs_y],[np.ones(batch_size), imgs_y,
    imgs_x, imgs_y])[0]
    gen_yx_loss = cycle_gan.gen_yx_train([imgs_y, imgs_x], [np.ones(batch_size), imgs_x,
    imgs_y, imgs_x])[0]

    print("%d [Dx loss: %f] [Dy loss: %f] [Gxy loss: %f] [Gyx loss: %f]" % (epoch,
    disc_x_loss, disc_y_loss, gen_xy_loss, gen_yx_loss))

    if save_images:
        if epoch % save_interval == 0:
            cycle_gan.save_imgs(imgs=imgs_x, path="results/cycle_gan/mnist_%d.png" %
            epoch)
            cycle_gan.save_imgs(imgs=gen_imgs_y, path="results/cycle_gan/usps_%d.png" %
            epoch)
