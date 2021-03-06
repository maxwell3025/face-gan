import tensorflow as tf
from tensorflow.keras import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image

from tensorflow.python.ops.gen_math_ops import equal

BATCH_SIZE = 8

face_data = tf.data.Dataset.from_tensor_slices(np.load("data/data.npy")).shuffle(10000).batch(BATCH_SIZE)

def create_generator():
    model = Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), use_bias=False, strides=(1,1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), use_bias=False, strides=(1,1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), use_bias=False, strides=(1,1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), use_bias=False, strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), use_bias=False, strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), use_bias=False, strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), use_bias=False, strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    return model

def create_discriminator():
    model = Sequential()
    model.add(layers.Conv2D(16, (5, 5), strides=(2,2), padding='same', input_shape=(128,128,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(32, (5, 5), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, (5, 5), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    return model

generator = create_generator()
discriminator = create_discriminator()

cross_entropy = losses.BinaryCrossentropy(from_logits=True)

def get_gen_loss(fake):
    return cross_entropy(tf.ones_like(fake), fake)

def get_dis_loss(real, fake):
    return cross_entropy(tf.zeros_like(fake), fake) + cross_entropy(tf.ones_like(real), real)

gen_optimizer = optimizers.Adam(1e-4)
dis_optimizer = optimizers.Adam(1e-4)

@tf.function
def train_step(batch):
    noise = tf.random.normal((BATCH_SIZE, 100))
    with tf.GradientTape() as gentape, tf.GradientTape() as distape:
        fake_batch = generator(noise, training = True)

        real_out = discriminator(batch, training = True)
        fake_out = discriminator(fake_batch, training = True)

        gen_loss = get_gen_loss(fake_out)
        dis_loss = get_dis_loss(real_out, fake_out)
    gen_grad = gentape.gradient(gen_loss, generator.trainable_variables)
    dis_grad = distape.gradient(dis_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(dis_grad, discriminator.trainable_variables))

sample_noise = tf.random.normal((16,100))
def train(epochs, savelocation):
    os.makedirs("out/{}".format(savelocation), exist_ok=True)
    for epoch in range (epochs):
        start = time.time()
        for batch in face_data:
            train_step(batch)
        #save sample
        sample = generator(sample_noise)
        patch = Image.new("RGB", (512, 512))
        for x in range(4):
            for y in range(4):
                im = Image.fromarray((sample[x+y*4].numpy()*128+128).astype("uint8"), "RGB")
                patch.paste(im=im, box = (x*128, y*128))
        patch.save("out/{}/{:04d}.png".format(savelocation, epoch))
        print("finished epoch {} in time {}".format(epoch+1, time.time() - start))


    





















