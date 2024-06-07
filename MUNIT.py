import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import datetime
from utils import *
from ops import *
from glob import glob
import time
import os
from tensorflow.keras import layers

'''
Created by Ebraham Alskaf June 2024
MUNIT for style transfer using Tensorflow 2._
'''

# Define the generator and discriminator models
class VAEGen(tf.keras.Model):
    def __init__(self, input_dim, z_dim=8):
        super(VAEGen, self).__init__()
        self.encoder = self.build_encoder(input_dim, z_dim)
        self.decoder = self.build_decoder(z_dim)

    def build_encoder(self, input_dim, z_dim):
        inputs = layers.Input(shape=(input_dim, input_dim, 3))
        x = layers.Conv2D(64, kernel_size=7, strides=1, padding='same')(inputs)
        x = layers.ReLU()(x)
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        mean = layers.Dense(z_dim)(x)
        log_var = layers.Dense(z_dim)(x)
        return tf.keras.Model(inputs, [mean, log_var], name='encoder')

    def build_decoder(self, z_dim):
        inputs = layers.Input(shape=(z_dim,))
        x = layers.Dense(64 * 64 * 256)(inputs)
        x = layers.Reshape((64, 64, 256))(x)
        x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh')(x)
        return tf.keras.Model(inputs, x, name='decoder')

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_var * 0.5) + mean

    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z, _, _ = self.encode(x)
        return self.decode(z)

class DiscriminatorBlock(layers.Layer):
    def __init__(self, filters, use_norm=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same')
        self.norm = tfa.layers.InstanceNormalization() if use_norm else None
        self.act = layers.LeakyReLU(0.2)

    def call(self, inputs):
        x = self.conv(inputs)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        return x

class MsImageDis(tf.keras.Model):
    def __init__(self, input_dim, num_scales=3, num_filters=64):
        super(MsImageDis, self).__init__()
        self.num_scales = num_scales
        self.discriminators = [self.build_discriminator(input_dim, num_filters) for _ in range(num_scales)]

    def build_discriminator(self, input_dim, num_filters):
        inputs = layers.Input(shape=(input_dim, input_dim, 3))
        x = DiscriminatorBlock(num_filters, use_norm=False)(inputs)
        x = DiscriminatorBlock(num_filters * 2)(x)
        x = DiscriminatorBlock(num_filters * 4)(x)
        x = DiscriminatorBlock(num_filters * 8)(x)
        x = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
        return tf.keras.Model(inputs, x, name='discriminator')

    def call(self, x):
        outputs = []
        for discriminator in self.discriminators:
            x = tf.image.resize(x, [x.shape[1], x.shape[2]])
            outputs.append(discriminator(x))
        return outputs

# Register custom objects
tf.keras.utils.get_custom_objects().update({'VAEGen': VAEGen, 'MsImageDis': MsImageDis, 'DiscriminatorBlock': DiscriminatorBlock})

# Loading data
batch_size = 8

trainA_dataset = glob('CONTENT_A/*.png') #get_content_images('/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/Train_Style_images')
trainB_dataset = glob('STYLE_IMG_GRAY_A/*.png')
dataset_num = max(len(trainA_dataset), len(trainB_dataset))

iteration = len(trainA_dataset) // batch_size

print(f"Number of content images: {len(trainA_dataset)}")
print(f"Number of style images: {len(trainB_dataset)}")

trainA = tf.data.Dataset.from_tensor_slices(trainA_dataset)
trainB = tf.data.Dataset.from_tensor_slices(trainB_dataset)

trainA = (trainA.shuffle(len(trainA_dataset)).map(load_and_preprocess_image).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
trainB = (trainB.shuffle(len(trainB_dataset)).map(load_and_preprocess_image).repeat().batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
train_dataset = tf.data.Dataset.zip((trainA, trainB))

# Tensorboard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_writer = tf.summary.create_file_writer(log_dir + "/images")

# Training step function
@tf.function
def train_step(generator, discriminator, images_a, images_b, gen_optimizer, dis_optimizer, loss_fn, step, epoch):
    with tf.GradientTape(persistent=True) as tape:
        z_a, mean_a, log_var_a = generator.encode(images_a)
        z_b, mean_b, log_var_b = generator.encode(images_b)
        x_ab = generator.decode(z_a)
        x_ba = generator.decode(z_b)

        loss_gen_ab = loss_fn(images_b, x_ab)
        loss_gen_ba = loss_fn(images_a, x_ba)
        loss_gen_total = loss_gen_ab + loss_gen_ba

        real_a, real_b = discriminator(images_a), discriminator(images_b)
        fake_a, fake_b = discriminator(x_ba), discriminator(x_ab)

        loss_dis_a = sum([tf.reduce_mean(tf.square(real_a[i] - 1.0)) + tf.reduce_mean(tf.square(fake_a[i])) for i in range(len(real_a))])
        loss_dis_b = sum([tf.reduce_mean(tf.square(real_b[i] - 1.0)) + tf.reduce_mean(tf.square(fake_b[i])) for i in range(len(real_b))])
        loss_dis_total = loss_dis_a + loss_dis_b

    gradients_gen = tape.gradient(loss_gen_total, generator.trainable_variables)
    gradients_dis = tape.gradient(loss_dis_total, discriminator.trainable_variables)

    # Clip gradients to avoid exploding gradients
    gradients_gen = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_gen]
    gradients_dis = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_dis]

    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(gradients_dis, discriminator.trainable_variables))

    tf.print("Step:", step + 1, "Gen Loss:", loss_gen_total, "Disc Loss:", loss_dis_total)

    return loss_gen_total, loss_dis_total

# Training function
def train(generator, discriminator, gen_optimizer, dis_optimizer, epochs, loss_fn, checkpoint_dir):
    for epoch in range(epochs):
        avg_loss_gen = tf.keras.metrics.Mean()
        avg_loss_dis = tf.keras.metrics.Mean()
        for step, (images_a, images_b) in enumerate(train_dataset.take(iteration)):
            loss_gen, loss_dis = train_step(generator, discriminator, images_a, images_b, gen_optimizer, dis_optimizer, loss_fn, step, epoch)
            avg_loss_gen.update_state(loss_gen)
            avg_loss_dis.update_state(loss_dis)

        print(f'Epoch {epoch + 1}, Avg Generator Loss: {avg_loss_gen.result().numpy()}, Avg Discriminator Loss: {avg_loss_dis.result().numpy()}')

        # Log avaeraged losses
        with train_writer.as_default():
            tf.summary.scalar('Avg Generator Loss', avg_loss_gen.result(), step=epoch)
            tf.summary.scalar('Avg Discriminator Loss', avg_loss_dis.result(), step=epoch)

        # Log images at the end of each epoch
        with train_writer.as_default():
            def normalize_for_tensorboard(image):
                image = (image + 1.0) / 2.0
                image = tf.clip_by_value(image, 0.0, 1.0)
                image = image * 255.0
                return tf.cast(image, tf.uint8)

            # Pick random images from the dataset for logging
            sample_images_a = next(iter(trainA.batch(1)))[0]
            sample_images_b = next(iter(trainB.batch(1)))[0]
            z_sample_a, _, _ = generator.encode(sample_images_a)
            reconstructed_a = generator.decode(z_sample_a)

            tf.summary.image(f"Reconstructed A/epoch_{epoch + 1}", normalize_for_tensorboard(reconstructed_a),
                             max_outputs=1, step=epoch)
            tf.summary.image(f"Original A/epoch_{epoch + 1}", normalize_for_tensorboard(sample_images_a), max_outputs=1,
                             step=epoch)
            tf.summary.image(f"Original B/epoch_{epoch + 1}", normalize_for_tensorboard(sample_images_b), max_outputs=1,
                             step=epoch)

            print(f"Logged images at Epoch {epoch + 1}")
            train_writer.flush()

        # Save checkpoint
        #generator.save(os.path.join(checkpoint_dir, 'generator', f'epoch_{epoch + 1}'),
         #                              save_format='tf')
        #discriminator.save(os.path.join(checkpoint_dir, 'discriminator', f'epoch_{epoch + 1}'),
         #                                  save_format='tf')

# Main script
def main():
    epochs = 100
    checkpoint_dir = './checkpoint'

    # Create models
    generator = VAEGen(input_dim=256, z_dim=8)
    discriminator = MsImageDis(input_dim=256, num_scales=3, num_filters=64)

    # Optimizers and loss function
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
    dis_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    # Create checkpoint directory
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Start training
    train(generator, discriminator, gen_optimizer, dis_optimizer, epochs, loss_fn, checkpoint_dir)

if __name__ == '__main__':
    main()
