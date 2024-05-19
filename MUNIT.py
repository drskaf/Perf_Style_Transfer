import datetime

from utils import *
from glob import glob
import time
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Activation, Layer, Input, Conv2D, ReLU, GlobalAveragePooling2D, AveragePooling2D, Normalization, ReLU, Dense, Normalization, UpSampling2D, BatchNormalization, LeakyReLU, LayerNormalization
from tensorflow.keras.activations import tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# Set the mixed precision policy
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

'''
Created by Ebraham Alskaf May 2024
MUNIT for style transfer using Tensorflow 2._
'''

##################################################################################
# Define arguments
##################################################################################
checkpoint_dir = 'checkpoint'
result_dir = 'results'
sample_dir = 'samples'
model_dir = 'model'

epochs = 10
iteration = 10000
batch_size = 1
print_freq = 10
save_freq = 100
num_style = 2

img_h = 256
img_w = 256
img_ch = 3

init_lr = 0.0001
ch = 64

""" Weight """
gan_w = 1.0
recon_x_w = 10.0
recon_s_w = 1.0
recon_c_w = 1.0
recon_x_cyc_w = 1.0

""" Generator """
n_res = 4
n_sample = 2
mlp_dim = pow(2, n_sample) * ch # default : 256
n_downsample = n_sample
n_upsample = n_sample
style_dim = 8

""" Discriminator """
n_dis = 4
n_scale = 3
sample_dir = os.path.join(sample_dir, model_dir)

""" Data folders """
trainA_dataset = glob('CONTENT_A/*.png')
trainB_dataset = glob('STYLE_IMG_GRAY_A/*.png')
dataset_num = max(len(trainA_dataset), len(trainB_dataset))

print("##### Information #####")
print("# max dataset number : ", dataset_num)
print("# batch_size : ", batch_size)
print("# epoch : ", epochs)
print("# iteration per epoch : ", iteration)
print("# style in test phase : ", num_style)

print()

print("##### Generator #####")
print("# residual blocks : ", n_res)
print("# Style dimension : ", style_dim)
print("# MLP dimension : ", mlp_dim)
print("# Down sample : ", n_downsample)
print("# Up sample : ", n_upsample)

print()

print("##### Discriminator #####")
print("# Discriminator layer : ", n_dis)
print("# Multi-scale Dis : ", n_scale)

##################################################################################
# Encoder and Decoders
##################################################################################

class Style_Encoder(Model):
    def __init__(self, channels, style_dim, regularizer=l2(0.01)):
        super(Style_Encoder, self).__init__()
        self.conv_blocks = [
        Conv2D(channels, kernel_size=7, strides=1, padding='same', activation='relu', kernel_regularizer=regularizer),
        Conv2D(channels * 2, kernel_size=4, strides=2, padding='same', activation='relu', kernel_regularizer=regularizer),
        Conv2D(channels * 4, kernel_size=4, strides=2, padding='same', activation='relu',kernel_regularizer=regularizer),
        Conv2D(channels * 4, kernel_size=4, strides=2, padding='same', activation='relu', kernel_regularizer=regularizer),
        Conv2D(channels * 4, kernel_size=4, strides=2, padding='same', activation='relu', kernel_regularizer=regularizer)
        ]
        self.pool = AveragePooling2D()
        self.flatten = Flatten()
        self.dense = Dense(style_dim)
        #self.SElogit = Conv2D(style_dim, kernel_size=1, strides=1, name='SE_logit', kernel_regularizer=regularizer)

    def call(self, inputs):
        x = inputs
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)  # Flatten to get (batch_size, channels)
        x = self.dense(x)  # Final dense layer to get (batch_size, style_dim)

        return x #self.SElogit(x)

class ResBlock(Layer):
    def __init__(self, channels, name="resblock"):
        super(ResBlock, self).__init__(name=name)
        self.conv1 = Conv2D(channels, kernel_size=3, strides=1, padding='same')
        self.norm1 = LayerNormalization()
        self.activ1 = ReLU()
        self.conv2 = Conv2D(channels, kernel_size=3, strides=1, padding='same')
        self.norm2 = LayerNormalization()
        self.activ2 = ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ2(x)
        return x + inputs

class Content_Encoder(Model):
    def __init__(self, channels, regularizer=l2(0.01)):
        super(Content_Encoder, self).__init__()
        self.conv1 = Conv2D(channels, kernel_size=7, strides=1, padding='same', kernel_regularizer=regularizer)
        self.norm1= LayerNormalization()
        self.activ1 = ReLU()
        self.conv2 = Conv2D(channels * 2, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizer)
        self.norm2 = LayerNormalization()
        self.activ2 = ReLU()
        self.conv3 = Conv2D(channels * 4, kernel_size=4, strides=2, padding='same', kernel_regularizer=regularizer)
        self.norm3 = LayerNormalization()
        self.activ3 = ReLU()
        self.res_blocks = [ResBlock(channels * 4) for _ in range(n_res)]

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activ3(x)
        for res_block in self.res_blocks:
            x = res_block(x)

        return x

def adaptive_instance_norm(content, gamma, beta):
    mean, variance = tf.nn.moments(content, [1, 2], keepdims=True)
    normalized = (content - mean) / tf.sqrt(variance + 1e-5)
    gamma = tf.reshape(gamma, [-1, 1, 1, content.shape[-1]])
    beta = tf.reshape(beta, [-1, 1, 1, content.shape[-1]])
    return gamma * normalized + beta

class AdaptiveResBlock(Layer):
    def __init__(self, channels, use_bias=True, regularizer=l2(0.01)):
        super(AdaptiveResBlock, self).__init__()
        self.conv1 = Conv2D(channels, kernel_size=3, strides=1, padding='same', use_bias=use_bias, kernel_regularizer=regularizer)
        self.conv2 = Conv2D(channels, kernel_size=3, strides=1, padding='same', use_bias=use_bias, kernel_regularizer=regularizer)
        self.relu = ReLU()

    def call(self, x, gamma1, beta1, gamma2, beta2):
        x_init = x

        x = self.conv1(x)
        x = adaptive_instance_norm(x, gamma1, beta1)
        x = self.relu(x)

        x = self.conv2(x)
        x = adaptive_instance_norm(x, gamma2, beta2)
        x = self.relu(x)

        return x + x_init

class MLP(Model):
    def __init__(self, mlp_dim, n_res, regularizer=l2(0.01)):
        super(MLP, self).__init__()
        self.channel = mlp_dim
        self.dense_layers = [Dense(self.channel, activation='relu', kernel_regularizer=regularizer) for _ in range(2)]
        self.mu_layers = [Dense(self.channel, kernel_regularizer=regularizer) for _ in range(n_res * 2)]
        self.var_layers = [Dense(self.channel, kernel_regularizer=regularizer) for _ in range(n_res * 2)]

    def call(self, style):
        x = style
        for dense in self.dense_layers:
            x = dense(x)

        mu_list = []
        var_list = []
        for mu_layer, var_layer in zip(self.mu_layers, self.var_layers):
            mu = mu_layer(x)
            var = var_layer(x)
            mu = tf.reshape(mu, [-1, 1, 1, self.channel])
            var = tf.reshape(var, [-1, 1, 1, self.channel])
            mu_list.append(mu)
            var_list.append(var)

        return mu_list, var_list

class Generator(Model):
    def __init__(self, mlp_dim, n_res, img_ch, regularizer=l2(0.01)):
        super(Generator, self).__init__()
        self.n_res = n_res
        # Define MLP
        self.mlp = MLP(mlp_dim, n_res, regularizer)

        # Define resblocks and adaptive instance normalization parameters
        self.resblocks = [AdaptiveResBlock(mlp_dim, regularizer=regularizer) for _ in range(n_res)]

        # Define upsampling and convolution blocks
        self.upsampling1 = UpSampling2D(size=(2,2))
        self.conv1 = Conv2D(mlp_dim // 2, kernel_size=5, padding='same', activation='relu')
        self.norm1 = LayerNormalization()
        self.activ1 = ReLU()
        self.upsampling2 = UpSampling2D(size=(2, 2))
        self.conv2 = Conv2D(mlp_dim // 4, kernel_size=5, padding='same', activation='relu')
        self.norm2 = LayerNormalization()
        self.activ2 = ReLU()

        # Final convolution
        self.final_conv = Conv2D(img_ch, kernel_size=7, padding='same')
        self.activation = Activation('tanh')

    def call(self, contents, style):
        mu, var = self.mlp(style)
        x = contents

        # Apply adaptive resblocks
        for i in range(self.n_res):
            x = self.resblocks[i](x, mu[2 * i], var[2 * i], mu[2 * i + 1], var[2 * i + 1])

        # Apply upsampling and convolution blocks
        x = self.upsampling1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activ1(x)
        x = self.upsampling2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ2(x)

        # Apply final conv and activation
        x = self.final_conv(x)
        x = self.activation(x)
        return x

##################################################################################
# Discriminator
##################################################################################

class Discriminator(Model):
    def __init__(self, channels, n_scale, n_dis, regularizer=l2(0.01)):
        super(Discriminator, self).__init__()
        self.n_scale = n_scale
        self.n_dis = n_dis
        self.channels = channels

        # Create lists to hold layers for each scale
        self.conv_blocks = []
        for scale in range(n_scale):
            block = [
            Conv2D(channels, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(0.2),
                       kernel_regularizer=regularizer)
            ]
            current_channels = channels
            for _ in range(1, n_dis):
                current_channels *= 2
                block.append(
                    Conv2D(current_channels, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(0.2),
                           kernel_regularizer=regularizer))
            block.append(Conv2D(1, kernel_size=1, strides=1, padding='same',
                                kernel_regularizer=regularizer))  # Final layer in block
            self.conv_blocks.append(block)

    def call(self, x_init):
        D_logit = []
        x = x_init
        for block in self.conv_blocks:
            for layer in block:
                x = layer(x)
            D_logit.append(x)
            # Apply downsampling here if needed between scales
            x = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')

        return D_logit

##################################################################################
# Loading data
##################################################################################

trainA = tf.data.Dataset.from_tensor_slices(trainA_dataset)
trainB = tf.data.Dataset.from_tensor_slices(trainB_dataset)

trainA = (trainA.shuffle(dataset_num).map(load_and_preprocess_image).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
trainB = (trainB.shuffle(dataset_num).map(load_and_preprocess_image).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

trainA_iterator = iter(trainA)
trainB_iterator = iter(trainB)

domain_A = trainA_iterator.get_next()
domain_B = trainB_iterator.get_next()

##################################################################################
# Losses
##################################################################################

def generator_loss(fake_logits):
    total_loss = 0
    for fake in fake_logits:
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake), fake, from_logits=True))
        total_loss += loss
    return total_loss

def discriminator_loss(real_logits, fake_logits):
    total_loss = 0
    for real, fake in zip(real_logits, fake_logits):
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real), real, from_logits=True))
        fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake), fake, from_logits=True))
        total_loss += real_loss + fake_loss
    return total_loss

def cycle_consistency_loss(real_images, reconstructed_images):
    return tf.reduce_mean(tf.abs(real_images - reconstructed_images))

def style_reconstruction_loss(real_style, reconstructed_style):
    return tf.reduce_mean(tf.square(real_style - reconstructed_style))

def total_variation_loss(image):
    return tf.reduce_mean(tf.image.total_variation(image))

##################################################################################
# Initialise models
##################################################################################

style_encoder_A = Style_Encoder(channels=ch, style_dim=style_dim)
content_encoder_A = Content_Encoder(channels=ch)
generator_A = Generator(mlp_dim=mlp_dim, n_res=n_res, img_ch=img_ch)
discriminator_A = Discriminator(channels=ch, n_scale=n_scale, n_dis=n_dis)

style_encoder_B = Style_Encoder(channels=ch, style_dim=style_dim)
content_encoder_B = Content_Encoder(channels=ch)
generator_B = Generator(mlp_dim=mlp_dim, n_res=n_res, img_ch=img_ch)
discriminator_B = Discriminator(channels=ch, n_scale=n_scale, n_dis=n_dis)

##################################################################################
# Set up optimisers
##################################################################################

optimizer_gen = tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.5)
optimizer_disc = tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=0.5)

##################################################################################
# Training
##################################################################################

# Logs to TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Function to display reconstructed images
import matplotlib.pyplot as plt
import io
# Function to log images
def log_image(tag, image, step):
    tf.summary.image(tag, image, step=step)

@tf.function
def train_step(images_A, images_B):
    with tf.GradientTape(persistent=True) as tape:
        style_A = style_encoder_A(images_A)
        content_A = content_encoder_A(images_A)
        style_B = style_encoder_B(images_B)
        content_B = content_encoder_B(images_B)

        #tf.print("Encoded Style A shape:", tf.shape(style_A))
        #tf.print("Encoded Content A shape:", tf.shape(content_A))
        #tf.print("Encoded Style B shape:", tf.shape(style_B))
        #tf.print("Encoded Content B shape:", tf.shape(content_B))

        fake_A = generator_A(content_B, style_A)
        fake_B = generator_B(content_A, style_B)
        reconstruct_A = generator_A(content_A, style_encoder_A(fake_B))
        reconstruct_B = generator_B(content_B, style_encoder_B(fake_A))

        #tf.print("Fake A shape:", tf.shape(fake_A))
        #tf.print("Fake B shape:", tf.shape(fake_B))
        #tf.print("Reconstruct A shape:", tf.shape(reconstruct_A))
        #tf.print("Reconstruct B shape:", tf.shape(reconstruct_B))

        real_A_logits = discriminator_A(images_A)
        real_B_logits = discriminator_B(images_B)
        fake_A_logits = discriminator_A(fake_A)
        fake_B_logits = discriminator_B(fake_B)

        #tf.print("Real A logits:", real_A_logits)
        #tf.print("Real B logits:", real_B_logits)
        #tf.print("Fake A logits:", fake_A_logits)
        #tf.print("Fake B logits:", fake_B_logits)

        gen_loss_A = tf.cast(generator_loss(fake_A_logits), tf.float16)
        gen_loss_B = tf.cast(generator_loss(fake_B_logits), tf.float16)
        disc_loss_A = tf.cast(discriminator_loss(real_A_logits, fake_A_logits), tf.float16)
        disc_loss_B = tf.cast(discriminator_loss(real_B_logits, fake_B_logits), tf.float16)
        cycle_loss_A = tf.cast(cycle_consistency_loss(images_A, reconstruct_A), tf.float16)
        cycle_loss_B = tf.cast(cycle_consistency_loss(images_B, reconstruct_B), tf.float16)
        style_loss_A = tf.cast(style_reconstruction_loss(style_A, style_encoder_A(reconstruct_B)), tf.float16)
        style_loss_B = tf.cast(style_reconstruction_loss(style_B, style_encoder_B(reconstruct_A)), tf.float16)
        tv_loss_A = tf.cast(total_variation_loss(fake_A), tf.float16)
        tv_loss_B = tf.cast(total_variation_loss(fake_B), tf.float16)

        total_gen_loss = gen_loss_A + gen_loss_B + cycle_loss_A + cycle_loss_B + style_loss_A + style_loss_B + tv_loss_A + tv_loss_B
        total_disc_loss = disc_loss_A + disc_loss_B

        # Apply gradients
        gen_gradients = tape.gradient(total_gen_loss,
                                      style_encoder_A.trainable_variables + content_encoder_A.trainable_variables +
                                      generator_A.trainable_variables + style_encoder_B.trainable_variables +
                                      content_encoder_B.trainable_variables + generator_B.trainable_variables)
        gen_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gen_gradients]

        disc_gradients = tape.gradient(total_disc_loss,
                                       discriminator_A.trainable_variables + discriminator_B.trainable_variables)
        disc_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in disc_gradients]

        optimizer_gen.apply_gradients(zip(gen_gradients,
                                          style_encoder_A.trainable_variables + content_encoder_A.trainable_variables +
                                          generator_A.trainable_variables + style_encoder_B.trainable_variables +
                                          content_encoder_B.trainable_variables + generator_B.trainable_variables))
        optimizer_disc.apply_gradients(zip(disc_gradients,
                                           discriminator_A.trainable_variables + discriminator_B.trainable_variables))

        tf.print("Gen Loss:", gen_loss_B, "Disc Loss:", disc_loss_B, "Cyc Loss:", cycle_loss_A, "Sty Loss:", style_loss_B, "TV Loss:", tv_loss_B)
        #tf.print("Disc Loss A:", disc_loss_A, "Disc Loss B:", disc_loss_B)

        return total_gen_loss, total_disc_loss

# Initialise Tensorboard writers
train_writer = tf.summary.create_file_writer(log_dir + "/images")
train_writer.set_as_default()

# Function to ensure logs are written
def flush_file_writer():
    train_writer.flush()

# Training loop
for epoch in range(epochs):
    for step, (img_batch_A, img_batch_B) in enumerate(zip(trainA, trainB)):
        #Print shapes before entering train_step
        print(f"Epoch {epoch}, Step {step}")
        print(f"img_batch_A shape: {img_batch_A.shape}")
        print(f"img_batch_B shape: {img_batch_B.shape}")
        g_loss, d_loss = train_step(img_batch_A, img_batch_B)

        if step % print_freq == 0:
            print(f"Epoch {epoch}, Step {step}, Gen Loss: {g_loss.numpy()}, Disc Loss: {d_loss.numpy()}")

            try:
                # Get reconstructed images
                fake_B = generator_B(content_encoder_A(img_batch_A), style_encoder_B(img_batch_B))
                reconstructed_A = generator_A(content_encoder_A(img_batch_A), style_encoder_A(fake_B))

                # Log images
                with train_writer.as_default():
                    tf.summary.image("Original_A", img_batch_A, step=epoch * iteration + step)
                    tf.summary.image("Style_B", img_batch_B, step=epoch * iteration + step)
                    tf.summary.image("Fake_B", fake_B, step=epoch * iteration + step)
                    tf.summary.image("Reconstructed_A", reconstructed_A, step=epoch * iteration + step)
                    print(f"Logged images at Epoch {epoch}, Step {step}")  # Print statement for verification
            except Exception as e:
                print(f"Error in logging images: {e}")

            try:
                generator_A.save(os.path.join(checkpoint_dir, f'generator_A_{epoch}_{step}'), save_format='tf')
                generator_B.save(os.path.join(checkpoint_dir, f'generator_B_{epoch}_{step}'), save_format='tf')
                content_encoder_A.save(os.path.join(checkpoint_dir, f'content_encoder_A_{epoch}_{step}'), save_format='tf')
                style_encoder_A.save(os.path.join(checkpoint_dir, f'style_encoder_A_{epoch}_{step}'), save_format='tf')
                style_encoder_B.save(os.path.join(checkpoint_dir, f'style_encoder_B_{epoch}_{step}'), save_format='tf')
            except Exception as e:
                print(f"Error in saving models: {e}")

train_writer.flush()





