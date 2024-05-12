import numpy as np
from PIL import Image
from keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from keras.models import Sequential, Model, load_model
from keras.initializers import RandomNormal
import glob
import os
from keras.utils import Sequence

def load_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = img/255.0
    return img

# Data
PATHS = []
directory = "/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/Train_Style_images"
for root, dirs, files in os.walk(directory, topdown=True):
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    for dir in dirs:
        dir_path = os.path.join(directory, dir)
        files = sorted(os.listdir(dir_path))
        for file in files:
            file_path = os.path.join(dir_path, file)
            if "content" in file_path:
                PATHS.append(file_path)

content_image_paths = PATHS
print(len(content_image_paths))
style_image_paths = glob.glob(f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE_IMG_GRAY/*.png")

# Build MUNIT model
def residual_block(x, filters, kernel_size=3, strides=1, padding='same'):
    y = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size, strides=strides, padding=padding)(y)
    y = BatchNormalization()(y)
    return Add()([x, y])

def build_model(input_shape):
    # Define input layers
    content_input = Input(shape=input_shape, name='content_input')
    style_input = Input(shape=input_shape, name='style_input')

    # Shared encoder
    encoder = Conv2D(64, (3,3), strides=2, padding='same')(content_input)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Conv2D(128, (3,3), strides=2, padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)

    # Content encoder
    content_encoder = Conv2D(256, (3, 3), strides=2, padding='same')(encoder)
    content_encoder = BatchNormalization()(content_encoder)
    content_encoder = Activation('relu')(content_encoder)
    content_encoder = UpSampling2D(size=(2, 2))(content_encoder)
    content_encoder = UpSampling2D(size=(2, 2))(content_encoder)

    # Style encoder
    style_encoder = Conv2D(256, (3, 3), strides=2, padding='same')(style_input)
    style_encoder = BatchNormalization()(style_encoder)
    style_encoder = Activation('relu')(style_encoder)

    # Concatenate content and style encodings
    combined_encoding = Concatenate()([content_encoder, style_encoder])

    # Decoder
    decoder = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(combined_encoding)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2DTranspose(64, (3, 3), strides=1, padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2DTranspose(3, (3, 3), strides=1, padding='same')(decoder)

    # Output layer
    output = Activation('sigmoid')(decoder)

    # Define the model
    model = Model(inputs=[content_input, style_input], outputs=output, name='MUNIT')
    return model

# Building model
input_shape = (256, 256, 3)
MUNIT = build_model(input_shape)

# Load imagings
content_images = []
style_images = []
for path in content_image_paths:
    img = load_image(path)
    img = img[:, :, :3]
    content_images.append(img)

for path in style_image_paths:
    img = load_image(path)
    img = img[:, :, :3]
    style_images.append(img)

# Training the model
def train_model(model, content_images, style_images, batch_size=32, epochs=10):

    # Define loss function and optimizer
    # Compile the model
    model.compile(optimizer='adam', loss='mean_absolute_error')

    class DataGenerator(Sequence):
        def __init__(self, content_images, style_images, batch_size=batch_size):
            self.content_images = content_images
            self.style_images = style_images
            self.batch_size = batch_size
            self.num_samples = len(content_images)

        def __len__(self):
            return (self.num_samples + self.batch_size - 1) // self.batch_size

        def __getitem__(self, idx):
            batch_content_images = []
            batch_style_images = []

            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.num_samples)

            for i in range(start_idx, end_idx):
                content_img = content_images[i]
                style_img = style_images[i % len(self.style_images)]
                batch_content_images.append(content_img)
                batch_style_images.append(style_img)

            return [np.array(batch_content_images), np.array(batch_style_images)], np.array(batch_content_images)

    # Assuming load_and_preprocess_image is a function to load and preprocess an image

    # Create the data generator
    train_generator = DataGenerator(content_images, style_images, batch_size=batch_size)

    # Training loop
    epochs = epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for i in range(len(train_generator)):
            X_batch, y_batch = train_generator[i]
            loss = MUNIT.train_on_batch(X_batch, y_batch)
            print(f'Batch {i + 1}/{len(train_generator)} - Loss: {loss}')

    # Save the trained model after training
    model.save('munit_trained_model.h5')

train_model(MUNIT, content_images, style_images)
