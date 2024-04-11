import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import backend as K
from PIL import Image
import glob


# Function to preprocess images for VGG19
def preprocess_image(image_path, target_size=(244, 244)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image[:, :, :3]
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image

# Function to compute Gram matrix for a given feature map
def gram_matrix(feature_map):
    batch_size, height, width, channels = feature_map.shape
    features = tf.reshape(feature_map, (batch_size, height * width, channels))
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(height * width * channels, tf.float32)

# Function to compute style similarity metric using Gram matrix
def style_similarity(reference_style_image_path, generated_image_path):
    # Load VGG19 model pretrained on ImageNet data
    vgg = VGG19(weights='imagenet', include_top=False)
    # Define intermediate layers for style extraction
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
    # Create a model that outputs the style layers' activations
    style_extractor = Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in style_layers])

    # Load and preprocess the reference style image
    reference_style_image = preprocess_image(reference_style_image_path)
    # Extract style features from the reference style image
    reference_style_features = style_extractor.predict(reference_style_image)

    # Load and preprocess the generated image
    generated_image = preprocess_image(generated_image_path)
    # Extract style features from the generated image
    generated_style_features = style_extractor.predict(generated_image)

    # Compute Gram matrices for the style features
    reference_grams = [gram_matrix(feature_map) for feature_map in reference_style_features]
    generated_grams = [gram_matrix(feature_map) for feature_map in generated_style_features]

    # Compute mean squared error (MSE) between Gram matrices
    mse_scores = [np.mean(np.square(ref_gram - gen_gram)) for ref_gram, gen_gram in
                  zip(reference_grams, generated_grams)]

    # Compute average MSE across all style layers
    style_similarity_score = np.mean(mse_scores)

    return style_similarity_score

# Load data
NST_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/STYLE/*.png")
FST_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/FAST_STYLE_TRANSFER/STYLE/*.png")
MUNIT_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE/*.png")
GT_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/STYLE_IMG_GRAY/*.png")

SSM = []
for path1, path2 in zip(GT_pathS, NST_stylised_paths):
    style_similarity_score = style_similarity(path1, path2)
    SSM.append(style_similarity_score)
print("Style similarity score for NST:", np.mean(SSM))

SSM = []
for path1, path2 in zip(GT_pathS, FST_stylised_paths):
    style_similarity_score = style_similarity(path1, path2)
    SSM.append(style_similarity_score)
print("Style similarity score for FST:", np.mean(SSM))

SSM = []
for path1, path2 in zip(GT_pathS, MUNIT_stylised_paths):
    style_similarity_score = style_similarity(path1, path2)
    SSM.append(style_similarity_score)
print("Style similarity score for MUNIT:", np.mean(SSM))
