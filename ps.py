import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import glob

# Load the pre-trained VGG19 model without the top (fully connected) layers
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
vgg.trainable = False

# Define the layers to be used for feature extraction
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_style_layers = len(style_layers)

# Function to build the model for feature extraction
def get_feature_extractor(model, layers):
    outputs = [model.get_layer(name).output for name in layers]
    feature_extractor = Model(inputs=model.input, outputs=outputs)
    return feature_extractor

# Function to preprocess images for VGG19
def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = tf.image.resize(image, (224, 224))  # Resize to VGG input size
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image

# Function to compute perceptual similarity
def perceptual_similarity(image1, image2):
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    features1 = get_feature_extractor(vgg, style_layers)(image1)
    features2 = get_feature_extractor(vgg, style_layers)(image2)

    similarity = 0.0
    for feature1, feature2 in zip(features1, features2):
        similarity += tf.reduce_mean(tf.square(feature1 - feature2)) / tf.reduce_max(tf.square(feature1 - feature2))
    #similarity /= num_style_layers

    return similarity.numpy()

# Example usage
NST_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/STYLE/*.png")
FST_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/FAST_STYLE_TRANSFER/STYLE/*.png")
MUNIT_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE/*.png")
GT_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/STYLE_IMG_GRAY/*.png")

PS = []
# PS score for NST model
for path1, path2 in zip(NST_stylised_paths, GT_pathS):
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    similarity_score = perceptual_similarity(image1, image2)
    PS.append(similarity_score)
print("Perceptual similarity score for NST:", np.mean(PS))

# PS score for FST model
for path1, path2 in zip(FST_stylised_paths, GT_pathS):
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    similarity_score = perceptual_similarity(image1, image2)
    PS.append(similarity_score)
print("Perceptual similarity score for FST:", np.mean(PS))

# PS score for MUNIT model
for path1, path2 in zip(MUNIT_stylised_paths, GT_pathS):
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    similarity_score = perceptual_similarity(image1, image2)
    PS.append(similarity_score)
print("Perceptual similarity score for MUNIT:", np.mean(PS))
