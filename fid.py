import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from scipy.linalg import sqrtm
from PIL import Image
import glob
import os

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(299, 299)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image[:, :, :3]
    return image

# Function to calculate FID between two sets of images
def calculate_fid(images1, images2):
    # Load pre-trained InceptionV3 model
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Define feature extractor model
    feature_extractor = Model(inputs=inception.input, outputs=inception.layers[-1].output)

    # Extract features for real images
    real_features = []
    # Extract features for real images
    for image_path in images1:
        image = load_and_preprocess_image(image_path)
        feature = feature_extractor.predict(np.expand_dims(image, axis=0))
        real_features.append(feature)

    # Extract features for generated images
    generated_features = []
    # Extract features for generated images
    for image_path in images2:
        image = load_and_preprocess_image(image_path)
        feature = feature_extractor.predict(np.expand_dims(image, axis=0))
        generated_features.append(feature)

    # Convert lists to numpy arrays
    real_features = np.concatenate(real_features, axis=0)
    generated_features = np.concatenate(generated_features, axis=0)

    # Calculate mean and covariance for real and generated features
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Calculate squared Frobenius norm of difference between means
    sum_squared_diff = np.sum((mu1 - mu2) ** 2)

    # Calculate square root of product of covariances
    cov_sqrt = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrtm calculation
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Calculate FID score
    fid = sum_squared_diff + np.trace(sigma1 + sigma2 - 2 * cov_sqrt)

    return fid

# Example usage
real_images_path = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/STYLE_IMG_GRAY/*.png")
NSTgenerated_images_path = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/STYLE/*.png")
FSTgenerated_images_path = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/FAST_STYLE_TRANSFER/STYLE/*.png")
MUNITgenerated_images_path = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE/*.png")

fid_score1 = calculate_fid(real_images_path, NSTgenerated_images_path)
print("Fréchet Inception Distance (FID) score for NST:", fid_score1)

fid_score2 = calculate_fid(real_images_path, FSTgenerated_images_path)
print("Fréchet Inception Distance (FID) score for FST:", fid_score2)

fid_score3 = calculate_fid(real_images_path, MUNITgenerated_images_path)
print("Fréchet Inception Distance (FID) score for MUNIT:", fid_score3)
