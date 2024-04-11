import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.stats import entropy
from tqdm import tqdm
from PIL import Image
import os
import glob


# Load pre-trained InceptionV3 model
inception = InceptionV3(include_top=True, weights='imagenet')

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(299, 299)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image[:, :, :3]
    return image


# Function to calculate Inception Score
def calculate_inception_score(paths, num_splits=10):
    # Initialize lists to store predictions
    preds = []

    # Iterate over images
    for image_path in paths:
        image = load_and_preprocess_image(image_path)
        preds.extend(inception.predict(np.expand_dims(image, axis=0)))

    # Convert predictions to numpy array
    preds = np.array(preds)

    # Split predictions into num_splits splits
    splits = np.array_split(preds, num_splits)

    # Calculate per-class probabilities and then calculate the Inception Score
    scores = []
    for split in splits:
        p_yx = np.mean(split, axis=0)
        p_yx /= np.sum(p_yx) # Normalise probabilities to sum up to 1
        kl_divergence = entropy(p_yx.T, base=2)  # Specify base to ensure consistent calculation
        #kl_divergence_per_image = np.mean(kl_divergence)
        scores.append(np.exp(kl_divergence))

    # Return mean and standard deviation of Inception Scores
    return np.mean(scores), np.std(scores)

# Calculate IS score
NST_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/STYLE/*.png")
is_mean, is_std = calculate_inception_score(NST_paths)
print("Inception Score (mean) for NST:", is_mean)
print("Inception Score (std) for NST:", is_std)

FST_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/FAST_STYLE_TRANSFER/STYLE/*.png")
is_mean, is_std = calculate_inception_score(FST_paths)
print("Inception Score (mean) for FST:", is_mean)
print("Inception Score (std) for FST:", is_std)

MUNIT_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE/*.png")
is_mean, is_std = calculate_inception_score(MUNIT_paths)
print("Inception Score (mean) for MUNIT:", is_mean)
print("Inception Score (std) for MUNIT:", is_std)
