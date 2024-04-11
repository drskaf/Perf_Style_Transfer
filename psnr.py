import numpy as np
import cv2
import glob
import os
from PIL import Image

# Function to preprocess images for VGG19
def preprocess_image(image_path, target_size=(244, 244)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image[:, :, :3]
    return image

def psnr(original_path, reconstructed_path):
    # Convert images to grayscale if they are RGB
    original_image = preprocess_image(original_path)
    reconstructed_image = preprocess_image(reconstructed_path)

    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    if len(reconstructed_image.shape) == 3:
        reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2GRAY)

    # Calculate mean squared error (MSE)
    mse = np.mean((original_image - reconstructed_image) ** 2)

    # Calculate maximum pixel value (assumed to be 255 for 8-bit images)
    max_pixel = 255.0

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


# Load data
NST_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/STYLE/*.png")
FST_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/FAST_STYLE_TRANSFER/STYLE/*.png")
MUNIT_stylised_paths = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE/*.png")
GT_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/STYLE_IMG_GRAY/*.png")

PSNR = []
for path1, path2 in zip(GT_pathS, NST_stylised_paths):
    # Calculate PSNR
    psnr_score = psnr(path1, path2)
    PSNR.append(psnr_score)
print("PSNR for NST:", np.mean(PSNR))

PSNR = []
for path1, path2 in zip(GT_pathS, FST_stylised_paths):
    # Calculate PSNR
    psnr_score = psnr(path1, path2)
    PSNR.append(psnr_score)
print("PSNR for FST:", np.mean(PSNR))

PSNR = []
for path1, path2 in zip(GT_pathS, MUNIT_stylised_paths):
    # Calculate PSNR
    psnr_score = psnr(path1, path2)
    PSNR.append(psnr_score)
print("PSNR for MUNIT:", np.mean(PSNR))