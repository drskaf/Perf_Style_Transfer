import numpy as np
from PIL import Image
import glob

def load_and_preprocess_image(image_path, target_size=(244, 244)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image[:, :, :3]
    return image

def calculate_mse(content_image_path, stylised_image_path):
    # Load images
    content_image = load_and_preprocess_image(content_image_path)
    stylized_image = load_and_preprocess_image(stylised_image_path)

    # Convert images to numpy arrays
    content_array = np.array(content_image)
    stylized_array = np.array(stylized_image)

    # Ensure both images have the same shape
    if content_array.shape != stylized_array.shape:
        raise ValueError("Content and stylized images must have the same shape.")

    # Calculate Mean Squared Error (MSE)
    mse = np.mean(np.square(content_array - stylized_array))

    return mse

# Example usage
content_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/CONTENT/*.png")
NSTstylised_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/STYLE/*.png")
FSTstylised_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/FAST_STYLE_TRANSFER/STYLE/*.png")
MUNITstylised_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE/*.png")

# Calculate mean MSE
MSE = []
for content_image_path, stylised_image_path in zip(content_image_pathS, NSTstylised_image_pathS):
    mse_score = calculate_mse(content_image_path, stylised_image_path)
    MSE.append(mse_score)
print("Mean Squared Error (MSE) score for NST:", np.mean(MSE))

MSE = []
for content_image_path, stylised_image_path in zip(content_image_pathS, FSTstylised_image_pathS):
    mse_score = calculate_mse(content_image_path, stylised_image_path)
    MSE.append(mse_score)
print("Mean Squared Error (MSE) score for FST:", np.mean(MSE))

MSE = []
for content_image_path, stylised_image_path in zip(content_image_pathS, MUNITstylised_image_pathS):
    mse_score = calculate_mse(content_image_path, stylised_image_path)
    MSE.append(mse_score)
print("Mean Squared Error (MSE) score for MUNIT:", np.mean(MSE))
