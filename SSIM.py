from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import glob

def load_and_preprocess_image(image_path, target_size=(244, 244)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image[:, :, :3]
    return image

def calculate_ssim(content_image_path, stylised_image_path):
    # Load images
    content_image = load_and_preprocess_image(content_image_path)
    stylised_image = load_and_preprocess_image(stylised_image_path)

    # Calculate SSIM
    ssim_score, _ = ssim(content_image, stylised_image, multichannel=True, full=True)

    return ssim_score

# Example usage
content_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/CONTENT/*.png")
NSTstylised_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/NEURAL_STYLE_TRANSFER/STYLE/*.png")
FSTstylised_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/FAST_STYLE_TRANSFER/STYLE/*.png")
MUNITstylised_image_pathS = glob.glob("/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/MUNIT/STYLE/*.png")

# Calculate mean MSE
SSIM = []
for content_image_path, stylised_image_path in zip(content_image_pathS, NSTstylised_image_pathS):
    ssim_score = calculate_ssim(content_image_path, stylised_image_path)
    SSIM.append(ssim_score)
print("Structural Similarity Index (SSIM) score for NST:", np.mean(SSIM))

SSIM = []
for content_image_path, stylised_image_path in zip(content_image_pathS, FSTstylised_image_pathS):
    ssim_score = calculate_ssim(content_image_path, stylised_image_path)
    SSIM.append(ssim_score)
print("Structural Similarity Index (SSIM) score for FST:", np.mean(SSIM))

SSIM = []
for content_image_path, stylised_image_path in zip(content_image_pathS, MUNITstylised_image_pathS):
    ssim_score = calculate_ssim(content_image_path, stylised_image_path)
    SSIM.append(ssim_score)
print("Structural Similarity Index (SSIM) score for MUNIT:", np.mean(SSIM))


