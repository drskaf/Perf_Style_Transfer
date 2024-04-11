import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
loaded_model = load_model('munit_trained_model.h5')

# Load image function
def load_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = img/255.0
    return img

# Load new content and style images for testing
content_image_path = '73.png'
style_image_path = 'style.png'

# Preprocess content and style images
content_img = load_image(content_image_path)
content_img = content_img[:,:,:3]
style_img = load_image(style_image_path)
style_img = style_img[:,:,:3]

# Reshape the images to match the model input shape
content_img = np.expand_dims(content_img, axis=0)
style_img = np.expand_dims(style_img, axis=0)

# Use the model to generate an output image
output_img = loaded_model.predict([content_img, style_img])

# Postprocess the output image
output_img = (output_img * 255).astype(np.uint8)  # Rescale pixel values
output_img = output_img[0]  # Remove batch dimension

# Display or save the output image
output_image = Image.fromarray(output_img)
output_image.show()  # Display the output image
output_image.save('output_image.png')  # Save the output image