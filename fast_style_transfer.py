import functools
import os
import pydicom
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import utils
import shutil
import glob

print("TF Version: ", tf.__version__)
print("TF Hub verstion: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# Define image loading and visualisation functions
def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()

# Load content and style images
#(images, indices) = utils.load_perf_images('/Users/ebrahamalskaf/Documents/test', 224)
#for image, index in zip(images, indices):
 #   dir = f"{index}"
  #  path = os.path.join('Image_Paths', dir)
   # os.makedirs(path, exist_ok=True)

    #for m in range(len(image[0, 0, :])):
     #   plt.imshow(image[...,m], cmap='gray')
      #  plt.savefig(f"{m}.png")
       # shutil.move(f"{m}.png", f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/Fast_Style_Transfer/Image_Paths/{index}")

imgPaths = glob.glob(f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**//Fast_Style_Transfer/CONTENT/*.png")
    #imgRange = range(len(glob.glob(f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/Fast_Style_Transfer/Image_Paths/{index}/*.png")))

for imgPath in imgPaths:
    # Style transfer pipeline
    im_name = os.path.basename(imgPath)
    print(im_name)
    content_image_path = imgPath
    style_image_path = 'style.png'
    #style_dic = pydicom.dcmread(style_image_path)
    #style_dcm = style_dic.pixel_array
    #plt.imshow(style_dcm, cmap='gray')
    #plt.savefig("style.png")
    output_image_size = 256
    content_image_size = (output_image_size, output_image_size)

    # The style prediction model was trained with image size 256
    content_image = load_image(content_image_path, content_image_size)
    style_image_size = (256, 256)
    style_image = load_image("style.png", style_image_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
    #show_n([content_image, style_image], ['Content image', 'Style image'])

    # Load TF Hub module
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    ''' The signature is:
    outputs = hub_module(content_image, style_image)
    stylised_image = outputs[0] '''

    # Stylise content image with given style image
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylised_image = outputs[0]

    # Save stylised image
    plt.imshow(np.squeeze(stylised_image), cmap='gray')
    plt.savefig(f"{im_name}")
    shutil.move(f"{im_name}", "/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/Fast_Style_Transfer/STYLE")


    # Visualise input images and the generated stylised image
    #show_n([content_image, style_image, stylised_image],
     #          titles=['Original content image', 'Style image', 'Stylised image'])






