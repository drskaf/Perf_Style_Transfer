import os
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import pydicom
from skimage.transform import resize
import dl_helpers
import utils
import pydicom
import cv2
import torch
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


image = '1_final.png'
#img = Image.open(image)
#img.convert("1")
img = cv2.imread(image)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()
#img = np.add.reduce(img, 2)
print(img.shape)

Image = 'style_1.dcm'
Image = pydicom.dcmread(Image)
Image = Image.pixel_array
print(Image.shape)

(images, indices) = utils.load_perf_images('/Users/ebrahamalskaf/Documents/test', 224)
IMG = images[0][...,0 ]
print(IMG.shape)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Loading SAM check point and automated mask generator class
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(img)