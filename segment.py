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
from cv2 import UMat
import torch
import supervision as sv
from skimage.transform import resize

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image = '1_final.png'
#img = Image.open(image)
#img.convert("1")
img = cv2.imread(image)
#img = np.add.reduce(img, 2)
print(img.shape)

#Image = '00013.dcm'
#Image = pydicom.dcmread(Image)
#Image = Image.pixel_array
#plt.imshow(Image, cmap='magma')
#plt.show()

#(images, indices) = utils.load_perf_images('/Users/ebrahamalskaf/Documents/test', 224)
#IMG = images[0][...,0 ]
#print(IMG.shape)

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

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# Generating masks
masks = mask_generator.generate(img)
masks = sorted(masks, key=lambda x: x['area'], reverse=True)
print(len(masks))
print(masks[0].keys())

# Show masks overlaying the image
plt.figure(figsize=(10,10))
plt.imshow(img)
show_anns(masks)
plt.axis('off')
plt.show()
print(masks[3])
plt.imshow(masks[3]['segmentation'], cmap='gray')
plt.show()

# Cropping
mask = masks[3]['segmentation']
mask = resize(mask, (img.shape[0], img.shape[1]))
mask = mask.astype(np.uint8)
img = img / 255.0
print(img.shape)
print(mask.shape)
print(img.dtype)
print(mask.dtype)
print(img)
print(mask)

masked = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(masked)
plt.show()

# Interact with masks
masks = [
    mask['segmentation']
    for mask
    in sorted(masks, key=lambda x: x['area'], reverse=True)
]
sv.plot_images_grid(
    images=masks,
    grid_size=(3, int(len(masks) / 3)),
    size=(12, 12)
)

