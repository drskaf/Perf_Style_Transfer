import os
   
# define the content layer from which feature map will be extracted
contentLayers = ["block4_conv2"]

# define the list of style layer blocks from our pre-trained CNN
styleLayers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",   
    "block5_conv1"
]

# define the style weight, content weight, and total-variation loss weight
styleWeight = 2e3
contentWeight = 1e7
tvWeight = 20.0

# define the number of epochs to train for along with the steps per each epoch
epochs = 10
stepsPerEpoch = 100

# define the path to the input content image, input style image, final output image, and path to the directory that will
# store the intermediate outputs
contentImage = "cont.dcm"
styleImage = "style.dcm"
finalImage = "final.png"
intermOutputs = "intermediate_outputs"

