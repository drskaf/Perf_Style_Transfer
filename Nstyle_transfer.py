# import necessary packages
import shutil
import matplotlib.pyplot as plt
import Nstyle_transfer_config as config
from neural_style import NeuralStyle
import tensorflow as tf
import os
import utils
import glob
import pandas as pd

def loadImage(imagePath):
	# specify the maximum dimension to which the image is to be
	# resized
	maxDim = 512

	# load the image from the given path, convert the image bytes
	# to a tensor, and convert the data type of the image
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_image(image, channels=3)
	image = tf.image.convert_image_dtype(image, tf.float32)

	# grab the height and width of the image, cast them to floats,
	# determine the larger dimension between height and width, and
	# determine the scaling factor
	shape = tf.cast(tf.shape(image)[:-1], tf.float32)
	longDim = max(shape)
	scale = maxDim / longDim

	# scale back the new shape, cast it to an integer, resize the
	# image to the new shape, and  add a batch dimension
	newShape = tf.cast(shape * scale, tf.int32)
	image = tf.image.resize(image, newShape)
	image = image[tf.newaxis, :]

	# return the resized image
	return image

@tf.function
def trainOneStep(image, styleTargets, contentTargets):
	# derive the style and content loss weight values
	styleWeight = config.styleWeight / len(config.styleLayers)
	contentWeight = config.contentWeight / len(config.contentLayers)

	# keep track of our gradients
	with tf.GradientTape() as tape:
		# run the content image through our neural style network to 
		# get its features, determine the loss, and add total
		# variational loss to regularize it
		outputs = extractor(image)
		loss = extractor.styleContentLoss(outputs, styleTargets, 
			contentTargets, styleWeight, contentWeight)
		loss += config.tvWeight * tf.image.total_variation(image)

	# grab the gradients of the loss with respect to the image and
	#  apply the gradients to update the image after clipping the
	# values to [0, 1] range
	grad = tape.gradient(loss, image)
	opt.apply_gradients([(grad, image)])
	image.assign(extractor.clipPixels(image))

# Load files and images
label_file = pd.read_csv('/Users/ebrahamalskaf/Documents/final_surv.csv')
(images, indices) = utils.load_perf_images('/Users/ebrahamalskaf/Documents/**PERFUSION_CLASSIFICATION**/STRESS_images', 224)

for image, index in zip(images, indices):
	if index in label_file['ID'].values:
		print(index)
		dir = f"{index}"
		path = os.path.join('Train_Style_images', dir)
		os.makedirs(path, exist_ok=True)

		for m in range(len(image[0, 0, :])):
			plt.imshow(image[..., m], cmap='gray')
			plt.savefig(f"{m}_content.png")
			shutil.move(f"{m}_content.png",
						f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/Fast_Style_Transfer/Train_Style_images/{index}")

		imgPaths = glob.glob(
			f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/Fast_Style_Transfer/Train_Style_images/{index}/*.png")
		imgRange = range(len(glob.glob(
			f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/Fast_Style_Transfer/Train_Style_images/{index}/*.png")))

		for imgPath, j in zip(imgPaths, imgRange):
			print(imgPath)
			# initialize the Adam optimizer
			opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99,
									 epsilon=1e-1)

			# load the content and style images
			print("[INFO] loading content and style images...")
			contentImage = loadImage(imgPath)
			styleImage = "style.png"
			styleImage = loadImage(styleImage)

			# grab the contents layer from which feature maps will be extracted
			# along with the style layer blocks
			contentLayers = config.contentLayers
			styleLayers = config.styleLayers

			# initialize our network to extract features from the style and
			# content images
			print("[INFO] initializing off the extractor network...")
			extractor = NeuralStyle(styleLayers, contentLayers)

			# extract the features from the style and content images
			styleTargets = extractor(styleImage)["style"]
			contentTargets = extractor(contentImage)["content"]

			# initialize the content image as a TensorFlow variable along with
			# the total number of steps taken in the current epoch
			print("[INFO] training the style transfer model...")
			image = tf.Variable(contentImage)
			step = 0

			# loop over the number of epochs
			for epoch in range(config.epochs):
				# loop over the number of steps in the epoch
				for i in range(config.stepsPerEpoch):
					# perform a single training step, then increment our step
					# counter
					trainOneStep(image, styleTargets, contentTargets)
					step += 1

				# construct the path to the intermediate resulting image (for
				# visualization purposes) and save it
				print("[INFO] training step: {}".format(step))
				p = "_".join([str(epoch), str(i)])
				p = "{}.png".format(p)
				p = os.path.join(config.intermOutputs, p)
				extractor.tensorToImage(image).save(p)

			# save the final stylized image
			extractor.tensorToImage(image).save(f"{j}_final.png")
			shutil.move(f"{j}_final.png", f"/Users/ebrahamalskaf/Documents/**STYLE_TRANSFER**/Train_Style_images/{index}")

	else:
		continue



