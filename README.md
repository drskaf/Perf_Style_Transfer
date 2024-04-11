# Overview
This repositry contains 3 models training script for style transfer of legacy stress perfusion CMR to high resolution maps:
1. Neural Style Transfer: this network will iterate through each example content image and generate a stylised image
2. Fast Style Transfer: this network will be loaded with trained weights and deployed on all content images to produce near-real-time stylised images
3. MUNIT: multimodal unsupervised image-to-image translation will use encoder and decoder to generate stylised images, trained on the whole dataset.

4. ## Performance metrics
5. 1. PS.py: Perceptual Similarity score, the higher the score the more similar the stylised image to original map.
   2. FID.py: Frechet Inception Distance, the lower the score the more similar extracted features. 
