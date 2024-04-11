# Overview
This repositry contains 3 models training script for style transfer of legacy stress perfusion CMR to high resolution maps:
1. Neural Style Transfer: this network will iterate through each example content image and generate a stylised image
2. Fast Style Transfer: this network will be loaded with trained weights and deployed on all content images to produce near-real-time stylised images
3. MUNIT: multimodal unsupervised image-to-image translation will use encoder and decoder to generate stylised images, trained on the whole dataset.

4. ## Performance metrics
5. 1. ps.py: Perceptual Similarity score, the higher the score the more similar the stylised image to original map.
   2. fid.py: Frechet Inception Distance, the lower the score the more similar extracted features.
   3. is.py: Inception Score, normally 1 -1000, higher is better, unless falling outside the range, meaning more diverse and less authentic.
      mse.py: Mean Squared Error difference from content images, the lower the more preserved contents.
   5. ssim.py: Structural Similarity Index, the higher value the more preservation to structural information.
   6. ssm.py: Style Similarity Score, the higher the score the more similarity with style reference. 

