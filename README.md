# custom_starganv2
CS_506 Project

SUMMARY

We’re analyzing the performance of a Cycle GAN (StarGAN v2) when
trained on multiple unique domains of human faces. We are also
developing new measures for correctness of cross domain style transfer.

DATA

There are 3 sources of data:
1. CelebA HQ Dataset
2. A Black Celebrities dataset (Custom)
3. Google (We may scrape the internet for some more images based
on the final makeup of the project group)

STEPS

1. Scrape the internet for more images and clean them
2. Build a model to cluster images within domains based on visual
similarity
3. Train StarGAN V2 based on all the domains, evaluate the success
of cross-domain style transfer by evaluating visual similarity
4. Evaluate the success of cross-domain style transfer by evaluating
visual similarity and dissimilarity of the model outputs and real
images in the training data
