from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import pandas as pd

# dimensions of our images
img_width, img_height = 256, 256

# load the model we saved
fps = []
model = load_model('./classifier_model.h5')
model.compile(
              loss='categorical_crossentropy', 
              metrics=['acc'],
              optimizer='adam'
             )

# predicting images
# load some new test data based on the 2 diff GANs
# load the images
preds = []
files = os.listdir("./CUST")
for fil in files:
    path = "CUST/"+fil
    img = image.load_img(path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack(arr)
    classes = model.predict_classes(images, batch_size=32)
    preds.append(classes)
    fps.append(fil)

# save the paths and the predictions 
output = [fps,preds]
output = pd.DataFrame(output)
output.to_csv("cust_preds.csv")
