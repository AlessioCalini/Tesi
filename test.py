import PIL.Image as Image
import keras
import numpy as np
from tensorflow.keras.models import load_model
test_dictionary={
    0:'carcinoma orale',
    1:'leucoplachia',
    2:'lichenplanus',
    3:'sano'
}

injury_model= load_model("model/injury_model.h5")


IMAGE_SHAPE=(224,224)

img=Image.open("Normal-tongue-out--1296x728-gallery_slide1.jpg").resize(IMAGE_SHAPE)
img=np.array(img)
print(img)
b=img/255.0
print(b)


prediction=injury_model.predict(b[np.newaxis, ...])
print(prediction)
maxindex=np.argmax(prediction)
print(maxindex)
print(test_dictionary[maxindex])