from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import numpy as np
import os
import pathlib
import cv2


data_dir=os.getcwd()+'\dataset'
data_dir=pathlib.Path(data_dir)

print(data_dir)

image_count=len(list(data_dir.glob('*/*.jpg')))
image_count=image_count+(len(list(data_dir.glob('*/*.jpeg'))))
image_count=image_count+(len(list(data_dir.glob('*/*.png'))))
print(image_count)

INIT_LR = 1e-4
EPOCHS = 30
BS = 32

injury_images_dict={
    'carcinoma_orale': list(data_dir.glob('carcinoma_orale/*')),
    'leucoplachia':list(data_dir.glob('leucoplachia/*')),
    'lichen_planus': list(data_dir.glob('lichen_planus/*')),
    'sano': list(data_dir.glob('sano/*'))
}

injury_labels_dict={
    'carcinoma_orale':0,
    'leucoplachia':1,
    'lichen_planus':2,
    'sano':3
}

X,y=[],[]


for injury_name, images in injury_images_dict.items():
    for image in images:
        img=cv2.imread(str(image))
        resized_img=cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(injury_labels_dict[injury_name])

X=np.array(X)
y=np.array(y)

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)
X_train_scaled= X_train/255
X_test_scaled= X_test/255

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest"
)

baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

headModel= baseModel.output
headModel=Flatten(name='flatten')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(4, activation='softmax')(headModel)

model=Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")

model.fit(
    aug.flow(X_train_scaled, y_train, batch_size=BS),
    steps_per_epoch=len(X_train_scaled) // BS,
    validation_data=(X_test_scaled,y_test),
    validation_steps=len(X_test_scaled) // BS,
    epochs=EPOCHS
)

info= model.evaluate(X_test_scaled, y_test)
print(info)

model_path=os.getcwd()+"//model//injury_model"


print("[INFO] saving mask detector model... path: %s"%(model_path+".h5"))
model.save(model_path+".h5")
