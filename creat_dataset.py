from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


datagen= ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.1,0.9]
)

for imageFolder in os.listdir('dataset/'):
    try: os.mkdir('newDataset')
    except: pass

    for file in os.listdir('dataset/'+imageFolder+'/'):
        if not os.path.exists('newDataset/'+imageFolder):
            os.mkdir('newDataset/'+imageFolder)
        img= load_img('dataset/'+imageFolder+'/'+file)
        x=img_to_array(img)
        x=x.reshape((1,)+ x.shape)
        i=0

        for batch in datagen.flow(x, batch_size=1, save_to_dir='newDataset/'+imageFolder, save_prefix=imageFolder, save_format='jpg'):
            i+=1
            if i > 5:
                break