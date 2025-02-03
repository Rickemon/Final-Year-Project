import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
import numpy as np
import os
from scipy.signal import resample

root =os.path.dirname(os.path.abspath("Solution.py"))[:40]

model = keras.models.Sequential()


model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(256, 256, 3)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Dropout(0.5))


model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

# Output
model.add(Conv2D(3, (1,1), activation='softmax'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


img_names = sorted(os.listdir(root + "Datasets\\PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0\\FundusImagesResized"))
images = np.empty((int(len(img_names)*2), 256, 256, 3))
done = 0
for i in img_names:
  image = (cv.imread(root + "Datasets\\PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0\\FundusImagesResized\\" + i))
  images[done*2] = image
  images[done*2+1] = image
  done +=1
  print(str(int(100* (done/488)))+'% done')

mask_names = sorted(os.listdir(root + "Datasets\\PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0\\ExpertsSegmentations\\Masks"))
masks = np.empty((int(len(mask_names)), 256, 256, 3))
done = 0
for i in mask_names:
  image = np.loadtxt(root + "Datasets\\PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0\\ExpertsSegmentations\\Masks\\"+ i[:8]+'_exp1.mask', delimiter=',')
  image = image.reshape(256, 256,3)
  masks[done] = image
  done +=1
  print(str(int(100* (done/976)))+'% done')

train_images = images[:488]
train_masks = masks[:488]
val_images = images[488:]
val_masks = masks[488:]

model.fit(
    train_images,
    train_masks,
    batch_size=10,
    epochs=50,
    validation_data=(val_images, val_masks))

model = keras.models.load_model('Final Year Project\\Models\\Exstract4.h5')
