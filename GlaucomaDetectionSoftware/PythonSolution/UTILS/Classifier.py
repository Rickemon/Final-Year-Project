import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
import numpy as np
import os
from scipy.signal import resample

root =os.path.dirname(os.path.abspath("Solution.py"))[:40]

model = Sequential()

model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(32, 2)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


X = np.loadtxt(root + 'Datasets\PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0\ExpertsSegmentations\FixedConcatenatedResized\concatenated_array.csv', delimiter=',')
X = resized_coords.reshape(-1, 32, 2)
#print(resized_coords.shape)

y = np.loadtxt(root + 'Datasets\PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0\ExpertsSegmentations\Labels.txt', delimiter=',')
#print(y.shape)

model.fit(X, y, epochs=50, validation_split=0.2)
