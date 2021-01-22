import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import PIL

print(tf.__version__)
#Download the pre existing dataset from google
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
# get the image count
image_count = len(list(data_dir.glob('*/*.jpg')))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))

batch_size = 32
img_heigh = 180
img_width = 180

#split the data set in to 2. One for training the model and another one for validating the model

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_heigh, img_width),
        batch_size=batch_size )
# dataset for validating the model
val_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_heigh, img_width),
        batch_size=batch_size)

#different class names in the dataset
class_names = train_ds.class_names

# The RGB channel values are in the [0, 255] range. This is not ideal for a neural network; 
#in general you should seek to make your input values small. Here, we will standardize values to be in the [0, 1] 
#by using a Rescaling layer.
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_heigh, img_width, 3))

# caching the data set for performance.
# prefetch() overlaps data preprocessing and model execution while training.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create a model
model = Sequential([normalization_layer, 
                    layers.Conv2D(16,3,activation='relu', padding = 'same'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32,3,activation='relu', padding = 'same'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64,3,activation='relu', padding = 'same'),
                    layers.MaxPooling2D(),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(len(class_names))
                   ])
# Compile a Model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
)

# Data augmentation takes the approach of generating additional training data from your existing examples 
# by augmenting them using random transformations that yield believable-looking images
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                input_shape=(img_heigh, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)
])

# model with data augmentation
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(class_names))
])

# compile the augmented model
model.compile(optimizer='adam',
             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Predict on new data

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(sunflower_path, target_size=(img_heigh, img_width))
# img
img_array = keras.preprocessing.image.img_to_array(img)
# print(img_array)
img_array =tf.expand_dims(img_array,0)
img_array

predictions = model.predict(img_array)
predictions
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
