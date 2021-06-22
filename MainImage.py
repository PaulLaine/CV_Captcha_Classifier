import os
import random
import glob
import cv2
import shutil

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

nbOfClasses = 8
path = 'natural_images/'
nameClass2Test = 'car'

"""lenClass2Test = len(os.listdir(path + nameClass2Test))
repartition = int(lenClass2Test/nbOfClasses)


shutil.rmtree(path + "/SecondClass")
os.mkdir(path + "/SecondClass")
newfolder = []

folderList = os.listdir(path)
del folderList[folderList.index(nameClass2Test)]
del folderList[folderList.index('SecondClass')]
print(folderList)

for i in folderList :
    cpt = 0
    print(i)
    while(cpt < repartition):
        a = (random.choice([x for x in os.listdir(path + i) if os.path.isfile(os.path.join(path + i, x))]))
        if(a not in newfolder):
            img = cv2.imread(path + i + '/' + a)
            cv2.imwrite(os.path.join(path + "/SecondClass/", a), img)
            newfolder.append(a)
            cpt += 1
"""


#path = r"D:\UQAC\Deep Learning\Projet\CaptchaImages\natural_images"
"""num_skipped = 0
dirs = os.listdir(path)
print(dirs)

for folder_name in (dirs):
    print(folder_name)
    folder_path = os.path.join(path, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)"""

#Dataset Generation
image_size = (100, 100)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory("Classes", validation_split=0.2, subset="training", seed=1337,
    image_size=image_size, batch_size=batch_size,)

val_ds = tf.keras.preprocessing.image_dataset_from_directory("Classes", validation_split=0.2, subset="validation", seed=1337,
    image_size=image_size, batch_size=batch_size,)

"""plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
"""

data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),])

"""
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
"""

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)

epochs = 30
#callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),]
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=epochs, validation_data=val_ds,)

model.save("my_model")
