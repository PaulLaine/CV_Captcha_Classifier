import numpy as np
import pandas as pd
import Functions as fct
import LetterDetection as ld
import glob
import os
import keras
import cv2
from sklearn.model_selection import train_test_split
from random import sample
import random



path_dataset = 'C:/Users/plain/PycharmProjects/CaptchaProject/dataset/*.png'
path_dataset_train_char = 'C:/Users/plain/PycharmProjects/CaptchaProject/char_dataset_train/*.png'
path_dataset_test_char = 'C:/Users/plain/PycharmProjects/CaptchaProject/char_dataset_test/*.png'

dataset = glob.glob(path_dataset)

idx = random.sample(range(len(dataset)), int(len(dataset)*0.7))
idx = sorted(idx)

train_dataset = []
for ids in idx:
    train_dataset.append(dataset[ids])

print(len(train_dataset))


ld.letterDetection(train_dataset, 'char_dataset_train')

train_dataset_char = glob.glob(path_dataset_train_char)

# --- TRAINING DATA PROCESSING ---

X_train, y_train = fct.processCaptcha(train_dataset_char)


# --- MODEL CREATION AND FIT ---

input_shape = (28, 28, 1)

model = fct.createModel(input_shape=input_shape)

print(model.summary)

batch_size = 128
epochs = 50

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)




# --- MODEL TEST AND VALIDATION ---

for ids in reversed(idx):
    del dataset[ids]

test_dataset = dataset

print(len(test_dataset))

ld.letterDetection(test_dataset, 'char_dataset_test')

test_dataset_char = glob.glob(path_dataset_test_char)



X_test, y_test = fct.processCaptcha(test_dataset_char)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])















