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
import tensorflow as tf

list_char = fct.createList()

main_path = 'C:/Users/plain/PycharmProjects/CaptchaProject/'
path_dataset = 'C:/Users/plain/PycharmProjects/CaptchaProject/dataset/*.png'
path_dataset_train_char = 'C:/Users/plain/PycharmProjects/CaptchaProject/char_dataset_train/*.png'
path_dataset_test_char = 'C:/Users/plain/PycharmProjects/CaptchaProject/char_dataset_test/*.png'
path_dataset_valid_char = 'C:/Users/plain/PycharmProjects/CaptchaProject/char_dataset_valid/*.png'

dataset = glob.glob(path_dataset)

idx = random.sample(range(len(dataset)), int(len(dataset)*0.7))
idx = sorted(idx)

train_dataset = []
for ids in idx:
    train_dataset.append(dataset[ids])

print('letterDetections for train dataset...')

ld.letterDetection(train_dataset, 'char_dataset_train')

train_dataset_char = glob.glob(path_dataset_train_char)

# --- TRAINING DATA PROCESSING ---

X_train, y_train = fct.processCaptcha(train_dataset_char)


# --- MODEL CREATION AND FIT ---

input_shape = (28, 28, 1)

model = fct.createModel(input_shape=input_shape)

print(model.summary)

batch_size = 128
epochs = 100

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(), metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)




# --- MODEL TEST ---

for ids in reversed(idx):
    del dataset[ids]

test_dataset = dataset

print(len(test_dataset))

print('letterDetections for test dataset...')

ld.letterDetection(test_dataset, 'char_dataset_test')

test_dataset_char = glob.glob(path_dataset_test_char)

X_test, y_test = fct.processCaptcha(test_dataset_char)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# --- MODEL VALIDATION ---

code = fct.generateCaptcha()

validation_captcha = main_path + "valid_captcha.png"

ld.letterDetection(validation_captcha, "char_dataset_valid", code_captcha=code, single_image=True)

valid_dataset_char = glob.glob(path_dataset_valid_char)

X_valid, y_valid = fct.processCaptcha(valid_dataset_char)

code_predicted = ""

for i in range(len(X_valid)):
    array_predicted = model.predict(X_valid[i])
    char_predicted = list_char[array_predicted.index(1)]
    code_predicted = code_predicted + char_predicted

print("Actual code : " + code)
print("Code predicted : " + code_predicted)

if (code == code_predicted):
    print("Result : Predicted !")
else :
    print("Result : Not predicted...")












