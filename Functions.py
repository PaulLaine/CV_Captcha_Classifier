import pandas as pd
import numpy as np
import keras
from keras import layers
import os
import cv2
from captcha.image import ImageCaptcha
from random import randrange

WHITE = [255, 255, 255]

def createList():
    asclist = []
    for i in range(48,58,1):
        asclist.append(chr(i))
    for i in range(97,123,1):
        asclist.append(chr(i))
    return asclist


def generateCaptcha():
    code = ""
    char_list = createList()
    for _ in range(5):
        randnum = randrange(len(char_list))
        char = char_list[randnum]
        code = code + char
        del char_list[randnum]
    image = ImageCaptcha(width=280, height=90)
    #data = image.generate(code)
    image.write(code, "valid_captcha.png")
    return code

def getLabel(path):
    code = os.path.split(path)
    code = code[1][:2]
    return code

def processCaptcha(folder):
    # --- DATA PROCESSING ---

    df_train = pd.DataFrame(folder, columns=['file'])

    label_char = []

    for elem in folder:
        label = getLabel(elem)
        label = label.replace("_", "")
        label_char.append(int(label))

    df_train["label"] = label_char

    encoded_char = []

    for label in label_char:
        encode_label = np.zeros(36)
        encode_label[label] = 1
        encoded_char.append(encode_label)

    df_train['encode_char'] = encoded_char

    # --- DATA PREPARATION ---

    # Store all png images into one numpy array
    images = []

    input_shape = (28, 28, 3)
    # I want to be sure that every image is consitent
    for i, row in df_train.iterrows():
        img_name = row['file']

        numpy_image = cv2.imread(os.path.join(img_name))
        #cv2.imshow("original", numpy_image)
        #cv2.waitKey(0)
        image_res = cv2.resize(numpy_image, (14, 28), interpolation=cv2.INTER_AREA)
        #cv2.imshow("resized", image_res)
        #cv2.waitKey(0)
        image_border = cv2.copyMakeBorder(image_res, top=0, bottom=0, left=7, right=7, borderType=cv2.BORDER_CONSTANT,
                                       value=WHITE)
        #cv2.imshow("border", image_border)
        #cv2.waitKey(0)
        ret, thresh = cv2.threshold(image_border, 200, 255, cv2.THRESH_BINARY)

        image_final = thresh[:,:,:1]

        #cv2.imshow("final", image_final)
        #cv2.waitKey(0)
        images.append(image_final)



    # Normalize array of images
    images = np.array(images) / 255

    # Define X_data and target
    X = np.array(images.copy())
    y = np.array(encoded_char.copy())

    return X, y


def createModel(input_shape):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(36, activation="softmax"),
        ]
    )
    return model