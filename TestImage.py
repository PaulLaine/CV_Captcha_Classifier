import keras

import tensorflow as tf

from keras import preprocessing
from keras.models import load_model

model = load_model("my_model")

image_size = (100, 100)
img = keras.preprocessing.image.load_img("natural_images/car/car_0075.jpg", target_size=image_size)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
predictions = model.predict(img_array)
score = predictions[0]
print("This image is %.2f percent car and %.2f percent other." % (100 * score), 100 * (1 - score))

