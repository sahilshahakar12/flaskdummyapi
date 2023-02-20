from flask import Flask, request, jsonify, escape
from PIL import Image
import io
import tensorflow as tf
from keras.models import load_model
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
import codecs

model = load_model("categorization.h5")

class_name = ['overripe',
 'raw-green',
 'ripe']

def findRipenessFunc(img_path):
    test_image= load_img(img_path, target_size=(300,300))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    predicted_class= class_name[np.argmax(prediction[0])]
    result = str(predicted_class)

    if result == 'overripe':
        carbLevel = 28
    elif result == 'raw-green':
        carbLevel = 20
    elif result == 'ripe':
        carbLevel = 24
    else:
        carbLevel = 0
    return carbLevel, result
