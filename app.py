from flask import Flask, request, jsonify, escape
from PIL import Image
import io
import tensorflow as tf
from keras.models import load_model
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
import codecs

from findRipeness import findRipenessFunc

model = load_model("recognition.h5")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

class_name = ['Apples',
 'Bananas',
 'Oranges']

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['image']
    filename= f.filename
    target = os.path.join(APP_ROOT, 'images/')
    # print(target)
    des = "/".join([target, filename])
    f.save(des)

    test_image_path = ('images\\'+filename)
    test_image= load_img(test_image_path, target_size=(300,300))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    predicted_class= class_name[np.argmax(prediction[0])]
    result = str(predicted_class)
    carb_level, category = findRipenessFunc(test_image_path)
    print(result)
    return jsonify({'fruit': result, 'cat': category, 'carb':carb_level})

if __name__ == '__main__':
    app.run()
