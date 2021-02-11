import cv2 as cv
import keras as kr
import numpy as np
import os
import tensorflow as tf

from PIL import Image
from shapes import find_signs


ROOT_DIR = os.path.abspath(os.curdir)
EPS = 0.1


def choice_best(model, signs, rects):
    list_of_signs = []
    for i in range(2):
        s = Image.fromarray(signs[0])
        s = s.convert('L')
        s = np.array(s)
        list_of_signs.append(s)
    list_of_signs = np.asarray(list_of_signs)
    data_img = tf.keras.utils.normalize(list_of_signs, axis=1)
    predictions = model.predict(data_img)

    pred_ind_1 = np.argmax(predictions[0])
    pred_num_1 = predictions[0][pred_ind_1]
    pred_ind_2 = np.argmax(predictions[1])
    pred_num_2 = predictions[1][pred_ind_2]

    if pred_num_1 > pred_num_2:

        return [predictions[0], signs[0], rects[0]]
    else:
        return [predictions[1], signs[1], rects[1]]


def count_predictions(cross_signs, clear_signs, model, rectangles_cross, rectangles_clear):

    cross_predictions = []
    rects = []
    clear_predictions = []
    cross_predictions = []
    if len(cross_signs):
        for i in range(0, len(cross_signs),2):
            cross_predictions.append(choice_best(model, cross_signs[i], rectangles_cross[i]))

    if len(clear_signs):
        list_of_signs = []
        for sign in clear_signs:
            s = Image.fromarray(sign)
            s = s.convert('L')
            s = np.array(s)
            list_of_signs.append(s)
        list_of_signs = np.asarray(list_of_signs)

        data_img = tf.keras.utils.normalize(list_of_signs)
        print(len(data_img))
        print(data_img)

        predictions = model.predict(data_img)

        for i in range(len(clear_signs)):
            clear_predictions.append([predictions[i], clear_signs[i], rectangles_clear[i]])

    return clear_predictions  + cross_predictions


def exclude(predictions):
    counter = 0
    list_del = []

    for pred in predictions:
        index = np.argmax(pred[0])
        if pred[0][index] <= EPS:
            list_del.append(counter)
        counter+=1

    for i in range(len(list_del)-1, -1, -1):
        if i in list_del:
            del predictions[i]

    return predictions


def get_xy(a, b):
    return (a[0],int((a[1] + b[1]) / 2))


def predict(fn_in):
    model = tf.keras.models.load_model('road_signs_neu')
    cross_signs, clear_signs, rectangles_clear, rectangles_cross = find_signs(fn_in)
    predictions = count_predictions(cross_signs, clear_signs, model, rectangles_cross, rectangles_clear)
    predictions =  exclude(predictions)
    image = cv.imread(fn_in)

    for rect in predictions:
        cv.drawContours(image, [rect[2]], 0, (0, 255, 0), 3)
        cv.putText(
            img = image,
            text = str(np.argmax(rect[0])),
            org = (rect[2][3][0], rect[2][3][1] + 20),
            fontFace = cv.FONT_HERSHEY_TRIPLEX,
            fontScale = 1,
            color = (0, 255, 0),
            lineType = 2)

    cv.imwrite("output\\sign_output.png", image)
