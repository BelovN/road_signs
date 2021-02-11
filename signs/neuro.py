import csv
import numpy as np
import os
import tensorflow as tf
import numpy as np

from PIL import Image


ROOT_DIR = os.path.abspath(os.curdir)


def load_data(train_img, train_classes, test_img, test_classes):
    convert_data(25000, ROOT_DIR + "\\dataset\\train\\", train_img)
    convert_data(7000, ROOT_DIR + "\\dataset\\test\\", test_img)

    train_img = np.asarray(train_img)
    test_img = np.asarray(test_img)

    f_test = open(ROOT_DIR + "\\dataset\\gt_test.csv")
    f_train = open(ROOT_DIR + "\\dataset\\gt_train.csv")

    gt_test = csv.reader(f_test)
    gt_train = csv.reader(f_train)

    # Пропускаем 1 строку
    next(gt_test)
    next(gt_train)

    for i in range(7000):
        test_classes.append(np.uint8(next(gt_test)[1]))

    for i in range(25000):
        train_classes.append(np.uint8(next(gt_train)[1]))


def convert_data(_r, way, data):
    for i in range(_r):
        s = ""
        i_str = str(i)
        for i in range(6 - len(str(i))):
            s+="0"
        s = s + i_str

        image = Image.open(way + s + ".png").convert("L")
        image_arr = np.array(image)
        data.append(image_arr)
        image.close()

def build(_train_img, _train_classes, _test_img, test_classes, _epochs=3):

    _train_img = tf.keras.utils.normalize(_train_img, axis=1)
    _test_img = tf.keras.utils.normalize(_test_img, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(67, activation=tf.nn.softmax))

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(_train_img, _train_classes, epochs=_epochs)

    val_loss, val_acc=model.evaluate(test_img, test_classes)
    print(val_loss, val_acc)

    return model


def main():
    test_classes = []
    train_classes = []

    train_img = []
    test_img = []

    load_data(train_img, train_classes, test_img, test_classes)

    train_img = np.asarray(train_img)
    test_img = np.asarray(test_img)
    test_classes = np.asarray(test_classes)
    train_classes = np.asarray(train_classes)

    model = build(train_img, train_classes,  test_img, test_classes, 10)

    model.save('road_signs_neu')


if __name__ == '__main__':
    main()
