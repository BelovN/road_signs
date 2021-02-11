import cv2 as cv
import math
import numpy as np
import sys

from PIL import Image, ImageFilter


RELATION_REC = 1.5
LENGTH_REC = 30
MAX_LENGTH = 750
SPACE_EPS = 50

hsv_min = np.array((0, 77, 17), np.uint8)
hsv_max = np.array((208, 255, 255), np.uint8)


def load_image(image_location):
    return Image.open(image_location)


def add_image_filters(img):
    img = img.filter(ImageFilter.BoxBlur(2.5))
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    filtered_img = np.array(img)
    return filtered_img


def convert_to_cv2(img):
    img_cv2 = np.array(img)
    img_cv2 = img_cv2[:, :, ::-1].copy()
    return img_cv2


def rotate_rectangle(points):
    X = []
    Y = []

    for i in points:
        X.append(i[0])
        Y.append(i[1])

    x_max = max(X)
    x_min = min(X)
    y_max = max(Y)
    y_min = min(Y)

    return np.array([
            [x_max,y_max],
            [x_max,y_min],
            [x_min,y_min],
            [x_min,y_max]
        ])


def rotate_all_rectangles(rectangles):
    list = []
    for rect in rectangles:
        rect = rotate_rectangle(rect)
        list.append(rect)
    return list


def crop_rectangles(points, img):
    X = []
    Y = []

    for i in points:
        X.append(i[0])
        Y.append(i[1])

    x_max = max(X)
    x_min = min(X)
    y_max = max(Y)
    y_min = min(Y)

    height, width = img.shape[:2]

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if y_max > height:
        y_max = height
    if x_max > width:
        x_max = width

    coordinates = [max(X), min(X), max(Y), min(Y)]
    croped_image = img[y_min:y_max , x_min:x_max]
    return croped_image


def detect_rectangles(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)
    contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for cnt in contours0:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        k = length_vector(box[0], box[1])
        m = length_vector(box[0], box[3])

        if (k > LENGTH_REC) and (m > LENGTH_REC) and (k < MAX_LENGTH) and (m < MAX_LENGTH):
            if k > m:
                if k / m <= RELATION_REC :
                    rect = [box]
                    cv.drawContours(img, rect, 0, (0,0,255), 3)
                    rectangles.append(rect[0])
            else:
                if m / k <= RELATION_REC:
                    rect = [box]
                    cv.drawContours(img, rect, 0, (0,0,255), 3)
                    rectangles.append(rect[0])

    cv.imwrite("output\\rectangles.png", img)
    return rectangles


def length_vector(a,b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def find_diag(points):
    A = points[0]
    B = points[1]
    C = points[2]
    D = points[3]

    m = length_vector(A,B)
    k = length_vector(B,C)

    diag = math.sqrt(m * m + k * k)

    return diag


def find_max_coordinate(rectangle):
    max_coordinate = [0, 0]
    for point in rectangle:
        if point[0] > max_coordinate[0] or point[1] > max_coordinate[1]:
            max_coordinate = point

    return max_coordinate


def compare_rects(A, B):
    t = find_max_coordinate(A)
    n = find_max_coordinate(B)

    if t[0] == n[0] and t[1] == n[1]:
        return True

    return False


def check_rects(rectangles):
    clear_rects = []
    list = []

    for rect_1 in rectangles:
        for rect_2 in rectangles:
            A = find_max_coordinate(rect_1)
            B = find_max_coordinate(rect_2)
            if not compare_rects(rect_1, rect_2):

                y1, x1, z1 = rect_1[1]

                A1 = length_vector(x1, z1)
                B1 = length_vector(x1, y1)

                max1 = find_max_coordinate(rect_1)

                point_1 = [max1[0] - A1, max1[1] - B1]

                x2 = rect_2[1]
                y2 = rect_2[0]
                z2 = rect_2[2]

                B2 = length_vector(x2, y2)
                A2 = length_vector(x2, z2)

                max2 = find_max_coordinate(rect_2)

                point_2 = [max2[0] - A2, max2[1] - B2]

                length = length_vector(point_1, point_2)

                if length < SPACE_EPS:
                    list.append([rect_1, rect_2])

    for rect in rectangles:
        flag = False
        for rectangles_ in list:
            if compare_rects(rectangles_[0], rect) or compare_rects(rectangles_[1], rect):
                flag = True
        if not flag:
            clear_rects.append(rect)

    return list, clear_rects


def find_signs(fn):
    img = load_image(fn)
    add_image_filters(img)
    img = convert_to_cv2(img)

    rectangles = detect_rectangles(img)
    rectangles = rotate_all_rectangles(rectangles)
    cross_rects, rectangles_ = check_rects(rectangles)

    img_original = load_image(fn)
    img_original = convert_to_cv2(img_original)

    list_signs = []
    list_cross = []

    rects = []

    for rect in rectangles_:
        croped_img = crop_rectangles(rect, img_original)
        resized_img = cv.resize(croped_img, (48, 48))
        list_signs.append(resized_img)
        rects.append(rect)

    for rect in cross_rects:
        croped_img_1 = crop_rectangles(rect[0], img_original)
        resized_img_1 = cv.resize(croped_img_1, (48, 48))
        croped_img_2 = crop_rectangles(rect[1], img_original)
        resized_img_2 = cv.resize(croped_img_2, (48, 48))
        list_cross.append([resized_img_1, resized_img_2, rect[0], rect[1]])

    return list_cross, list_signs, rectangles_, cross_rects
