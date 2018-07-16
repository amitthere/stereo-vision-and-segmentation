import cv2
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# h is window size
h = 30
# ms is mean shift
ms = 20


def load_image():
    image = cv2.imread('Butterfly.jpg', cv2.IMREAD_COLOR)
    return image


def feature_extraction(image):
    rows = image.shape[0]
    cols = image.shape[1]
    reshaped = image.reshape(rows*cols, image.shape[2])

    feature_vector = np.zeros((reshaped.shape[0], reshaped.shape[1] + 2), dtype=float)
    feature_vector[:, : reshaped.shape[1]] = reshaped

    column = np.arange(cols)
    x_coord = np.repeat(np.arange(rows), cols)
    y_coord = np.tile(column, rows)

    feature_vector[:, 3] = x_coord
    feature_vector[:, 4] = y_coord

    return feature_vector


def euclid_dist(a, b):
    distance = np.linalg.norm(a - b)
    return distance


def compare(centroid, pixel):
    if euclid_dist(centroid, pixel) <= h :
        return 1
    else:
        return 0


def form_window(centroid, fv):
    window = np.zeros((fv.shape[0]))
    # for i in range(fv.shape[0]):
    #     window[i] = np.apply_along_axis(compare(centroid), 0, fv)
    window = np.apply_along_axis(compare, 0, fv)
    return window


def get_test_data():
    tdata = np.random.randint(0, 255, (4, 5, 3))
    return tdata


def means_shift_segmentation(fv, segmented_image):
    fg = 0
    while(len(fv) > 0):
        centroid_index = random.randint(0, fv.shape[0])

        if fg == 0:
            centroid = fv[centroid_index]

        window = []
        distance = 0

        window_mask = np.zeros((fv.shape[0]))
        window_mask.dtype = 'int32'
        centroid = random.randint(0, fv.shape[0])
        for r in range(fv.shape[0]):
            window_mask[r] = compare(fv[centroid], fv[r])

        window = np.zeros((np.sum(window_mask), 5))
        i = 0
        for r in range(fv.shape[0]):
            if window_mask[r] == 1:
                window[i] = fv[r]
                i = i + 1
        new_centroid = np.mean(window, axis=0)

        new_ms = compare(new_centroid, centroid)

        if(new_ms < ms):
            for i in range(np.sum(window_mask)):
                x = int(window[i][3])
                y = int(window[i][4])
                segmented_image[x][y][0] = new_centroid[0]
                segmented_image[x][y][1] = new_centroid[1]
                segmented_image[x][y][2] = new_centroid[2]
            fv = np.delete(fv, window_mask, 0)
            fg = 0
        else:
            centroid = new_centroid
            fg = 1
    cv2.imwrite('segmented_butterfly.jpg',segmented_image)
    return


def main():
    image = load_image()
    fv = feature_extraction(image)
    segmented_image = np.zeros(image.shape)

    means_shift_segmentation(fv, segmented_image)
    print('here')
    return


main()
