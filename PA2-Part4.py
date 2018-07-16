import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


view1 = cv2.imread('view1.png')
view5 = cv2.imread('view5.png')

disp1 = cv2.imread('disp1.png', cv2.IMREAD_GRAYSCALE)
disp5 = cv2.imread('disp5.png', cv2.IMREAD_GRAYSCALE)


def view_synthesis():
    height, width, rgb = view1.shape
    view3 = np.zeros(view1.shape, dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            mid = disp1[row, col] / 2
            if col - mid < 0:
                continue
            view3[row, int(col - mid)] = view1[row, col]

    cv2.imwrite('view3_partial.jpg', view3)

    for row in range(height):
        for col in range(width):
            mid = disp5[row, col] / 2
            if col + mid >= width:
                continue
            if view3[row, int(col + mid)].all() == 0:
                view3[row, int(col + mid)] = view5[row, col]

    cv2.imwrite('view3_synthesized.jpg', view3)
    return view3


def main():
    view3 = view_synthesis()
    return


main()
