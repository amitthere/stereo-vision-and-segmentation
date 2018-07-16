import cv2
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

left_img = cv2.imread('view1.png', cv2.IMREAD_GRAYSCALE)  #read it as a grayscale image
right_img = cv2.imread('view5.png', cv2.IMREAD_GRAYSCALE)

#Disparity Computation for Left Image

#OcclusionCost = 20 (You can adjust this, depending on how much threshold you want to give for noise)
OcclusionCost = 20

rows, cols = left_img.shape

d_map_l = np.zeros(left_img.shape)
d_map_r = np.zeros(right_img.shape)


def optimal_match():
    for x in range(rows):

        CostMatrix = np.zeros((cols+1, cols+1))
        DirectionMatrix = np.zeros((cols+1, cols+1))

        for i in range(1, cols+1):
            CostMatrix[i, 0] = i * OcclusionCost
            CostMatrix[0, i] = i * OcclusionCost

        for r in range(1, cols+1):
            for c in range(1, cols+1):
                min1 = CostMatrix[r-1, c-1] + np.abs(left_img[x, r-1] - right_img[x, c-1])
                min2 = CostMatrix[r-1, c] + OcclusionCost
                min3 = CostMatrix[r, c-1] + OcclusionCost

                CostMatrix[r, c] = min([min1, min2, min3])
                cmin = CostMatrix[r, c]

                if cmin == min1 :
                    DirectionMatrix[r, c] = 1
                elif cmin == min2 :
                    DirectionMatrix[r, c] = 2
                elif cmin == min3 :
                    DirectionMatrix[r, c] = 3

        p = cols
        q = cols

        while (p != 0 and q != 0):
            if DirectionMatrix[p, q] == 1 :
                p = p - 1
                q = q - 1
                d_map_l[x][p] = np.abs(p - q)
                d_map_r[x][q] = np.abs(p - q)
            elif DirectionMatrix[p][q] == 2 :
                p = p - 1
                d_map_l[x][p] = np.abs(p - q)
            elif DirectionMatrix[p][q] == 3 :
                q = q - 1
                d_map_r[x][q] = np.abs(p - q)


def show_image(title, image):
    max_val = image.max()
    # image = np.absolute(image)
    image = np.divide(image, max_val)
    # cv2.imshow(title, image)
    cv2.imwrite(title+str(random.randint(1, 100))+'.jpg', image*255)


def main():
    optimal_match()
    show_image('D-Map_Left_using_DP_', d_map_l)
    show_image('D-Map_Right_using_DP_', d_map_r)


main()





