import cv2
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


grayscale_max = 255


def load_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image


def show_image(title, image):
    max_val = image.max()
    # image = np.absolute(image)
    image = np.divide(image, max_val)
    # cv2.imshow(title, image)
    cv2.imwrite(title+str(random.randint(1, 100))+'.jpg', image*grayscale_max)


def add_padding(input, padding):
    rows = input.shape[0]
    columns = input.shape[1]
    output = np.zeros((rows + padding * 2, columns + padding * 2), dtype=float)
    output[ padding : rows + padding, padding : columns + padding] = input
    return output


def add_replicate_padding(image):
    # zero_padded = add_padding(image, padding)
    # size = image.shape[0]
    top_row = image[0, :]
    image = np.vstack((top_row, image))

    bottom_row = image[-1, :]
    image = np.vstack((image, bottom_row))

    left_column = image[:, 0]
    left_column = np.reshape(left_column, (left_column.shape[0], 1))
    image = np.hstack((left_column, image))

    right_column = image[:, -1]
    right_column = np.reshape(right_column, (right_column.shape[0], 1))
    image = np.hstack((image, right_column))

    return image


def euclid_dist(a, b):
    distance = np.linalg.norm(a - b)
    return distance


def get_search_bounds(column, block_size, width):
    disparity_range = 25
    left_bound = column - disparity_range
    if left_bound < 0:
        left_bound = 0
    right_bound = column + disparity_range
    if right_bound > width:
        right_bound = width - block_size + 1
    return left_bound, right_bound


def search_bounds(column, block_size, width, rshift):
    disparity_range = 75
    padding = block_size // 2
    right_bound = column
    if rshift:
        left_bound = column - disparity_range
        if left_bound < padding:
            left_bound = padding
        step = 1
    else:
        left_bound = column + disparity_range
        if left_bound >= (width - 2*padding):
            left_bound = width - 2*padding - 2
        step = -1
    return left_bound, right_bound, step


# max disparity 30
def disparity_map(left, right, block_size, rshift):

    padding = block_size // 2
    left_img = add_padding(left, padding)
    right_img = add_padding(right, padding)

    height, width = left_img.shape

    # d_map = np.zeros((height - padding*2, width - padding*2), dtype=float)
    d_map = np.zeros(left.shape , dtype=float)

    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):

            bestdist = float('inf')
            shift = 0
            left_pixel = left_img[row:row + block_size, col:col + block_size]
            l_bound, r_bound, step = search_bounds(col, block_size, width, rshift)

            # for i in range(l_bound, r_bound - padding*2):
            for i in range(l_bound, r_bound, step):
                right_pixel = right_img[row:row + block_size, i:i + block_size]

                # if euclid_dist(left_pixel, right_pixel) < bestdist :
                ssd = np.sum((left_pixel - right_pixel) ** 2)
                # print('row:',row,' col:',col,' i:',i,' bestdist:',bestdist,' shift:',shift,' ssd:',ssd)
                if ssd < bestdist:
                    bestdist = ssd
                    shift = i

            if rshift:
                d_map[row, col] = col - shift
            else:
                d_map[row, col] = shift - col
            print('Calculated Disparity at ('+str(row)+','+str(col)+') :', d_map[row,col])

    return d_map


def mean_square_error(disparity_map, ground_truth):
    # ssd = np.sum((disparity_map - ground_truth)**2)
    # mse = ssd/(ground_truth.shape[0]*ground_truth.shape[1])
    mse = np.mean((disparity_map - ground_truth)**2)
    return mse


def consistency_map_mse_l(d_map_left, d_map_right, left_ground_truth):
    rows, cols = d_map_left.shape
    consistency_map = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            left_pixel = d_map_left[r, c]

            if cols > c - left_pixel > 0:
                right_pixel = d_map_right[r, int(c - left_pixel)]
            else:
                right_pixel = d_map_right[r, c]

            if left_pixel == right_pixel:
                consistency_map[r, c] = left_pixel
            else:
                consistency_map[r, c] = 0

    sum = 0
    for r in range(rows):
        for c in range(cols):
            if consistency_map[r, c] != 0:
                sum = sum + (left_ground_truth[r, c] - consistency_map[r, c]) ** 2

    mse_c_left = sum / (rows * cols)
    return mse_c_left, consistency_map


def consistency_map_mse_r(d_map_left, d_map_right, right_ground_truth):
    rows, cols = d_map_right.shape
    consistency_map = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            right_pixel = d_map_right[r, c]

            if c + right_pixel < cols:
                left_pixel = d_map_left[r, int(c + right_pixel)]
            else:
                left_pixel = d_map_left[r, c]

            if right_pixel == left_pixel:
                consistency_map[r, c] = right_pixel
            else:
                consistency_map[r, c] = 0

    sum = 0
    for r in range(rows):
        for c in range(cols):
            if consistency_map[r, c] != 0:
                sum = sum + (right_ground_truth[r, c] - consistency_map[r, c]) ** 2

    mse_c_right = sum / (rows * cols)
    return mse_c_right, consistency_map


def main():
    l = load_image('view1.png')
    r = load_image('view5.png')


    # Disparity Maps
    d_map_lr_3 = disparity_map(l, r, 3, True)
    show_image('D_Map_lr_block3_', d_map_lr_3)

    d_map_rl_3 = disparity_map(r, l, 3, False)
    show_image('D_Map_rl_block3_', d_map_rl_3)

    d_map_lr_9 = disparity_map(l, r, 9, True)
    show_image('D_Map_lr_block9_', d_map_lr_9)

    d_map_rl_9 = disparity_map(r, l, 9, False)
    show_image('D_Map_rl_block9_', d_map_rl_9)


    # Mean Squared Error
    ground_truth_1 = load_image('disp1.png')
    ground_truth_2 = load_image('disp5.png')

    mse_3_lr = mean_square_error(d_map_lr_3, ground_truth_1)
    print('MSE for view1 using block size of 3 is', mse_3_lr)

    mse_3_rl = mean_square_error(d_map_rl_3, ground_truth_2)
    print('MSE for view5 using block size of 3 is', mse_3_rl)

    mse_9_lr = mean_square_error(d_map_lr_9, ground_truth_1)
    print('MSE for view1 using block size of 9 is', mse_9_lr)

    mse_9_rl = mean_square_error(d_map_lr_3, ground_truth_2)
    print('MSE for view5 using block size of 9 is', mse_9_rl)


    # MSE after Consistency Check
    mse_3c_left, c_map_3cl = consistency_map_mse_l(d_map_lr_3, d_map_rl_3, ground_truth_1)
    cv2.imwrite('consistency_map_block3_view1.jpg', c_map_3cl)
    print('MSE for view1 after Consistency check using block size of 3 is', mse_3c_left)

    mse_3c_right, c_map_3cr = consistency_map_mse_r(d_map_lr_3, d_map_rl_3, ground_truth_2)
    cv2.imwrite('consistency_map_block3_view5.jpg', c_map_3cr)
    print('MSE for view5 after Consistency check using block size of 3 is', mse_3c_right)

    mse_9c_left, c_map_9cl = consistency_map_mse_l(d_map_lr_9, d_map_rl_9, ground_truth_1)
    cv2.imwrite('consistency_map_block9_view1.jpg', c_map_9cl)
    print('MSE for view1 after Consistency check using block size of 9 is', mse_9c_left)

    mse_9c_right, c_map_9cr = consistency_map_mse_r(d_map_lr_9, d_map_rl_9, ground_truth_2)
    cv2.imwrite('consistency_map_block9_view5.jpg', c_map_9cr)
    print('MSE for view5 after Consistency check using block size of 9 is', mse_9c_right)

    return


main()
