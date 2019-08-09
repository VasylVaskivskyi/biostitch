import pandas as pd
import cv2 as cv
import numpy as np
from stitch_images import crop_image


def convert_homography_from_saved_table(homography_table):
    homography_list = []
    for row in range(0, len(homography_table)):
        homography_converted = np.array([homography_table.loc[row, [0, 1, 2]].to_list(), homography_table.loc[row, [3, 4, 5]].to_list()])
        homography_list.append(homography_converted)
        #picture_size_list.append(tuple(homography_table.loc[row, [6, 7]].astype(np.uint32).to_list()))
    return homography_list


def import_homography(h_path_horizontal, h_path_vertical):
    homography_horizontal_table = pd.read_csv(h_path_horizontal, sep='\t',  header=None, dtype=np.float32)
    homography_vertical_table = pd.read_csv(h_path_vertical, sep='\t', header=None, dtype=np.float32)

    homography_horizontal_list = convert_homography_from_saved_table(homography_horizontal_table)
    homography_vertical_list = convert_homography_from_saved_table(homography_vertical_table)
    return homography_horizontal_list, homography_vertical_list


def stitch_with_computed_homography_horizontal(left, right, homography):
    dst = cv.warpAffine(right, homography, (right.shape[1] + left.shape[1], right.shape[0]), None)
    dst[:left.shape[0], :left.shape[1]] = left
    return crop_image(dst)


def stitch_with_computed_homography_vertical(top, bottom, homography):
    shift_x = 0
    if homography[0][2] < 0:
        shift_x = int(round(homography[0][2] * -1))
        homography[0][2] = 0.0

    longest = max(bottom.shape[1], top.shape[1])
    dst = cv.warpAffine(bottom, homography, (longest + shift_x, bottom.shape[0] + top.shape[0]), None)
    dst[:top.shape[0], shift_x:top.shape[1] + shift_x] = top
    return crop_image(dst)


def process_images_pairwise_horizontal2(array, homography_horizontal_list, iterator):
    arr = array.copy()  # copy to prevent .pop from modifying original list
    last = None

    # shortcuts
    if len(arr) == 1:
        print(-1)
        return arr
    elif len(arr) == 2:
        print(0)
        x = next(iterator)
        H = homography_horizontal_list[x]
        res = stitch_with_computed_homography_horizontal(arr[0], arr[1], H)
        return res

    if len(arr) % 2 != 0:
        last = arr.pop()

    # create pairs of images
    pairs = []
    for i in range(0, len(arr), 2):
        pairs.append([arr[i], arr[i + 1]])

    # stitch each pair
    res = []
    for p in pairs:
        x = next(iterator)
        H = homography_horizontal_list[x]
        res.append(stitch_with_computed_homography_horizontal(p[0], p[1], H))

    # if there are any elements left unstitched repeat
    if last is None and len(res) < 2:
        print(1)
        return res
    elif last is None and len(res) >= 2:
        print(2)
        return process_images_pairwise_horizontal2(res, homography_horizontal_list, iterator)
    elif last is not None:
        res.append(last)
        print(3)
        return process_images_pairwise_horizontal2(res, homography_horizontal_list, iterator)


def process_images_pairwise_vertical2(array, homography_vertical_list, iterator):
    arr = array.copy()  # copy to prevent .pop from modifying original list
    last = None

    # shortcuts
    if len(arr) == 1:
        print(-1)
        return arr
    elif len(arr) == 2:
        print(0)
        y = next(iterator)
        H = homography_vertical_list[y]
        res = stitch_with_computed_homography_vertical(arr[0], arr[1], H)
        return res

    if len(arr) % 2 != 0:
        last = arr.pop()

    # create pairs of images
    pairs = []
    for i in range(0, len(arr), 2):
        pairs.append([arr[i], arr[i + 1]])

    # stitch each pair
    res = []
    for p in pairs:
        y = next(iterator)
        H = homography_vertical_list[y]
        res.append(stitch_with_computed_homography_vertical(p[0], p[1], H))

    # if there are any elements left unstitched repeat
    if last is None and len(res) < 2:
        print(1)
        return res
    elif last is None and len(res) >= 2:
        print(2)
        return process_images_pairwise_vertical2(res, homography_vertical_list, iterator)
    elif last is not None:
        res.append(last)
        print(3)
        return process_images_pairwise_vertical2(res, homography_vertical_list,iterator)


def get_iter(length):
    for i in range(0, length):
        yield i


def stitch_images2(images, image_positions, homography_horizontal_list, homography_vertical_list):
    print('stitching horizontal')

    # create generator to iterate over the rows of files homography_*.tsv
    x_iter = get_iter(len(homography_horizontal_list))
    y_iter = get_iter(len(homography_vertical_list))

    res_h = []
    j = 0
    for row in image_positions:
        img_ids = [i[2] for i in row]
        img_ids.reverse()
        img_row = [images[i] for i in img_ids]
        print('row ', j, 'images', [i + 1 for i in img_ids])  # print image numbers from 1

        res_h.append(process_images_pairwise_horizontal2(img_row, homography_horizontal_list, x_iter))
        j += 1

    print('stitching vertical')
    res_v = process_images_pairwise_vertical2(res_h, homography_vertical_list, y_iter)
    return res_v


