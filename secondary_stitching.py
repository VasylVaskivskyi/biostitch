import pandas as pd
import cv2 as cv
import numpy as np


def convert_homography_from_saved_table(homography_table):
    homography_list = []
    picture_size_list = []
    for row in range(0, len(homography_table)):
        homography_converted = np.array([homography_table.loc[row, [0, 1, 2]].to_list(), homography_table.loc[row, [3, 4, 5]].to_list()])
        homography_list.append(homography_converted)
        picture_size_list.append(tuple(homography_table.loc[row, [7, 6]].astype(np.uint32).to_list()))
    return homography_list, picture_size_list


def import_homography(h_path_horizontal, h_path_vertical):
    homography_horizontal_table = pd.read_csv(h_path_horizontal, sep='\t',  header=None)
    homography_vertical_table = pd.read_csv(h_path_vertical, sep='\t', header=None)

    homography_horizontal_list, picture_size_horizontal_list = convert_homography_from_saved_table(homography_horizontal_table)
    homography_vertical_list, picture_size_vertical_list = convert_homography_from_saved_table(homography_vertical_table)
    return homography_horizontal_list, picture_size_horizontal_list, homography_vertical_list, picture_size_vertical_list


def stitch_with_computed_homography_horizontal(left, right, homography, picture_size):
    H = homography.copy()
    dst = cv.warpAffine(right, H, picture_size, None)
    dst[:left.shape[0], :left.shape[1]] = left
    return dst


def stitch_with_computed_homography_vertical(top, bottom, homography, picture_size):
    H = homography.copy()
    shift_x = 0
    if H[0][2] < 0:
        shift_x = int(round(H[0][2] * -1))
        H[0][2] = 0.0

    #longest = max(bottom.shape[1], top.shape[1])
    dst = cv.warpAffine(bottom, H, picture_size, None)
    dst[:top.shape[0], shift_x:top.shape[1] + shift_x] = top
    return dst


def process_images_pairwise_horizontal2(array, homography_horizontal_list, picture_size_horizontal_list, iterator):
    arr = array.copy()  # copy to prevent .pop from modifying original list
    last = None

    # shortcuts
    if len(arr) == 1:
        return arr
    elif len(arr) == 2:
        x = next(iterator)
        H = homography_horizontal_list[x]
        image_size = picture_size_horizontal_list[x]
        res = stitch_with_computed_homography_horizontal(arr[0], arr[1], H, image_size)
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
        image_size = picture_size_horizontal_list[x]
        res.append(stitch_with_computed_homography_horizontal(p[0], p[1], H, image_size))

    # if there are any elements left unstitched repeat
    if last is None and len(res) < 2:
        return res
    elif last is None and len(res) >= 2:
        return process_images_pairwise_horizontal2(res, homography_horizontal_list, picture_size_horizontal_list, iterator)
    elif last is not None:
        res.append(last)
        return process_images_pairwise_horizontal2(res, homography_horizontal_list, picture_size_horizontal_list, iterator)


def process_images_pairwise_vertical2(array, homography_vertical_list, picture_size_vertical_list, iterator):
    arr = array.copy()  # copy to prevent .pop from modifying original list
    last = None

    # shortcuts
    if len(arr) == 1:
        return arr
    elif len(arr) == 2:
        y = next(iterator)
        H = homography_vertical_list[y]
        image_size = picture_size_vertical_list[y]
        res = stitch_with_computed_homography_vertical(arr[0], arr[1], H, image_size)
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
        image_size = picture_size_vertical_list[y]
        res.append(stitch_with_computed_homography_vertical(p[0], p[1], H, image_size))

    # if there are any elements left unstitched repeat
    if last is None and len(res) < 2:
        return res
    elif last is None and len(res) >= 2:
        return process_images_pairwise_vertical2(res, homography_vertical_list, picture_size_vertical_list, iterator)
    elif last is not None:
        res.append(last)
        return process_images_pairwise_vertical2(res, homography_vertical_list, picture_size_vertical_list, iterator)


def get_iter(length):
    for i in range(0, length):
        yield i

def stitch_images2(images, image_positions, homography_horizontal_list, picture_size_horizontal_list, homography_vertical_list, picture_size_vertical_list):
    #print('stitching horizontal')

    # create generator to iterate over the rows of files homography_*.tsv
    x_iter = get_iter(len(homography_horizontal_list))
    y_iter = get_iter(len(homography_vertical_list))

    res_h = []
    j = 0
    for row in image_positions:
        img_ids = [i[2] for i in row]
        img_ids.reverse()
        img_row = [images[i] for i in img_ids]
        #print('row ', j, 'images', [i + 1 for i in img_ids])  # print image numbers from 1

        res_h.append(process_images_pairwise_horizontal2(img_row, homography_horizontal_list, picture_size_horizontal_list, x_iter))
        j += 1

    #print('stitching vertical')
    res_v = process_images_pairwise_vertical2(res_h, homography_vertical_list, picture_size_vertical_list, y_iter)
    return res_v




"""
from utilities import crop_image
from preprocess_images import read_images, equalize_histograms
from get_image_positions import get_image_postions

h_path_horizontal = 'C:/Users/vv3/Desktop/image/stitched/homography_horizontal.tsv'
h_path_vertical = 'C:/Users/vv3/Desktop/image/stitched/homography_vertical.tsv'
homography_horizontal_list, picture_size_horizontal_list, homography_vertical_list, picture_size_vertical_list = import_homography(h_path_horizontal, h_path_vertical)


img_dir = 'C:/Users/vv3/Desktop/image/out_zmax/'
img_list = read_images(img_dir, is_dir=True)
images = equalize_histograms(img_list)

xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
image_positions = get_image_postions(xml_path)

tif.imsave('C:/Users/vv3/Desktop/image/stitched/res1.tif', res_v)
"""