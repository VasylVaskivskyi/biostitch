import os
import tifffile as tif
import cv2 as cv
import numpy as np
from utilities import alphaNumOrder
import dask

def equalize_histograms(img_list: list, contrast_limit: int = 127, grid_size: (int, int)= (41, 41)) -> list:
    """ function for adaptive normalization of image histogram CLAHE """

    clahe = cv.createCLAHE(contrast_limit, grid_size)
    task = [dask.delayed(clahe.apply(img)) for img in img_list]
    img_list = dask.compute(task)[0]

    #img_list = list(map(clahe.apply, img_list))

    return img_list


def read_images(path: [str, list], is_dir: bool) -> list:
    """read images in natural order (with respect to numbers)"""

    allowed_extensions = ('tif', 'tiff')

    if is_dir == True:
        file_list = [fn for fn in os.listdir(path) if fn.endswith(allowed_extensions)]
        file_list.sort(key=alphaNumOrder)
        task = [dask.delayed(path + fn) for fn in file_list]
        img_list = dask.compute(task)[0]
        #img_list = list(map(tif.imread, [path + fn for fn in file_list]))
    else:
        if type(path) == list:
            task = [dask.delayed(tif.imread(p)) for p in path]
            img_list = dask.compute(task)[0]
            #img_list = list(map(tif.imread, path))
        else:
            img_list = tif.imread(path)

    return img_list


def plus1(a):
    return [a+1]


def create_z_projection_for_initial_stitching(main_channel: str, path_list: dict) -> list:
    channel = path_list[main_channel]
    # read images, convert them into stack, get max projection
    z_max_img_list = []
    task = [dask.delayed(np.max(np.stack(read_images(field, is_dir=False), axis=0), axis=0)) for field in channel]
    z_max_img_list = dask.compute(task)[0]
    #for field in channel:
    #   z_max_img_list.append(np.max(np.stack(read_images(field, is_dir=False), axis=0), axis=0))

    return z_max_img_list




def preprocess_images(img_dir: str) -> list:
    """Function wrapper that imports images and does preprocessing(histogram equalization)"""

    img_list = read_images(img_dir)
    equalized_images = equalize_histograms(img_list)
    return equalized_images
