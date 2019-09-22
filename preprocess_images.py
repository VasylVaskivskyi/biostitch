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
    img_list = dask.compute(*task)
    #img_list = list(map(clahe.apply, img_list))

    return img_list


def read_images(path: [str, list], is_dir: bool) -> list:
    """read images in natural order (with respect to numbers)"""

    allowed_extensions = ('tif', 'tiff')

    if is_dir == True:
        file_list = [fn for fn in os.listdir(path) if fn.endswith(allowed_extensions)]
        file_list.sort(key=alphaNumOrder)
        task = [dask.delayed(path + fn) for fn in file_list]
        img_list = dask.compute(*task)
        #img_list = list(map(tif.imread, [path + fn for fn in file_list]))

    else:
        if type(path) == list:
            task = [dask.delayed(tif.imread(p)) for p in path]
            img_list = dask.compute(*task)
            #img_list = list(map(tif.imread, path))

        else:
            img_list = tif.imread(path)

    return img_list


def z_project(field):
    """wrapper function to support multiprocessing"""
    return np.max(np.stack(read_images(field, is_dir=False), axis=0), axis=0)


def create_z_projection_for_fov(channel_name: str, path_list: dict) -> list:
    """ read images, convert them into stack, get max z-projection"""
    channel = path_list[channel_name]
    task = [dask.delayed(z_project(field)) for field in channel]
    z_max_img_list = dask.compute(*task)

    #for field in channel:
    #   z_max_img_list.append(np.max(np.stack(read_images(field, is_dir=False), axis=0), axis=0))
    return z_max_img_list

def create_z_projection(channel_name, fields_path_list, ids, x_size, y_size, do_illum_cor):
    z_max_img_list = create_z_projection_for_fov(channel_name, fields_path_list)
    
    if do_illum_cor == True:
        images = equalize_histograms(z_max_img_list)
        z_proj = stitch_images(images, ids, x_size, y_size)
    else:
        z_proj = stitch_images(z_max_img_list, ids, x_size, y_size)
    
    return z_proj
