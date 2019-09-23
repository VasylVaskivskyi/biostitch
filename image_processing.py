import os
from mod_lib_tifffile import tifffile as tif
import cv2 as cv
import numpy as np
import dask


def alphaNumOrder(string):
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return ''.join([format(int(x), '05d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])


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


def equalize_histograms(img_list: list, contrast_limit: int = 127, grid_size: (int, int)= (41, 41)) -> list:
    """ function for adaptive normalization of image histogram CLAHE """

    clahe = cv.createCLAHE(contrast_limit, grid_size)
    task = [dask.delayed(clahe.apply(img)) for img in img_list]
    img_list = dask.compute(*task)
    #img_list = list(map(clahe.apply, img_list))

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


def crop_images(images, ids, x_sizes, y_sizes):
    """read data from dataframe ids, series x_sizes and y_sizes and crop images"""
    x_sizes = x_sizes.to_list()
    y_sizes = y_sizes.to_list()
    ids = ids.to_list()
    j = 0
    r_images = []
    for i in ids:
        if i == 'zeros':
            img = np.zeros((y_sizes[j], x_sizes[j]), dtype=np.uint16)
        else:
            x_shift = images[i].shape[1] - x_sizes[j]
            y_shift = images[i].shape[0] - y_sizes[j]

            img = images[i][y_shift:, x_shift:]
        r_images.append(img)
        j += 1
    return r_images


def stitch_images(images, ids, x_size, y_size):
    """stitch cropped images"""
    nrows = ids.shape[0]
    res_h = []
    for row in range(0, nrows):
        res_h.append(
            np.concatenate(crop_images(images, ids.iloc[row, :], x_size.iloc[row, :], y_size.iloc[row, :]), axis=1))

    res = np.concatenate(res_h, axis=0)

    return res


def stitch_plane(plane_paths, clahe, ids, x_size, y_size, do_illum_cor):
    """do histogram normalization and stitch multiple images into one plane"""
    img_list = read_images(plane_paths, is_dir=False)
    if do_illum_cor == True:
        images = list(map(clahe.apply, img_list))
        result_plane = stitch_images(images, ids, x_size, y_size)
        clahe.collectGarbage()
    else:
        result_plane = stitch_images(img_list, ids, x_size, y_size)
    return result_plane

def stitch_plane2(plane_paths, clahe, ids, x_size, y_size, do_illum_cor):
    """do histogram normalization and stitch multiple images into one plane"""
    img_list = read_images(plane_paths, is_dir=False)
    ncols = sum(x_size.iloc[0, :])
    nrows = sum(y_size.iloc[:, 0])
    result_plane = np.zeros((1, nrows,ncols), dtype=np.uint16)
    if do_illum_cor == True:
        images = list(map(clahe.apply, img_list))
        result_plane[0,:,:] = stitch_images(images, ids, x_size, y_size)
        clahe.collectGarbage()
    else:
        result_plane[0,:,:] = stitch_images(img_list, ids, x_size, y_size)
    return result_plane


def stitch_series_of_planes(channel, planes_path_list, ids, x_size, y_size, do_illum_cor):
    """stitch planes into one channel"""
    contrast_limit = 127
    grid_size = (41, 41)
    clahe = cv.createCLAHE(contrast_limit, grid_size)

    ncols = sum(x_size.iloc[0, :])
    nrows = sum(y_size.iloc[:, 0])
    nplanes = len(planes_path_list[channel])

    result_channel = np.zeros((nplanes, nrows, ncols), dtype=np.uint16)
    delete = '\b'*20
    for i, plane in enumerate(planes_path_list[channel]):
        print('{0}plane {1}/{2}'.format(delete, i+1, nplanes), end='', flush=True)
        result_channel[i, :, :] = stitch_plane(plane, clahe, ids, x_size, y_size, do_illum_cor)
    print('\n')
    #tif.imwrite(img_out_dir + channel + '.tif', final_image)
    return result_channel
