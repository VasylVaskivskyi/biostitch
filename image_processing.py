import os
import tifffile as tif
import cv2 as cv
import numpy as np
import dask
import re


def alphaNumOrder(string):
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return ''.join([format(int(x), '05d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])


def read_images(path, is_dir):
    """ Rread images in natural order (with respect to numbers) """

    allowed_extensions = ('tif', 'tiff')

    if is_dir:
        file_list = [fn for fn in os.listdir(path) if fn.endswith(allowed_extensions)]
        file_list.sort(key=alphaNumOrder)
        task = [dask.delayed(tif.imread)(path + fn) for fn in file_list]
        img_list = dask.compute(*task)
        #img_list = list(map(tif.imread, [path + fn for fn in file_list]))
    else:
        if type(path) == list:
            task = [dask.delayed(tif.imread)(p) for p in path]
            img_list = dask.compute(*task)
            #img_list = list(map(tif.imread, path))
        else:
            img_list = tif.imread(path)

    return img_list


def remove_bg(image, image_shape=None):
    if image_shape == None:
        kernel_size1 = [max(image.shape) // 10] * 2
        kernel_size2 = [kernel_size1[0] // 2, kernel_size1[1] // 2]
    else:
        kernel_size1 = (image_shape[1] // 10, image_shape[0] // 10)
        kernel_size2 = [kernel_size1[0] // 2, kernel_size1[1] // 2]
    kernel_size1 = tuple(i if i % 2 != 0 else i + 1 for i in kernel_size1)
    kernel_size2 = tuple(i if i % 2 != 0 else i + 1 for i in kernel_size2)

    kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size1)
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size2)
    img = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    cor = cv.erode(img, kernel_erode, None)
    cor = cv.dilate(cor, kernel_dilate, None)
    cor = cv.GaussianBlur(cor, (0, 0), kernel_size2[0], None, kernel_size2[1])
    res = img - cor
    result = cv.normalize(res, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)
    return result


def equalize_histograms(img_list, contrast_limit=127, grid_size=(41, 41)):
    """ Function for adaptive normalization of image histogram CLAHE """
    nrows, ncols = img_list[0].shape
    grid_size = [int(round(max((ncols, nrows)) / 20))] * 2
    grid_size = tuple(i if i % 2 != 0 else i + 1 for i in grid_size)
    contrast_limit = 256

    clahe = cv.createCLAHE(contrast_limit, grid_size)
    task = [dask.delayed(clahe.apply)(img) for img in img_list]
    img_list = dask.compute(*task)
    #img_list = list(map(clahe.apply, img_list))
    clahe.collectGarbage()
    return img_list


def z_project(field):
    """ Wrapper function to support multiprocessing """
    return np.max(np.stack(read_images(field, is_dir=False), axis=0), axis=0)


def create_z_projection_for_fov(channel_name, path_list):
    """ Read images, convert them into stack, get max z-projection"""
    channel = path_list[channel_name]
    task = [dask.delayed(z_project)(field) for field in channel]
    z_max_img_list = dask.compute(*task)

    #for field in channel:
    #   z_max_img_list.append(np.max(np.stack(read_images(field, is_dir=False), axis=0), axis=0))
    return z_max_img_list


def stitch_z_projection(channel_name, fields_path_list, ids, x_size, y_size, do_illum_cor, scan_mode):
    """ Create max z projection for each field of view """
    z_max_fov_list = create_z_projection_for_fov(channel_name, fields_path_list)
    
    if do_illum_cor:
        z_max_fov_list = equalize_histograms(z_max_fov_list)
        z_proj = stitch_images(z_max_fov_list, ids, x_size, y_size, scan_mode)
    else:
        z_proj = stitch_images(z_max_fov_list, ids, x_size, y_size, scan_mode)
    
    return z_proj


def crop_images_scan_manual(images, ids, x_sizes, y_sizes):
    """ Read data from dataframe ids, series x_sizes and y_sizes and crop images """
    x_sizes = x_sizes.to_list()
    y_sizes = y_sizes.to_list()
    ids = ids.to_list()
    default_img_shape = images[0].shape
    r_images = []
    for j, _id in enumerate(ids):
        if _id == 'zeros':
            img = np.zeros((y_sizes[j], x_sizes[j]), dtype=np.uint16)
        else:
            x_shift = default_img_shape[1] - x_sizes[j]
            y_shift = default_img_shape[0] - y_sizes[j]

            img = images[_id][y_shift:, x_shift:]
        r_images.append(img)
    return r_images


def crop_images_scan_auto(images, ids, x_sizes, y_sizes):
    default_img_shape = images[0].shape
    r_images = []
    for j, _id in enumerate(ids):
        if _id == 'zeros':
            img = np.zeros((y_sizes, x_sizes[j]), dtype=np.uint16)
        else:
            x_shift = default_img_shape[1] - x_sizes[j]
            y_shift = default_img_shape[0] - y_sizes
            img = images[_id][y_shift:, x_shift:]

        r_images.append(img)

    return r_images


def stitch_images(images, ids, x_size, y_size, scan_mode):
    """ Stitch cropped images by concatenating them horizontally and vertically """
    if scan_mode == 'auto':
        plane_width = sum(x_size[0])
        plane_height = sum(y_size)
        res = np.zeros((plane_height, plane_width), dtype=np.uint16)
        nrows = len(y_size)
        y_pos_plane = list(np.cumsum(y_size))
        y_pos_plane.insert(0, 0)

        for row in range(0, nrows):
            f = y_pos_plane[row]  # from
            t = y_pos_plane[row + 1]  # to
            res[f:t, :] = np.concatenate(crop_images_scan_auto(images, ids[row], x_size[row], y_size[row]), axis=1)
    elif scan_mode == 'manual':
        plane_width = sum(x_size.iloc[0, :])
        plane_height = sum(y_size.iloc[:, 0])
        res = np.zeros((plane_height, plane_width), dtype=np.uint16)
        nrows = ids.shape[0]
        y_pos_plane = list(np.cumsum(y_size.iloc[:, 0]))
        y_pos_plane.insert(0, 0)

        for row in range(0, nrows):
            f = y_pos_plane[row]  # from
            t = y_pos_plane[row + 1]  # to
            res[f:t, :] = np.concatenate(crop_images_scan_manual(images, ids.iloc[row, :], x_size.iloc[row, :], y_size.iloc[row, :]), axis=1)
    return res


def stitch_plane(plane_paths, clahe, ids, x_size, y_size, do_illum_cor, scan_mode):
    """ Do histogram equalization and stitch multiple images into one plane """
    img_list = read_images(plane_paths, is_dir=False)
    if do_illum_cor:
        img_list = list(map(clahe.apply, img_list))
        clahe.collectGarbage()
        result_plane = stitch_images(img_list, ids, x_size, y_size, scan_mode)
    else:
        result_plane = stitch_images(img_list, ids, x_size, y_size, scan_mode)
    return result_plane


def stitch_plane2(plane_paths, clahe, ids, x_size, y_size, do_illum_cor, scan_mode):
    """ Do histogram normalization and stitch multiple images into one plane """
    img_list = read_images(plane_paths, is_dir=False)
    ncols = sum(x_size.iloc[0, :])
    nrows = sum(y_size.iloc[:, 0])
    result_plane = np.zeros((1, nrows, ncols), dtype=np.uint16)
    if do_illum_cor:
        img_list = list(map(clahe.apply, img_list))
        clahe.collectGarbage()
        result_plane[0, :, :] = stitch_images(img_list, ids, x_size, y_size, scan_mode)
    else:
        result_plane[0, :, :] = stitch_images(img_list, ids, x_size, y_size, scan_mode)
    return result_plane


def stitch_series_of_planes(channel, planes_path_list, ids, x_size, y_size, do_illum_cor, scan_mode):
    """ Stitch planes into one channel """
    ncols = sum(x_size.iloc[0, :])
    nrows = sum(y_size.iloc[:, 0])
    nplanes = len(planes_path_list[channel])

    grid_size = [int(round(max((ncols, nrows)) / 20))] * 2
    grid_size = tuple(i if i % 2 != 0 else i + 1 for i in grid_size)
    contrast_limit = 256
    clahe = cv.createCLAHE(contrast_limit, grid_size)

    result_channel = np.zeros((nplanes, nrows, ncols), dtype=np.uint16)
    delete = '\b'*20
    for i, plane in enumerate(planes_path_list[channel]):
        print('{0}plane {1}/{2}'.format(delete, i+1, nplanes), end='', flush=True)
        result_channel[i, :, :] = stitch_plane(plane, clahe, ids, x_size, y_size, do_illum_cor, scan_mode)
    print('\n')
    #tif.imwrite(img_out_dir + channel + '.tif', final_image)
    return result_channel
