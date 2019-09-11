import numpy as np
import cv2 as cv
from preprocess_images import read_images


def cut_images(images, ids, x_sizes, y_sizes):
    x_sizes = x_sizes.to_list()
    y_sizes = y_sizes.to_list()
    ids = ids.to_list()
    j = 0
    r_images = []
    for i in ids:
        if i == 'zeros':
            img = np.zeros((y_sizes[j], x_sizes[j]), dtype=np.uint16)
        else:
            x_shift = images[i].shape[0] - x_sizes[j]
            y_shift = images[i].shape[1] - y_sizes[j]

            img = images[i][y_shift:, x_shift:]
        r_images.append(img)
        j += 1
    return r_images



def stitch_images(images, ids, x_size, y_size):
    nrows = ids.shape[0]
    res_h = []
    for row in range(0, nrows):
        res_h.append(
            np.concatenate(cut_images(images, ids.iloc[row, :], x_size.iloc[row, :], y_size.iloc[row, :]), axis=1))

    res = np.concatenate(res_h, axis=0)

    return res


def stitch_plane(plane_paths, clahe, ids, x_size, y_size):
    img_list = read_images(plane_paths, is_dir=False)
    # correct uneven illumination
    images = list(map(clahe.apply, img_list))
    result_plane = stitch_images(images, ids, x_size, y_size)
    clahe.collectGarbage()
    return result_plane


def stitch_big_image(channel, planes_path_list, ids, x_size, y_size, img_out_dir):
    # write channel multilayer image to file
    contrast_limit = 127
    grid_size = (41, 41)
    clahe = cv.createCLAHE(contrast_limit, grid_size)

    ncols = sum(x_size.iloc[0, :])
    nrows = sum(y_size.iloc[:, 0])
    nplanes = len(planes_path_list[channel])

    result_channel = np.zeros((nplanes, nrows, ncols), dtype=np.uint16)
    delete = '\b'*20
    j = 0
    for plane in planes_path_list[channel]:
        print('{0}plane {1}/{2}'.format(delete, j+1, nplanes), end='', flush=True)
        result_channel[j, :, :] = stitch_plane(plane, clahe, ids, x_size, y_size)
        j += 1

    #tif.imwrite(img_out_dir + channel + '.tif', final_image)
    return result_channel
