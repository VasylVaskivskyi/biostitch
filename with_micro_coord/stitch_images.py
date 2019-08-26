import numpy as np


def cut_images2(images, ids, x_sizes, y_sizes):
    x_sizes = x_sizes.to_list()
    #y_sizes = y_sizes.to_list()
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
