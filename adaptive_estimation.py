import numpy as np
import pandas as pd
from skimage.feature import register_translation

np.set_printoptions(suppress=True)  # use normal numeric notation instead of exponential
pd.set_option('display.width', 1000)


class AdaptiveShiftEstimation:
    def __init__(self):
        self.__horizontal_overlap = 0
        self.__vertical_overlap = 0
        self.__default_image_shape = (0, 0)

    def estimate_image_sizes(self, images, ids, horizontal_overlap_percent, vertical_overlap_percent):
        self.__horizontal_overlap = int(images[0].shape[1] * horizontal_overlap_percent)
        self.__vertical_overlap = int(images[0].shape[0] * vertical_overlap_percent)
        self.__default_image_shape = images[0].shape
        x_size = self.find_translation_x(images, ids)
        #image_rows = self.stitch_images_x(images, ids, x_size)
        y_size = self.find_translation_y(images, ids)
        return x_size, y_size

    def find_translation_hor(self, left, right, overlap):
        left_overlap = left.shape[1] - overlap
        right_overlap = overlap
        shift, error, diffphase = register_translation(left[:, left_overlap:], right[:, :right_overlap], 100)
        shift = overlap - shift[1]
        return shift


    def find_translation_row(self, images, row):
        res = row.copy()
        res[:] = np.nan
        img_ids = list(row[row != 'zeros'])
        img_locs = list(row[row != 'zeros'].index)

        for i in range(0, len(img_ids)):
            left = img_ids[i]
            if i < len(img_ids) - 1:
                right = img_ids[i + 1]
                res[img_locs[i+1]] = self.find_translation_hor(images[left], images[right], self.__horizontal_overlap)
            else:
                return res

        return res


    def normalize_by_cols(self, x_size):
        _std = x_size.std(axis=0).mean()  # std of all table
        _mean = x_size.mean(axis=0).mean()  # column mean
        diff = abs(x_size - _mean)
        x_size[diff > _std] = np.nan
        return x_size

    def normalize_by_rows(self, y_size):
        _std = y_size.std(axis=1).mean()  # std of all table
        _mean = y_size.mean(axis=1).mean()  # column mean
        diff = abs(y_size - _mean)
        y_size[diff > _std] = np.nan
        return y_size


    def find_translation_x(self, images, ids):
        x_size = ids.copy()
        x_size.loc[:, :] = 0.0
        nrows, ncols = x_size.shape

        for i in range(0, nrows):
            x_size.iloc[i,:] = self.find_translation_row(images, ids.iloc[i, :])
        x_size = self.normalize_by_cols(x_size)
        x_size = self.__default_image_shape[1] - x_size
        x_size.iloc[:,0] = self.__default_image_shape[1]
        col_means = list(x_size.mean(axis=0))

        for i in range(0, ncols):
            x_size.iloc[:,i] = round(col_means[i])

        return x_size


    def crop_images(self, images, ids, x_size):
        x_size = x_size.to_list()
        y_size = images[0].shape[0]
        ids = ids.to_list()
        j = 0
        r_images = []
        for i in ids:
            if i == 'zeros':
                img = np.zeros((y_size, x_size[j]), dtype=np.uint16)
            else:
                x_shift = images[i].shape[0] - x_size[j]
                img = images[i][:, x_shift:]
            r_images.append(img)
            j += 1
        return r_images


    def stitch_images_x(self, images, ids, x_size):
        nrows = ids.shape[0]
        res_h = []
        for row in range(0, nrows):
            res_h.append(
                np.concatenate(self.crop_images(images, ids.iloc[row, :], x_size.iloc[row, :]), axis=1))

        return res_h


    def find_translation_ver(self, top, bottom, overlap):
        top_overlap = top.shape[0] - overlap
        bottom_overlap = overlap
        shift, error, diffphase = register_translation(top[top_overlap:, :], bottom[:bottom_overlap, :], 100)
        shift = overlap - shift[0]
        return shift

    def find_translation_col(self, images, col):
        res = col.copy()
        res[:] = np.nan
        img_ids = list(col[col != 'zeros'])
        img_locs = list(col[col != 'zeros'].index)
        for i in range(0, len(img_ids)):
            top = img_ids[i]
            if i < len(img_ids) - 1:
                bottom = img_ids[i + 1]
                res[img_locs[i + 1]] = self.find_translation_ver(images[top], images[bottom], self.__vertical_overlap)
            else:
                return res

        return res

    def find_translation_y(self, images, ids):
        y_size = ids.copy()
        y_size.loc[:, :] = 0.0
        nrows, ncols = y_size.shape

        for i in range(0, ncols):
            y_size.iloc[:,i] = self.find_translation_col(images, ids.iloc[:, i])
        y_size = self.normalize_by_rows(y_size)
        y_size = self.__default_image_shape[0] - y_size
        y_size.iloc[0,:] = self.__default_image_shape[0]
        row_means = list(y_size.mean(axis=1))

        for i in range(0, nrows):
            y_size.iloc[i,:] = round(row_means[i])

        return y_size
    """
    def find_translation_y(self, image_rows, x_size):
        y_size_li = []

        y_size = pd.DataFrame(data=0, columns=x_size.columns, index=x_size.index, dtype=np.int64)

        for i in range(0, len(image_rows)):
            if i < len(image_rows) - 1:
                y_translation = int(round(self.find_translation_ver(image_rows[i], image_rows[i + 1], self.__vertical_overlap) ))
                y_translation = self.__default_image_shape[0] - y_translation
                y_size_li.append(y_translation)

        y_size_li.insert(0, self.__default_image_shape[0])
        for i, row in enumerate(y_size.index):
            y_size.loc[row, :] = int(y_size_li[i])

        return y_size
    """

    def stitch_reference_channel(self, image_rows, y_size):

        shape_x = image_rows[0].shape[1]
        y_size_li = y_size.iloc[:, 0].to_list()
        shape_y = sum(y_size_li)

        y_cum_sizes = list(np.cumsum(y_size_li, axis=0, dtype=np.int64))

        fin_img = np.zeros((shape_y, shape_x), dtype=np.uint16)
        fin_img[:y_cum_sizes[0], :] = image_rows[0]

        for i in range(1, len(image_rows)):
            y_shift = image_rows[i].shape[0] - y_size_li[i]
            fin_img[y_cum_sizes[i-1]: y_cum_sizes[i], :] = image_rows[i][y_shift:, :]

        return fin_img

