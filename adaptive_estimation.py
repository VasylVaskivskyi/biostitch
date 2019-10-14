import numpy as np
import pandas as pd
from skimage.feature import register_translation

np.set_printoptions(suppress=True)  # use normal numeric notation instead of exponential
pd.set_option('display.width', 1000)


class AdaptiveShiftEstimation:
    def __init__(self):
        self._horizontal_overlap = 0.1
        self._vertical_overlap = 0.1
        self._default_image_shape = (0, 0)

    def estimate_image_sizes(self, images, ids, horizontal_overlap_percent, vertical_overlap_percent):
        self._horizontal_overlap = int(images[0].shape[1] * horizontal_overlap_percent)
        self._vertical_overlap = int(images[0].shape[0] * vertical_overlap_percent)
        self._default_image_shape = images[0].shape
        x_size = self.find_translation_x(images, ids)
        y_size = self.find_translation_y(images, ids)
        return x_size, y_size

    def remove_outliers(self, df, axis):
        """ Compute score for all values in dataframe
            and remove those values which have more than 1 std from median"""
        dataframe = df.round(0).copy()
        _std = dataframe.std(axis=axis, skipna=True, numeric_only=True).mean(skipna=True)  # mean std of all cols
        _median = dataframe.median(axis=axis, skipna=True, numeric_only=True).median(skipna=True)  # median of column medians
        score = abs(dataframe - _median) / (_std + 0.000001)  # prevent division by 0
        df[score > 1] = np.nan
        return df

    def find_pairwise_shift(self, img1, img2, overlap, mode):
        if mode == 'horizontal':
            img1_overlap = img1.shape[1] - overlap
            img2_overlap = overlap
            shift, error, diffphase = register_translation(img1[:, img1_overlap:], img2[:, :img2_overlap], 100)
            pairwise_shift = overlap - shift[1]
        elif mode == 'vertical':
            img1_overlap = img1.shape[0] - overlap
            img2_overlap = overlap
            shift, error, diffphase = register_translation(img1[img1_overlap:, :], img2[:img2_overlap, :], 100)
            pairwise_shift = overlap - shift[0]
        return pairwise_shift

    def find_shift_series(self, images, id_series, mode):
        ids = id_series.copy()
        res = id_series.copy()
        res[:] = np.nan
        for i in range(0, len(ids) - 1):
            if ids.iloc[i] == 'zeros':
                res.iloc[i] = np.nan
            elif ids.iloc[i + 1] == 'zeros':
                res.iloc[i + 1] = np.nan
            else:
                img1 = ids.iloc[i]
                img2 = ids.iloc[i + 1]

                if mode == 'horizontal':
                    overlap = self._horizontal_overlap
                elif mode == 'vertical':
                    overlap = self._vertical_overlap

                res.iloc[i + 1] = self.find_pairwise_shift(images[img1], images[img2], overlap, mode)

        return res

    def find_translation_x(self, images, ids):
        x_size = ids.copy()
        x_size.loc[:, :] = 0.0
        nrows, ncols = x_size.shape

        for i in range(0, nrows):
            x_size.iloc[i,:] = self.find_shift_series(images, ids.iloc[i, :], 'horizontal')
        x_size = self.remove_outliers(x_size, axis=0)
        x_size = self._default_image_shape[1] - x_size
        x_size.iloc[:, 0] = self._default_image_shape[1]

        col_means = list(x_size.mean(axis=0))

        for i in range(0, ncols):
            x_size.iloc[:,i] = int(round(col_means[i]))
        x_size = x_size.astype(np.int64)
        return x_size

    def find_translation_y(self, images, ids):
        y_size = ids.copy()
        y_size.loc[:, :] = 0.0
        nrows, ncols = y_size.shape

        for i in range(0, ncols):
            y_size.iloc[:, i] = self.find_shift_series(images, ids.iloc[:, i], 'vertical')
        y_size = self.remove_outliers(y_size, axis=1)
        y_size = self._default_image_shape[0] - y_size
        y_size.iloc[0, :] = self._default_image_shape[0]

        row_means = list(y_size.mean(axis=1))

        for i in range(0, nrows):
            y_size.iloc[i,:] = int(round(row_means[i]))
        y_size = y_size.astype(np.int64)
        return y_size
