import numpy as np
import pandas as pd
import cv2 as cv
#from skimage.feature import register_translation
np.set_printoptions(suppress=True)  # use normal numeric notation instead of exponential
pd.set_option('display.width', 1000)


class AdaptiveShiftEstimation:
    def __init__(self):
        self._horizontal_overlap_percent = 0.1
        self._vertical_overlap_percent = 0.1
        self._horizontal_overlap = 0
        self._vertical_overlap = 0
        self._default_image_shape = (0, 0)

    @property
    def horizontal_overlap_percent(self):
        return self._horizontal_overlap_percent

    @horizontal_overlap_percent.setter
    def horizontal_overlap_percent(self, value):
        self._horizontal_overlap_percent = value

    @property
    def vertical_overlap_percent(self):
        return self._vertical_overlap_percent

    @vertical_overlap_percent.setter
    def vertical_overlap_percent(self, value):
        self._vertical_overlap_percent = value

    def estimate(self, images, parameters, scan_mode):
        self._horizontal_overlap = int(round(images[0].shape[1] * self.horizontal_overlap_percent))
        self._vertical_overlap = int(round(images[0].shape[0] * self.vertical_overlap_percent))
        self._default_image_shape = images[0].shape
        if scan_mode == 'auto':
            x_size, y_size = self.estimate_sizes_scan_auto(images, parameters)
        elif scan_mode == 'manual':
            x_size, y_size = self.estimate_image_sizes_scan_manual(images, parameters)
        return x_size, y_size

    def estimate_image_sizes_scan_manual(self, images, ids):
        x_size = self.find_translation_x(images, ids)
        y_size = self.find_translation_y(images, ids)
        return x_size, y_size

    def use_median(self, df, axis):
        """ Replace all values in rows or cols with respective medians"""
        dataframe = df.copy()
        if axis == 1:
            nrows = len(dataframe.index)
            row_medians = list(dataframe.median(axis=1, skipna=True))
            for i in range(0, nrows):
                if pd.isna(row_medians[i]):
                    dataframe.iloc[i, :] = np.nan
                else:
                    dataframe.iloc[i, :] = int(round(row_medians[i]))
        elif axis == 0:
            ncols = len(dataframe.index)
            col_medians = list(dataframe.median(axis=0, skipna=True))
            for i in range(0, ncols):
                if pd.isna(col_medians[i]):
                    dataframe.iloc[i, :] = np.nan
                else:
                    dataframe.iloc[i, :] = int(round(col_medians[i]))

        return dataframe

    def find_pairwise_shift(self, img1, img2, overlap, mode):
        if mode == 'horizontal':
            img1_overlap = img1.shape[1] - overlap
            img2_overlap = overlap
            part1 = cv.normalize(img1[:, img1_overlap:], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
            part2 = cv.normalize(img2[:, :img2_overlap], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
            shift, error = cv.phaseCorrelate(part1, part2)
            pairwise_shift = self._default_image_shape[1] - (overlap - (-shift[0]))
        elif mode == 'vertical':
            img1_overlap = img1.shape[0] - overlap
            img2_overlap = overlap
            part1 = cv.normalize(img1[img1_overlap:, :], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
            part2 = cv.normalize(img2[:img2_overlap, :], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
            shift, error = cv.phaseCorrelate(part1, part2)
            pairwise_shift = self._default_image_shape[0] - (overlap - (-shift[1]))

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
        x_size = self.use_median(x_size, axis=0)
        x_size.iloc[:, 0] = self._default_image_shape[1]
        
        col_means = list(x_size.mean(axis=0))
        for i in range(0, ncols):
            x_size.iloc[:, i] = int(round(col_means[i]))
            
        x_size = x_size.astype(np.int64)
        return x_size

    def find_translation_y(self, images, ids):
        y_size = ids.copy()
        y_size.loc[:, :] = 0.0
        nrows, ncols = y_size.shape

        for i in range(0, ncols):
            y_size.iloc[:, i] = self.find_shift_series(images, ids.iloc[:, i], 'vertical')
        y_size = self.use_median(y_size, axis=1)
        y_size.iloc[0, :] = self._default_image_shape[0]
        
        row_means = list(y_size.mean(axis=1))
        for i in range(0, nrows):
            y_size.iloc[i, :] = int(round(row_means[i]))

        y_size = y_size.astype(np.int64)
        return y_size

    # ----------- Estimation of auto scanned images -----------------

    def estimate_sizes_scan_auto(self, images, image_sizes):
        # get maximum row width calculated from microscope coordinates
        # it will serve as initial estimation of zero padding for the right side of the image
        row_width = []
        for row in image_sizes:
            row_width.append(sum([i[0] for i in row]))
        max_row_width = max(row_width)

        # estimate row width and height
        x_sizes = []
        y_sizes = [self._default_image_shape[0]]
        image_rows = []

        nrows = len(image_sizes)
        for row in range(0, nrows - 1):
            print('row', row)

            this_row_ids = [i[2] for i in image_sizes[row]]
            this_row_x_sizes_from_micro = [i[0] for i in image_sizes[row]]
            this_row_x_sizes = self.find_translation_x_scan_auto(images, this_row_ids, this_row_x_sizes_from_micro)
            x_sizes.append(this_row_x_sizes)

            diff = max_row_width - sum(this_row_x_sizes)
            if diff > 0:
                this_row_x_sizes[-1] += diff
            elif diff < 0:
                max_row_width = sum(this_row_x_sizes)
                new_diff = abs(diff)
                for x in x_sizes:
                    x[-1] += new_diff

            image_rows.append(self.stitch_images_x_scan_auto(images, this_row_ids, this_row_x_sizes))

            if row == 0:
                continue
            else:
                this_row_y_size = int(round(self.find_translation_y_scan_auto(image_rows)))
                y_sizes.append(this_row_y_size)

                del image_rows[0]

        return x_sizes, y_sizes

    def find_translation_x_scan_auto(self, images, ids, x_sizes):
        res = [0] * len(ids)

        for i in range(0, len(ids) - 1):
            if ids[i] == 'zeros':
                res[i] = x_sizes[i]
            elif ids[i + 1] == 'zeros':
                res[i + 1] = x_sizes[i]
            else:
                img1 = ids[i]
                img2 = ids[i + 1]

                res[i + 1] = int(round(self.find_pairwise_shift(images[img1], images[img2], self._horizontal_overlap, 'horizontal')))

                if ids[i - 1] == 'zeros':
                    res[i] = self._default_image_shape[1]

        return res

    def stitch_images_x_scan_auto(self, images, ids, x_sizes):
        # cropping
        r_images = []
        for j, _id in enumerate(ids):
            if _id == 'zeros':
                img = np.zeros((self._default_image_shape[0], x_sizes[j]), dtype=np.uint16)
            else:
                x_shift = self._default_image_shape[1] - x_sizes[j]
                img = images[_id][:, x_shift:]

            r_images.append(img)
        # stitching
        res = np.concatenate(r_images, axis=1)

        return res

    def find_translation_y_scan_auto(self, image_rows):
        y_size = self.find_pairwise_shift(image_rows[0], image_rows[1], self._vertical_overlap, 'vertical')
        return y_size

