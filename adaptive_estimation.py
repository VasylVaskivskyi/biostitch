import numpy as np
import pandas as pd
import cv2 as cv
from copy import deepcopy
from itertools import chain
#from skimage.feature import register_translation
np.set_printoptions(suppress=True)  # use normal numeric notation instead of exponential
pd.set_option('display.width', 1000)


class AdaptiveShiftEstimation:
    def __init__(self):
        self._scan = ''
        self._micro_ids = None
        self._micro_x_size = None
        self._micro_y_size = None
        self._default_image_shape = (0, 0)
        self._x_overlap = None
        self._y_overlap = None

    @property
    def scan(self):
        return self._scan

    @scan.setter
    def scan(self, value):
        self._scan = value

    @property
    def micro_ids(self):
        return self._micro_ids

    @micro_ids.setter
    def micro_ids(self, value):
        self._micro_ids = value

    @property
    def micro_x_size(self):
        return self._micro_x_size

    @micro_x_size.setter
    def micro_x_size(self, value):
        self._micro_x_size = value

    @property
    def micro_y_size(self):
        return self._micro_x_size

    @micro_y_size.setter
    def micro_y_size(self, value):
        self._micro_y_size = value

    def estimate(self, images):
        self._default_image_shape = images[0].shape
        self.estimate_overlap()
        if self._scan == 'auto':
            x_size, y_size = self.estimate_sizes_scan_auto(images)
        elif self._scan == 'manual':
            x_size, y_size = self.estimate_image_sizes_scan_manual(images)
        return x_size, y_size

    def estimate_image_sizes_scan_manual(self, images):
        x_size = self.find_translation_x(images, self._micro_ids)
        y_size = self.find_translation_y(images, self._micro_ids)
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
            hor_shift = shift[0]
            if hor_shift < 0:
                pairwise_shift = self._default_image_shape[1] - (overlap + hor_shift)
            else:
                pairwise_shift = self._default_image_shape[1] - hor_shift
        elif mode == 'vertical':
            img1_overlap = img1.shape[0] - overlap
            img2_overlap = overlap
            part1 = cv.normalize(img1[img1_overlap:, :], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
            part2 = cv.normalize(img2[:img2_overlap, :], None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
            shift, error = cv.phaseCorrelate(part1, part2)
            ver_shift = shift[1]
            if ver_shift < 0:
                pairwise_shift = self._default_image_shape[0] - (overlap + ver_shift)
            else:
                pairwise_shift = self._default_image_shape[0] - ver_shift

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

    def estimate_overlap(self):
        # x overlap
        def_shape = self._default_image_shape
        ids = self._micro_ids

        x_cor = int(round(def_shape[1] * 0.01))   # add 1 percent
        y_cor = int(round(def_shape[0] * 0.01))  # add 1 percent

        if self._scan == 'auto':
            x_overlap = []
            for i, row in enumerate(self._micro_x_size):
                this_row_overlap = []
                for j, el in enumerate(row):
                    if ids[i][j] != 'zeros':
                        this_row_overlap.append(x_cor + (def_shape[1] - el))
                    else:
                        this_row_overlap.append(0)
                x_overlap.append(this_row_overlap)

            # y overlap
            y_overlap = []
            for i, row in enumerate(self._micro_y_size):
                this_row_overlap = []
                for j, el in enumerate(row):
                    if ids[i][j] != 'zeros':
                        this_row_overlap.append(y_cor + (def_shape[0] - el))
                    else:
                        this_row_overlap.append(0)
                y_overlap.append(this_row_overlap)

        elif self._scan == 'manual':
            # data frames
            zeros = ids == 'zeros'
            x_overlap = (def_shape[1] - self._micro_x_size) + x_cor
            y_overlap = (def_shape[0] - self._micro_x_size) + y_cor

            x_overlap[zeros] = 0
            y_overlap[zeros] = 0


        self._x_overlap = x_overlap
        self._y_overlap = y_overlap

    def estimate_sizes_scan_auto(self, images, micro_img_sizes):
        # size from microscope xml metadata
        micro_ids = []
        micro_x_sizes = []
        micro_y_sizes = []
        for row in micro_img_sizes:
            micro_x_sizes.append([i[0] for i in row])
            micro_y_sizes.append([i[1] for i in row])
            micro_ids.append([i[2] for i in row])
        # estimate row width and height
        est_x_sizes = []
        est_y_sizes = [self._default_image_shape[0]]

        nrows = len(micro_ids)
        for row in range(0, nrows):

            this_row_ids = micro_ids[row]
            this_row_x_sizes_from_micro = micro_x_sizes[row]
            this_row_x_sizes = self.find_translation_x_scan_auto(images, this_row_ids, this_row_x_sizes_from_micro)
            est_x_sizes.append(this_row_x_sizes)

        max_row_width = max([sum(row) for row in est_x_sizes])

        for row in range(0, len(est_x_sizes)):
            est_x_sizes[row][-1] += (max_row_width - sum(est_x_sizes[row]))

        x_sizes = self.remapping_micro_param(micro_x_sizes, est_x_sizes, mode='x')
        
        # iteratively compare images from two rows
        # use only combinations without zero padding or gap images
        for row in range(0, nrows - 1):
            this_row_ids = micro_ids[row]
            next_row_ids = micro_ids[row + 1]
            combinations = zip(this_row_ids, next_row_ids)
            valid_combinations = [comb for comb in combinations if 'zeros' not in comb]
            
            next_row_y_size = []
            for comb in valid_combinations:
                this_row_img_id = comb[0]
                next_row_img_id = comb[1]
                next_row_y_size.append(self.find_translation_y_scan_auto(images[this_row_img_id], images[next_row_img_id]))
            est_y_sizes.append(int(round(np.median(next_row_y_size))))

        y_sizes_arr = []
        for row in range(0, len(est_x_sizes)):
            y_sizes_arr.append( [est_y_sizes[row]] * len(est_x_sizes[row]) )

        #y_sizes = self.remapping_micro_param(micro_y_sizes, y_sizes_arr, mode='y')
        y_sizes = y_sizes_arr
        return x_sizes, y_sizes

    def find_translation_x_scan_auto(self, images, ids, x_sizes, row):
        res = [0] * len(ids)

        for i in range(0, len(ids) - 1):
            if ids[i] == 'zeros':
                res[i] = x_sizes[i]
            elif ids[i + 1] == 'zeros':
                res[i + 1] = x_sizes[i + 1]
            else:
                img1 = ids[i]
                img2 = ids[i + 1]

                res[i + 1] = int(round(self.find_pairwise_shift(images[img1], images[img2], self._x_overlap[row][i+1], 'horizontal')))

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

    def find_translation_y_scan_auto(self, img1, img2, row):
        y_size = self.find_pairwise_shift(img1, img2, self._y_overlap[row][0], 'vertical')
        return y_size

    def remapping_micro_param(self, micro_arr, est_arr, mode):
        result_arr = deepcopy(micro_arr)
        corr = []
        for i in range(0, len(est_arr)):
            corr.append(list(zip(result_arr[i], est_arr[i])))

        flat_cor = list(chain.from_iterable(corr))

        ref_dict = {}
        for t in flat_cor:
            if t[0] in ref_dict:
                ref_dict[t[0]] += [t[1]]
            else:
                ref_dict[t[0]] = [t[1]]

        corr_dict = {}
        for key, val in ref_dict.items():
            if len(val) > 1:
                corr_dict[key] = int(round(np.median(val)))
            else:
                corr_dict[key] = val[0]

        for row in range(0, len(result_arr)):
            for x in range(0, len(result_arr[row])):
                this_size = result_arr[row][x]
                if this_size in corr_dict:
                    result_arr[row][x] = corr_dict[this_size]

        if mode == 'x':
            max_row_width = max([sum(row) for row in result_arr])
            for row in range(0, len(result_arr)):
                result_arr[row][-1] += (max_row_width - sum(result_arr[row]))

        return result_arr
