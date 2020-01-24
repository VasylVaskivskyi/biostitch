"""This module performs estimation of image overlap or shift
using phase correlation from OpenCV.
It contains class AdaptiveShiftEstimation
that can handle manual and automatically scanned image data sets.

"""


from copy import deepcopy
from itertools import chain
import numpy as np
import pandas as pd
import cv2 as cv


class AdaptiveShiftEstimation:
    def __init__(self):
        self._scan = ''
        self._micro_ids = None
        self._micro_x_size = None
        self._micro_y_size = None
        self._ids_in_clusters = []
        self._default_image_shape = (0, 0)

    def estimate(self, images):
        self._default_image_shape = images[0].shape
        if self._scan == 'auto':
            ids, x_size, y_size = self.estimate_image_sizes_scan_auto(images)
            return ids, x_size, y_size
        elif self._scan == 'manual':
            x_size, y_size = self.estimate_image_sizes_scan_manual(images)
            return self._micro_ids, x_size, y_size

    def estimate_image_sizes_scan_manual(self, images):
        x_size = self.find_translation_x(images)
        y_size = self.find_translation_y(images)
        return x_size, y_size

    def median_error_cor(self, df, axis):
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
            ncols = len(dataframe.columns)
            col_medians = list(dataframe.median(axis=0, skipna=True))
            for i in range(0, ncols):
                if pd.isna(col_medians[i]):
                    dataframe.iloc[:, i] = np.nan
                else:
                    dataframe.iloc[:, i] = int(round(col_medians[i]))

        return dataframe

    def find_pairwise_shift(self, img1, img2, overlap, mode):
        if mode == 'horizontal':
            if overlap >= self._default_image_shape[1]:
                return 0

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
            if overlap >= self._default_image_shape[0]:
                return 0

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


    def find_shift_series(self, images, id_series, size_series, mode):
        ids = id_series.copy()
        res = id_series.copy()
        res[:] = np.nan
        res.iloc[0] = size_series[0]
        if mode == 'horizontal':
            axis = 1
        elif mode == 'vertical':
            axis = 0
        add_prcnt = int(round(self._default_image_shape[axis] * 0.01))

        for i in range(1, len(ids)):
            if ids.iloc[i] == 'zeros':
                res.iloc[i] = np.nan
            elif ids.iloc[i - 1] == 'zeros':
                res.iloc[i - 1] = np.nan
            else:
                img1 = ids.iloc[i - 1]
                img2 = ids.iloc[i]

                overlap = int(round((self._default_image_shape[axis] - size_series.iloc[i]) + add_prcnt))
                res.iloc[i] = self.find_pairwise_shift(images[img1], images[img2], overlap, mode)

        return res

    def find_translation_x(self, images):
        ids = self._micro_ids
        x_size = ids.copy()
        x_size.loc[:, :] = 0.0
        nrows, ncols = x_size.shape

        for i in range(0, nrows):
            x_size.iloc[i, :] = self.find_shift_series(images, ids.iloc[i, :], self._micro_x_size.iloc[i, :], 'horizontal')
        x_size = self.median_error_cor(x_size, axis=0)
        x_size.iloc[:, 0] = self._default_image_shape[1]

        col_means = list(x_size.mean(axis=0))
        for i in range(0, ncols):
            x_size.iloc[:, i] = int(round(col_means[i]))
            
        x_size = x_size.astype(np.int32)
        return x_size

    def find_translation_y(self, images):
        ids = self._micro_ids
        y_size = ids.copy()
        y_size.loc[:, :] = 0.0
        nrows, ncols = y_size.shape

        for i in range(0, ncols):
            y_size.iloc[:, i] = self.find_shift_series(images, ids.iloc[:, i], self._micro_y_size.iloc[:, i], 'vertical')
        y_size = self.median_error_cor(y_size, axis=1)
        y_size.iloc[0, :] = self._default_image_shape[0]
        
        row_means = list(y_size.mean(axis=1))
        for i in range(0, nrows):
            y_size.iloc[i, :] = int(round(row_means[i]))

        y_size = y_size.astype(np.int32)
        return y_size

    # ----------- Estimation of auto scanned images -----------------

    def estimate_image_sizes_scan_auto(self, images):
        # size from microscope xml metadata
        ids_in_clusters = self._ids_in_clusters
        micro_ids = self._micro_ids
        micro_x_sizes = self._micro_x_size
        micro_y_sizes = self._micro_y_size

        rows_in_clusters = []
        for c in range(0, len(ids_in_clusters)):
            this_cluster_rows = []
            for row in micro_ids:
                if row[0] in ids_in_clusters[c]:
                    this_cluster_rows.append(row)
            rows_in_clusters.append(set(this_cluster_rows))
        print(rows_in_clusters)
        ids = []
        x_sizes = []
        y_sizes = []
        for cluster in rows_in_clusters:
            micro_ids_sub = []
            micro_x_sizes_sub = []
            micro_y_sizes_sub = []
            for row in range(0, len(micro_ids)):
                if micro_ids[row][0] in cluster:
                    micro_ids_sub.append(micro_ids[row])
                    micro_x_sizes_sub.append(micro_x_sizes[row])
                    micro_y_sizes_sub.append(micro_y_sizes[row])
            print('\n', micro_ids_sub)
            print('\n', micro_x_sizes_sub)
            print('\n', micro_y_sizes_sub)
            this_cluster_x_sizes, this_cluster_y_sizes = self.calculate_image_sizes_scan_auto(images, micro_ids_sub, micro_x_sizes_sub, micro_y_sizes_sub)
            ids.append(micro_ids_sub)
            x_sizes.append(this_cluster_x_sizes)
            y_sizes.append(this_cluster_y_sizes)
        return ids, x_sizes, y_sizes

    def calculate_image_sizes_scan_auto(self, images, micro_ids, micro_x_sizes, micro_y_sizes):
        # estimate row width and height
        est_x_sizes = []
        est_y_sizes = []

        # for each row of images find horizontal shift between images
        nrows = len(micro_ids)
        for row in range(0, nrows):
            this_row_ids = micro_ids[row] # image ids in the list
            this_row_x_sizes_from_micro = micro_x_sizes[row] # image size from microscope meta to calculate overlap
            this_row_x_sizes = self.find_translation_x_scan_auto(images, this_row_ids, this_row_x_sizes_from_micro)
            est_x_sizes.append(this_row_x_sizes)

        # calculating zero padding for image rows
        max_row_width = max([sum(row) for row in est_x_sizes])
        for row in range(0, len(est_x_sizes)):
            est_x_sizes[row][-1] += (max_row_width - sum(est_x_sizes[row]))

        # error correction using sizes from microscopy metadata
        x_sizes = self.remapping_micro_param(micro_ids, micro_x_sizes, est_x_sizes, mode='x')
        #x_sizes = est_x_sizes

        # iteratively compare images from two rows
        # use only combinations without zero padding or gap images
        for row in range(1, nrows):
            prev_row_ids = micro_ids[row - 1]
            this_row_ids = micro_ids[row]
            combinations = zip(prev_row_ids, this_row_ids)
            valid_combinations = [comb for comb in combinations if 'zeros' not in comb]

            this_row_y_size = []
            for comb in valid_combinations:
                prev_row_img_id = comb[0]
                this_row_img_id = comb[1]
                this_row_y_size.append(self.find_translation_y_scan_auto(images[prev_row_img_id], images[this_row_img_id], micro_y_sizes[row]))
            est_y_sizes.append(int(round(np.median(this_row_y_size))))
        est_y_sizes.append(self._default_image_shape[0])
        y_sizes_arr = []
        for row in range(0, len(est_y_sizes)):
            y_sizes_arr.append( [est_y_sizes[row]] * len(est_x_sizes[row]) )

        y_sizes = self.remapping_micro_param(micro_ids, micro_y_sizes, y_sizes_arr, mode='y')
        #y_sizes = y_sizes_arr
        return x_sizes, y_sizes

    def find_translation_x_scan_auto(self, images, ids, x_sizes):
        res = [0] * len(ids)
        res[0] = x_sizes[0]
        add_prcnt = int(round(self._default_image_shape[1] * 0.01))
        # in each row first picture is zero padding
        for i in range(1, len(ids)):
            if ids[i] == 'zeros':
                res[i] = x_sizes[i]
            elif ids[i - 1] == 'zeros':
                res[i] = self._default_image_shape[1]
            else:
                img1 = ids[i - 1]
                img2 = ids[i]
                overlap = int(round((self._default_image_shape[1] - x_sizes[i]) + add_prcnt))
                res[i] = int(round(self.find_pairwise_shift(images[img1], images[img2], overlap, 'horizontal')))

        return res

    def find_translation_y_scan_auto(self, img1, img2, y_sizes):
        # overlap is the same across the row, so we take only first element of the row
        add_prcnt = int(round(self._default_image_shape[1] * 0.01))
        overlap = int(round(self._default_image_shape[0] - y_sizes[0] + add_prcnt))
        y_size = self.find_pairwise_shift(img1, img2, overlap, 'vertical')
        return y_size

    def remapping_micro_param(self, ids, micro_arr, est_arr, mode):
        result_arr = deepcopy(micro_arr)

        idx_to_exclude = []
        for row in range(0, len(ids)):
            idx_to_exclude.append([_id for _id in range(0, len(ids[row])) if ids[row][_id] == 'zeros'])

        no_zeros_x_sizes = []
        no_zeros_micro_sizes = []
        for row in range(0, len(ids)):
            no_zeros_x_sizes.append([el for i, el in enumerate(est_arr[row]) if i not in idx_to_exclude[row]])
            no_zeros_micro_sizes.append([el for i, el in enumerate(micro_arr[row]) if i not in idx_to_exclude[row]])

        # create list of tuples with corresponding values
        # from list with microscopy sizes and estimated sizes
        corr = []
        for i in range(0, len(ids)):
            corr.append(list(zip(no_zeros_micro_sizes[i], no_zeros_x_sizes[i])))

        # flatten the list of the lists
        flat_corr = list(chain.from_iterable(corr))

        # create reference dictionary, where keys are micro sizes and values are estimated sizes
        # in such way each microscopy size can have several estimated sizes
        ref_dict = {}
        for tup in flat_corr:
            if tup[0] in ref_dict:
                ref_dict[tup[0]] += [tup[1]]
            else:
                ref_dict[tup[0]] = [tup[1]]

        # take median of values for each key
        corr_dict = {}
        for key, val in ref_dict.items():
            if len(val) > 1:
                corr_dict[key] = int(round(np.median(val)))
            else:
                corr_dict[key] = val[0]


        # replace values in the list of microscopy sizes with medians of estimated sizes
        if mode == 'y':
            for row in range(0, len(result_arr)):
                for x in range(0, len(result_arr[row])):
                    this_size = result_arr[row][x]
                    if this_size in corr_dict:
                        result_arr[row][x] = corr_dict[this_size]

        elif mode == 'x':
            for row in range(0, len(result_arr)):
                for x in range(0, len(result_arr[row])):
                    this_size = result_arr[row][x]
                    if this_size in corr_dict and x not in idx_to_exclude[row]:
                        result_arr[row][x] = corr_dict[this_size]

            # for x sizes update right zero padding
            max_row_width = max([sum(row) for row in result_arr])
            for row in range(0, len(result_arr)):
                result_arr[row][-1] += (max_row_width - sum(result_arr[row]))

        return result_arr

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

    @property
    def ids_in_clusters(self):
        return self._ids_in_clusters
    @ids_in_clusters.setter
    def ids_in_clusters(self, value):
        self._ids_in_clusters = value