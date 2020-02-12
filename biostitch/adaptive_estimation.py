"""This module performs estimation of image overlap or shift
using phase correlation from OpenCV.
It contains class AdaptiveShiftEstimation
that can handle manual and automatically scanned image data sets.

"""

from typing import List, Tuple, Union, Optional
from copy import deepcopy
from itertools import chain
import numpy as np
import pandas as pd
import cv2 as cv
from .my_types import Image, DF, Series


class AdaptiveShiftEstimation:
    def __init__(self):
        self._scan = ''
        self._micro_ids = None
        self._micro_x_size = None
        self._micro_y_size = None
        self._ids_in_clusters = []
        self._default_image_shape = (0, 0)

    def estimate(self, images: List[Image]) -> Union[Tuple[DF, DF, DF], Tuple[list, list, list]]:
        self._default_image_shape = images[0].shape
        if self._scan == 'auto':
            ids, x_size, y_size = self.estimate_image_sizes_scan_auto(images)
            return ids, x_size, y_size
        elif self._scan == 'manual':
            x_size, y_size = self.estimate_image_sizes_scan_manual(images)
            ids = pd.DataFrame(self._micro_ids)
            for j in ids.columns:
                for i in ids.index:
                    try:
                        val = ids.loc[i, j]
                        val = int(val)
                        ids.loc[i, j] = val
                    except ValueError:
                        pass
            return pd.DataFrame(self._micro_ids), pd.DataFrame(x_size), pd.DataFrame(y_size)

    def estimate_image_sizes_scan_manual(self, images: List[Image]) -> Tuple[DF, DF]:
        x_size = self.find_shift_x_scan_manual(images)
        y_size = self.find_shift_y_scan_manual(images)
        return x_size, y_size

    def median_error_cor(self, array: np.array, mode: str) -> np.array:
        """ Replace all values in rows or cols with respective medians"""
        arr = array.copy()
        if mode == 'row':
            nrows = arr.shape[0]
            row_medians = list(np.nanmedian(arr, axis=1))
            for i in range(0, nrows):
                arr[i, :] = int(round(row_medians[i]))
        elif mode == 'col':
            ncols = arr.shape[1]
            col_medians = list(np.nanmedian(arr, axis=0))
            for i in range(0, ncols):
                arr[:, i] = int(round(col_medians[i]))

        return arr

    def find_pairwise_shift(self, img1: Image, img2: Image, overlap: int, mode: str) -> int:
        """ Finds size of the img2

        """
        if mode == 'row':
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
        elif mode == 'col':
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

    def find_shift_row_col(self, images: List[Image], id_list: list, size_list: list, mode: str) -> list:
        if mode == 'row':
            axis = 1
        elif mode == 'col':
            axis = 0

        ids = id_list
        res = [np.nan] * len(ids)


        additional_space = int(round(self._default_image_shape[axis] * 0.01))

        for i in range(0, len(ids)-1):
            if ids[i] == 'zeros':
                res[i] = np.nan
            elif ids[i + 1] == 'zeros':
                res[i + 1] = np.nan
            else:
                img1 = int(ids[i])
                img2 = int(ids[i+1])

                overlap = int(round((self._default_image_shape[axis] - size_list[i]) + additional_space))
                res[i] = self.find_pairwise_shift(images[img1], images[img2], overlap, mode)
        res[-1] = self._default_image_shape[axis]
        return res

    def find_shift_x_scan_manual(self, images: List[Image]) -> DF:
        ids = self._micro_ids
        x_size = ids.copy()
        x_size[:, :] = 0.0
        x_size = x_size.astype(np.float32)

        nrows, ncols = x_size.shape
        for i in range(0, nrows):
            x_size[i, :] = self.find_shift_row_col(images, ids[i, :].tolist(), self._micro_x_size[i, :].tolist(), 'row')
        x_size = self.median_error_cor(x_size, 'col')
        x_size = x_size.astype(np.int32)
        return x_size

    def find_shift_y_scan_manual(self, images: List[Image]) -> DF:
        ids = self._micro_ids
        y_size = ids.copy()
        y_size[:, :] = 0.0
        y_size = y_size.astype(np.float32)

        nrows, ncols = y_size.shape
        for i in range(0, ncols):
            y_size[:, i] = self.find_shift_row_col(images, ids[:, i].tolist(), self._micro_y_size[:, i].tolist(), 'col')
        y_size = self.median_error_cor(y_size, 'row')
        y_size = y_size.astype(np.int32)
        return y_size

    # ----------- Estimation of auto scanned images -----------------

    def estimate_image_sizes_scan_auto(self, images: List[Image]) -> Tuple[list, list, list]:
        # size from microscope xml metadata
        ids_in_clusters = self._ids_in_clusters
        micro_ids = self._micro_ids
        micro_x_sizes = self._micro_x_size
        micro_y_sizes = self._micro_y_size
        print('ids_in_clusters\n',self._ids_in_clusters)
        rows_in_clusters = []
        for cluster in ids_in_clusters:
            print('cluster', cluster)
            this_cluster_rows = []
            for row in micro_ids:
                print('row',row)
                first_non_zero = [i for i in row if i != 'zeros'][0]
                if first_non_zero in cluster:
                    print('in cluster', row)
                    this_cluster_rows.append(row)
            rows_in_clusters.append(this_cluster_rows)
        print('rows_in_clusters\n',rows_in_clusters)
        ids = []
        x_sizes = []
        y_sizes = []
    
        for cls in rows_in_clusters:
            cluster = list(chain(*cls))
            micro_ids_sub = []
            micro_x_sizes_sub = []
            micro_y_sizes_sub = []
            for row in range(0, len(micro_ids)):
                if micro_ids[row][1] in cluster and micro_ids[row][1] != 'zeros':
                    micro_ids_sub.append(micro_ids[row])
                    micro_x_sizes_sub.append(micro_x_sizes[row])
                    micro_y_sizes_sub.append(micro_y_sizes[row])

            this_cluster_x_sizes, this_cluster_y_sizes = self.calculate_image_sizes_scan_auto(images, micro_ids_sub, micro_x_sizes_sub, micro_y_sizes_sub)
            ids.extend(micro_ids_sub)
            x_sizes.extend(this_cluster_x_sizes)
            y_sizes.extend(this_cluster_y_sizes)
        return ids, x_sizes, y_sizes

    def calculate_image_sizes_scan_auto(self, images: List[Image], micro_ids: list, micro_x_sizes: list, micro_y_sizes: list) -> Tuple[list, list]:
        # estimate row width and height
        est_x_sizes = []
        est_y_sizes = []

        # for each row of images find horizontal shift between images
        nrows = len(micro_ids)
        for row in range(0, nrows):
            this_row_ids = micro_ids[row]  # image ids in the list
            this_row_x_sizes_from_micro = micro_x_sizes[row]  # image size from microscope meta to calculate overlap
            this_row_x_sizes = self.find_shift_x_scan_auto(images, this_row_ids, this_row_x_sizes_from_micro)
            est_x_sizes.append(this_row_x_sizes)
        
        # calculating zero padding for image rows
        max_row_width = max([sum(row) for row in est_x_sizes])
        for row in range(0, len(est_x_sizes)):
            est_x_sizes[row][-1] += (max_row_width - sum(est_x_sizes[row]))

        # error correction using sizes from microscopy metadata
        x_sizes = self.remapping_micro_param(micro_ids, micro_x_sizes, est_x_sizes, mode='x')
        #x_sizes = est_x_sizes


        x_pos = []
        for x in x_sizes:
            this_row = x.copy()
            this_row.insert(0, 0)
            x_pos.append(np.cumsum(this_row[:-1]))

        for row in range(0, nrows-1):
            prev_row_ids = micro_ids[row]
            this_row_ids = micro_ids[row+1]
            prev_row_x_pos = x_pos[row]
            this_row_x_pos = x_pos[row+1]

            valid_combinations = []
            for i, x in enumerate(this_row_x_pos):
                if this_row_ids[i] == 'zeros':
                    continue
                distance = (x - prev_row_x_pos) ** 2
                min_dist = np.min(distance)
                min_arg = np.argmin(distance)
                if not min_dist > self._default_image_shape[1]**2 and not prev_row_ids[min_arg] == 'zeros':
                    valid_combinations.append((prev_row_ids[min_arg], this_row_ids[i]))
                else:
                    continue
            this_row_y_size = []
            for comb in valid_combinations:
                prev_row_img_id = comb[0]
                this_row_img_id = comb[1]
                this_row_y_size.append(
                    self.find_shift_y_scan_auto(images[prev_row_img_id], images[this_row_img_id], micro_y_sizes[row]))
            est_y_sizes.append(int(round(np.median(this_row_y_size))))
        est_y_sizes.append(self._default_image_shape[0])

        y_sizes_arr = []
        for row in range(0, len(est_y_sizes)):
            y_sizes_arr.append( [est_y_sizes[row]] * len(est_x_sizes[row]) )

        #y_sizes = self.remapping_micro_param(micro_ids, micro_y_sizes, y_sizes_arr, mode='y')
        y_sizes = y_sizes_arr
        return x_sizes, y_sizes

    def find_shift_x_scan_auto(self, images: List[Image], ids: list, x_sizes: list) -> list:
        res = [0] * len(ids)

        add_prcnt = int(round(self._default_image_shape[1] * 0.01))
        # in each row first picture is zero padding
        for i in range(0, len(ids)-1):
            if ids[i] == 'zeros':
                res[i] = x_sizes[i]
            elif ids[i+1] == 'zeros':
                res[i] = self._default_image_shape[1]
            else:
                img1 = ids[i]
                img2 = ids[i+1]
                overlap = int(round((self._default_image_shape[1] - x_sizes[i]) + add_prcnt))
                res[i] = int(round(self.find_pairwise_shift(images[img1], images[img2], overlap, 'row')))
        res[-1] = self._default_image_shape[1]
        return res

    def find_shift_y_scan_auto(self, img1: Image, img2: Image, y_sizes: list) -> int:
        # overlap is the same across the row, so we take only first element of the row
        add_prcnt = int(round(self._default_image_shape[1] * 0.01))
        overlap = int(round(self._default_image_shape[0] - y_sizes[0] + add_prcnt))
        y_size = self.find_pairwise_shift(img1, img2, overlap, 'col')
        return y_size

    def remapping_micro_param(self, ids: list, micro_arr: list, est_arr: list, mode: str) -> list:
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
                for y in range(0, len(result_arr[row])):
                    this_size = result_arr[row][y]
                    if this_size in corr_dict:
                        result_arr[row][y] = corr_dict[this_size]

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
    def scan(self, value: str):
        self._scan = value

    @property
    def micro_ids(self):
        return self._micro_ids

    @micro_ids.setter
    def micro_ids(self, value: Union[list, DF]):
        self._micro_ids = value

    @property
    def micro_x_size(self):
        return self._micro_x_size

    @micro_x_size.setter
    def micro_x_size(self, value: Union[list, DF]):
        self._micro_x_size = value

    @property
    def micro_y_size(self):
        return self._micro_y_size

    @micro_y_size.setter
    def micro_y_size(self, value: Union[list, DF]):
        self._micro_y_size = value

    @property
    def ids_in_clusters(self):
        return self._ids_in_clusters

    @ids_in_clusters.setter
    def ids_in_clusters(self, value: list):
        self._ids_in_clusters = value
