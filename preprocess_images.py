import os
import re
import tifffile as tif
import cv2 as cv


def alphaNumOrder(string: str):
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return ''.join([format(int(x), '05d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])


def equalize_histograms(img_list: list, contract_limit: int = 127, kernel_size: int = (41, 41)) -> list:
    """ function for adaptive normalization of image histogram CLAHE """

    print('equalizing histograms')
    clahe = cv.createCLAHE(contract_limit, kernel_size)
    for i in range(0, len(img_list)):
        img_list[i] = clahe.apply(img_list[i])
    return img_list


def read_images(img_dir: str) -> list:
    """read images in natural order (with respect to numbers)"""

    print('reading images')
    allowed_extensions = ('tif', 'tiff')
    file_list = [fn for fn in os.listdir(img_dir) if fn.endswith(allowed_extensions)]
    file_list.sort(key=alphaNumOrder)

    img_list = []
    for fn in file_list:
        img_list.append(tif.imread(img_dir + fn))
    return img_list


def preprocess_images(img_dir: str) -> list:
    """Function wrapper that imports images and does preprocessing(histogram equalization)"""

    img_list = read_images(img_dir)
    equalized_images = equalize_histograms(img_list)
    return equalized_images
