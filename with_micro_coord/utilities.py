import numpy as np
import cv2 as cv
import re

def crop_image(img, tolerance=0):
    """remove black (zero-values) from image"""
    mask = img > tolerance
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def alphaNumOrder(string):
    """ Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return ''.join([format(int(x), '05d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])


def t_show(img):
    cv.imshow('test', img), cv.waitKey()


def z_project(img_stack):
    max_projection = np.max(img_stack, axis=0)
    return max_projection
