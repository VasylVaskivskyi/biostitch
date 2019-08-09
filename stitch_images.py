import numpy as np
import cv2 as cv
from cv2 import error as cv_errors

np.set_printoptions(suppress=True)  # use normal numeric notation instead of exponential


with open('homography_horizontal.tsv', 'w', encoding='utf-8') as f:    # create file or empty existing
    f.close()

with open('homography_vertical.tsv', 'w', encoding='utf-8') as f:    # create file or empty existing
    f.close()

'''
quick preview tool
def t_show(img):
    cv.imshow('test', img), cv.waitKey()
'''


def save_homography_and_size(homography, image_size, mode):
    """@mode: string "horizontal" or "vertical" """
    row1, row2 = homography.tolist()
    as_string = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(*row1, *row2, *image_size)
    with open('homography_' + mode + '.tsv', 'a', encoding='utf-8') as f:
        f.write(as_string)
        f.close()


def save_concat(left_img_width, top_image_height, image_size, mode):
    """@mode: string "horizontal" or "vertical"
        if mode "horizontal" use left_img_height
        if mode "vertical" use top_img_height"""
    if mode == 'horizontal':
        as_string = '1.0\t0.0\t{0}\t0.0\t1.0\t0.0\t{1}\t{2}\n'.format(left_img_width,*image_size)
    elif mode == 'vertical':
        as_string = '1.0\t0.0\t0.0\t0.0\t1.0\t{0}\t{1}\t{2}\n'.format(top_image_height, *image_size)
    with open('homography_' + mode + '.tsv', 'a', encoding='utf-8') as f:
        f.write(as_string)
        f.close()



def crop_image(img, tolerance=0):
    mask = img > tolerance
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def modify_affine_homography_x(homography):
    """ prevents image rotation, scaling and y-shift """
    x_coord = homography[0][2]
    new_homography = np.array([[1.0, 0.0, x_coord], [0.0, 1.0, 0.0]], dtype=np.float32)
    return new_homography


def modify_affine_homography_y(homography):
    """ prevents image rotation and scaling, but not shift """
    x_coord = homography[0][2]
    y_coord = homography[1][2]
    new_homography = np.array([[1.0, 0.0, x_coord], [0.0, 1.0, y_coord]], dtype=np.float32)
    return new_homography


def stitch_pair_horizontal(left, right):
    """ takes as input numpy arrays and stitches them horizontally"""

    # convert images to uint8, so detector can use them
    img1 = cv.convertScaleAbs(left, None, alpha=(255.0 / 65535.0))
    img2 = cv.convertScaleAbs(right, None, alpha=(255.0 / 65535.0))

    # create feature detector and keypoint descriptors
    detector = cv.AgastFeatureDetector_create(threshold=8)
    descriptor = cv.xfeatures2d.DAISY_create(radius=7, q_radius=3, q_theta=10, q_hist=10, norm=cv.xfeatures2d.DAISY_NRM_FULL)

    # detect keypoints and compute descriptors from them
    kp1 = detector.detect(img1)
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2 = detector.detect(img2)
    kp2, des2 = descriptor.compute(img2, kp2)

    # find similar descriptors in two images
    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.knnMatch(des2, des1, k = 2)

    # Filter out unreliable points
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # convert keypoints to format acceptable for estimator
    src_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        # find out how images shifted (compute affine transformation)
        A, mask = cv.estimateAffinePartial2D(dst_pts, src_pts)
    except cv_errors:
        # handle blank images
        dst = np.concatenate((left, right), axis=1)
        save_concat(left_img_width=left.shape[1], top_image_height=None, image_size=dst.shape, mode='horizontal')
        return dst

    # if images should be stitched, but don't have have enough features to do so, then just concatenate them
    if A[1][2] > 10:
        dst = np.concatenate((left, right), axis=1)
        save_concat(left_img_width=left.shape[1], top_image_height=None, image_size=dst.shape, mode='horizontal')
        return dst

    # modify affine transformation to keep only shift in x-axis
    H = modify_affine_homography_x(A)

    # shift image using parameters from homology matrix
    highest = max(right.shape[0], left.shape[0])    # some images may differ in few pixels in height
    dst = cv.warpAffine(right, H, (right.shape[1] + left.shape[1], highest), None)

    if dst.max() == 0:
        # if shift is to big and dst is empty than jus concatenate images
        dst = np.concatenate((left, right), axis=1)
        save_concat(left_img_width=left.shape[1], top_image_height=None, image_size=dst.shape, mode='horizontal')
    else:
        # else fill the left side of dst with left image
        dst[:left.shape[0], :left.shape[1]] = left
    #t_show(dst)
    #tif.imsave('C:/Users/vv3/Desktop/image/stitched/img1_t.tif', dst)
    dst = crop_image(dst)
    save_homography_and_size(H, dst.shape, mode='horizontal')
    return dst


def stitch_pair_vertical(top, bottom):
    img1 = cv.convertScaleAbs(top, None, alpha=(255.0 / 65535.0))
    img2 = cv.convertScaleAbs(bottom, None, alpha=(255.0 / 65535.0))

    detector = cv.AgastFeatureDetector_create(threshold=8)
    descriptor = cv.xfeatures2d.DAISY_create(radius=7, q_radius=3, q_theta=10, q_hist=10, norm=cv.xfeatures2d.DAISY_NRM_FULL)

    kp1 = detector.detect(img1)
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2 = detector.detect(img2)
    kp2, des2 = descriptor.compute(img2, kp2)

    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.knnMatch(des2, des1, k = 2)
    # Fiter out unreliable points
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    # handle blank images
    try:
        A, mask = cv.estimateAffinePartial2D(dst_pts, src_pts)
    except cv_errors:
        dst = np.concatenate((top, bottom), axis=0)
        save_concat(left_img_width=None, top_image_height=top.shape[0], image_size=dst.shape, mode='vertical')
        return dst

    H_save = modify_affine_homography_y(A)

    shift_x = 0  # used to enter images
    shift_y = 0  # used to trim empty space on bottom
    if A[0][2] < 0:
        shift_x = int(round(A[0][2] * -1))
        A[0][2] = 0.0


    if A[1][2] < 0 or A[1][2] > 0:
        shift_y = int(round(bottom.shape[0] - A[1][2]))

    H = modify_affine_homography_y(A)

    longest = max(bottom.shape[1], top.shape[1])
    dst = cv.warpAffine(bottom, H, (longest + shift_x, bottom.shape[0] - shift_y + top.shape[0]), None)  # cartesian coordinate system

    if dst.max() == 0:
        dst = np.concatenate((top, bottom), axis=0)
        save_concat(left_img_width=None, top_image_height=top.shape[0], image_size=dst.shape, mode='vertical')
        return dst
    else:
        dst[:top.shape[0], shift_x:top.shape[1]+shift_x] = top  # numpy coordinate system
    #t_show(dst)
    #tif.imsave('C:/Users/vv3/Desktop/image/stitched/img1_t.tif', dst)
    dst = crop_image(dst)
    save_homography_and_size(H_save, dst.shape, mode='vertical')
    return dst


def process_images_pairwise_horizontal(arr):
    #arr = array.copy()  # copy to prevent .pop from modifying original list
    last = None

    # shortcuts
    if len(arr) == 1:
        print(-1)
        return arr
    elif len(arr) == 2:
        print(0)
        res = stitch_pair_horizontal(arr[0], arr[1])
        return res

    if len(arr) % 2 != 0:
        last = arr.pop()

    # create pairs of images
    pairs = []
    for i in range(0, len(arr), 2):
        pairs.append([arr[i], arr[i + 1]])

    # stitch each pair
    res = []
    for p in pairs:
        res.append(stitch_pair_horizontal(p[0], p[1]))

    # if there are any elements left unstitched repeat
    if last is None and len(res) < 2:
        print(1)
        return res
    elif last is None and len(res) >= 2:
        print(2)
        return process_images_pairwise_horizontal(res)
    elif last is not None:
        res.append(last)
        print(3)
        return process_images_pairwise_horizontal(res)


def process_images_pairwise_vertical(arr):
    #arr = array.copy()  # copy to prevent .pop from modifying original list
    last = None

    # shortcuts
    if len(arr) == 1:
        print(-1)
        return arr
    elif len(arr) == 2:
        print(0)
        res = stitch_pair_vertical(arr[0], arr[1])
        return res

    if len(arr) % 2 != 0:
        last = arr.pop()

    # create pairs of images
    pairs = []
    for i in range(0, len(arr), 2):
        pairs.append([arr[i], arr[i + 1]])

    # stitch each pair
    res = []
    for p in pairs:
        res.append(stitch_pair_vertical(p[0], p[1]))

    # if there are any elements left unstitched repeat
    if last is None and len(res) < 2:
        print(1)
        return res
    elif last is None and len(res) >= 2:
        print(2)
        return process_images_pairwise_vertical(res)
    elif last is not None:
        res.append(last)
        print(3)
        return process_images_pairwise_vertical(res)


def stitch_images(images, image_positions):

    print('stitching horizontal')
    res_h = []
    j = 0
    for row in image_positions:
        img_ids = [i[2] for i in row]
        img_ids.reverse()
        img_row = [images[i] for i in img_ids]
        print('row ', j, 'images', [i + 1 for i in img_ids])  # print image numbers from 1
        res_h.append(process_images_pairwise_horizontal(img_row))
        j += 1

    print('stitching vertical')
    res_v = process_images_pairwise_vertical(res_h)
    return res_v

