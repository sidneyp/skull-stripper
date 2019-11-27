import numpy as np
import cv2 as cv
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """ Function to perform elastic transformation for data augmentation
        Elastic deformation of images as described in [Simard2003]_ (with modifications).
         [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

         Original code taken from:
         https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Perform random affine transformations
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv.getAffineTransform(pts1, pts2)
    image = cv.warpAffine(image, M, shape_size[::-1], borderMode=cv.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def draw_grid(im, grid_size):
    """
    Function to draw a grid on the image.
    """
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv.line(im, (0, j), (im.shape[1], j), color=(255,))


def get_elastic_transforms(imag,
                           imag_mask,
                           alpha = 1,
                           sigma = 0.098,
                           alpha_affine = 0.098):
    """
    """
    # Merge images into separete channels (shape will be (cols, rols, 2))
    im_merge = np.concatenate((imag[...,None], imag_mask[...,None]), axis=2)

    alpha1 = im_merge.shape[1] *alpha
    sigma1 = im_merge.shape[1] * sigma
    alpha_affine1 = im_merge.shape[1] * alpha_affine

    # Apply transformation on image
    im_merge_t = elastic_transform(im_merge, alpha1, sigma1, alpha_affine1)
    # Split image and mask
    im_t = im_merge_t[...,0]
    im_mask_t = im_merge_t[...,1]
    return im_t, im_mask_t


def show_result(imag,imag_mask):
    im_t, im_mask_t = get_elastic_transforms(imag, imag_mask)
    plt.figure(figsize = (8,8))
    plt.imshow(np.c_[np.r_[imag, imag_mask], np.r_[im_t, im_mask_t]], cmap='gray')
