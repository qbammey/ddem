import numpy as np
from mosaic_bayer import mosaic_bayer
from ARIgreen_interpolation import ARIgreen_interpolation
from ARIred_blue_interpolation_first import ARIred_blue_interpolation_first
from ARIred_blue_interpolation_second import ARIred_blue_interpolation_second


def demosaic_ARI(mosaic, pattern):
    """
    ARI (Adaptive Residual Interpolation) demosaicing main function
    """
    # guided filter epsilon
    eps = 1e-10

    # mosaic and mask (just to generate the mask)
    mosaic, mask = mosaic_bayer(mosaic, pattern)

    # green interpolation
    green = ARIgreen_interpolation(mosaic, mask, pattern, eps)

    # red and blue interpolation (first step: diagonal)
    red, blue = ARIred_blue_interpolation_first(green, mosaic, mask, eps)

    # red and blue interpolation (second step: horizontal/vertical)
    red, blue = ARIred_blue_interpolation_second(green, red, blue, mask, eps)

    rgb_dem = np.zeros(mosaic.shape)
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue

    return rgb_dem


if __name__ == "__main__":
    from skimage.io import imread, imsave
    rgb = imread('Sans_bruit_13.PNG')
    rgb = rgb.astype('float32')
    pattern = 'grbg'

    # generate the mosaic
    mosaic, _ = mosaic_bayer(rgb, pattern)

    rgb_dem = demosaic_ARI(mosaic, pattern)
    imsave('test_ARI.tiff', rgb_dem.astype('uint8'))
