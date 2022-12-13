import numpy as np
from mosaic_bayer import mosaic_bayer
from RIgreen_interpolation import green_interpolation
from RIred_interpolation import red_interpolation
from RIblue_interpolation import blue_interpolation



def demosaic_RI(mosaic, pattern, sigma, Algorithm):
    """
    Main function for the Residual Interpolation demosaicking
    algorithms 'GBTF', 'RI', 'MLRI', 'WMLRI'
    sigma is ignored by GBTF
    """

    # mosaic and mask (just to generate the mask)
    mosaic, mask = mosaic_bayer(mosaic, pattern)
    
    # imask
    imask = (mask == 0)

    # green interpolation
    green, dif = green_interpolation(mosaic, mask, pattern, sigma, Algorithm)

    # parameters for guided upsampling
    h = 5
    v = 5
    eps = 0

    # Red and Blue demosaicking
    red = red_interpolation(green, mosaic, mask, pattern, h, v, eps, dif, Algorithm)
    blue = blue_interpolation(green, mosaic, mask, pattern, h, v, eps, dif, Algorithm)


    # result image
    rgb_size = mosaic.shape
    rgb_dem = np.zeros((rgb_size[0], rgb_size[1], 3))
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue

    return rgb_dem




if __name__ == "__main__":
    from skimage.io import imread, imsave
    rgb = imread('Sans_bruit_13.PNG')
    
    rgb = rgb.astype('float32')
    pattern = 'grbg'
    sigma = 1.0
    Algorithm = 'MLRI'

    # generate the mosaic
    mosaic, _ = mosaic_bayer(rgb, pattern)

    # call demosaicing
    rgb_dem = demosaic_RI(mosaic, pattern, sigma, Algorithm)
    imsave('test_%s2.tiff'%Algorithm, rgb_dem.astype('uint8'))
