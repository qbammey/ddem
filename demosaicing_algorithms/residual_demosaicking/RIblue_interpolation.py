import numpy as np
from RIguidedfilter3gf import guidedfilter3gf
from filtertools import filter2D



def blue_interpolation(green, mosaic, mask, pattern, h, v, eps, dif, Algorithm):
    """ 
    blue interpolation implementing Residual Interpolation demosaicking
    algorithms ('GBTF', 'RI', 'MLRI', 'WMLRI')  
    Arguments: 
        green: image containing the interpolated green channel
        mosaic: 3 channel image containing the R G B mosaic
        mask: 3 channel image indicating where the mosaic is set
        pattern: Bayer pattern 'grbg', 'rggb', 'gbrg', 'bggr'
        h,v: support of the guided filter
        eps: guided filter regularization (use 0) 
        dif: green residual image (from RIXgreen_interpolation)
        Algorithm: one of 'GBTF', 'RI', 'MLRI', 'WMLRI'
    Returns: 
        blue: the interpolated blue channel 
    """
    if Algorithm == 'GBTF':
        # This functions implements Algorithm 4
        Prb = np.array([[0, 0, -1, 0, -1, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0], 
                        [-1, 0, 10, 0, 10, 0, -1],
                        [0, 0, 0, 0, 0, 0, 0], 
                        [-1, 0, 10, 0, 10, 0, -1], 
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, -1, 0, -1, 0, 0]]) / 32
        Aknl = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4

        blue = mosaic[:, :, 2] + mask[:, :, 0] * (green - filter2D(dif, Prb))
        tempimg = mosaic[:, :, 1] - mask[:, :, 1] * filter2D(green, Aknl)\
                  + mask[:, :, 1] * filter2D(blue, Aknl)
        blue = blue + tempimg

    else:
        # This functions implements Algorithm 6
        F = np.array([[0, 0, -1, 0, 0], 
                      [0, 0, 0, 0, 0], 
                      [-1, 0, 4, 0, -1], 
                      [0, 0, 0, 0, 0], 
                      [0, 0, -1, 0, 0]])
        H = np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]])

        tentativeB = guidedfilter3gf(green, mosaic[:, :, 2], mask[:, :, 2], h, v, eps, Algorithm, F)
        tentativeB = np.clip(tentativeB, 0, 255)
        residualB = mask[:, :, 2] * (mosaic[:, :, 2] - tentativeB)
        residualB = filter2D(residualB, H)
        blue = residualB + tentativeB

    # blue interpolation
    blue = np.clip(blue, 0, 255)

    return blue




