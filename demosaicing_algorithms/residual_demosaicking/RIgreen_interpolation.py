import numpy as np
from RIHaResidual import haresidual  # used by GBTF
from RIGuidefilterResidual import GuidefilterResidual  # used by RI, MLRI, and WMLRI
from filtertools import filter2D, getGaussianKernel
from mosaic_bayer import get_mosaic_masks


#  Directional weights
def DirectsSmooth4Kernel(Algorithm, sigma):
    """
    outputs the directional smoothing kernels for different 
    demosaicing Algorithms (GBTF, RI, MLRI, WMLRI)
    sigma is ignored by GBTF 
    """
    if Algorithm == 'GBTF':
        Ke = np.array([[0, 0, 0, 0, 26, 24, 21, 17, 12]]) / 100
        Kw = np.array([[12, 17, 21, 24, 26, 0, 0, 0, 0]]) / 100
    elif Algorithm == 'RI':
        h = getGaussianKernel(9, sigma).T
        Ke = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]]) * h
        Kw = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]]) * h
        Ke = Ke / np.sum(Ke, 1)
        Kw = Kw / np.sum(Kw, 1)
    elif Algorithm == 'MLRI':
        h = getGaussianKernel(9, sigma).T
        Ke = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]]) * h
        Kw = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]]) * h
        Ke = Ke / np.sum(Ke, 1)
        Kw = Kw / np.sum(Kw, 1)
    elif Algorithm == 'WMLRI':
        h = getGaussianKernel(9, sigma).T
        Ke = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]]) * h
        Kw = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]]) * h
        Ke = Ke / np.sum(Ke, 1)
        Kw = Kw / np.sum(Kw, 1)

    Ks = Ke.T
    Kn = Kw.T
    return Kn, Ks, Ke, Kw





#  Directional weights
def Means4Weights(Algorithm, difh2, difv2):
    """
    computes the weights used for the directional propagation (S,N,W,E) 
    for different demosaicing Algorithms (GBTF, RI, MLRI, WMLRI) 
    """    
    if Algorithm == 'GBTF':
        K = np.multiply(getGaussianKernel(5, 2), (getGaussianKernel(5, 2)).T)
        Kw = np.array([[1, 0, 0]])
        Ke = np.array([[0, 0, 1]])
    elif Algorithm == "RI":
        K = np.ones((5, 5))
        Kw = np.array([[1, 0, 0, 0, 0]])
        Ke = np.array([[0, 0, 0, 0, 1]])
        # 3-tap filters also work quite well
        #Kw = np.array([[1, 0, 0, 0, 0]])
        #Ke = np.array([[0, 0, 0, 0, 1]])
    elif Algorithm == "MLRI":
        K = np.ones((3, 3))
        Kw = np.array([[1, 0, 0]])
        Ke = np.array([[0, 0, 1]])
    elif Algorithm == "WMLRI":
        K = np.multiply(getGaussianKernel(5, 2), (getGaussianKernel(5, 2)).T)
        Kw = np.array([[1, 0, 0]])
        Ke = np.array([[0, 0, 1]])

    wh = filter2D(difh2, K)
    wv = filter2D(difv2, K)

    Ks = Ke.T
    Kn = Kw.T

    Ww = filter2D(wh, Kw)
    We = filter2D(wh, Ke)
    Wn = filter2D(wv, Kn)
    Ws = filter2D(wv, Ks)

    Ww = 1 / (Ww * Ww + 1e-32)
    We = 1 / (We * We + 1e-32)
    Ws = 1 / (Ws * Ws + 1e-32)
    Wn = 1 / (Wn * Wn + 1e-32)
 
    return Wn, Ws, We, Ww




def green_interpolation(mosaic, mask, pattern, sigma, Algorithm):
    """ 
    green interpolation implementing Residual Interpolation demosaicking 
    algorithms ('GBTF', 'RI', 'MLRI', 'WMLRI')  
    Arguments: 
        mosaic: 3 channel image containing the R G B mosaic
        mask: 3 channel image indicating where the mosaic is set
        pattern: Bayer pattern 'grbg', 'rggb', 'gbrg', 'bggr'
        sigma: directional weight smoothing (ignored by GBTF)
        Algorithm: one of 'GBTF', 'RI', 'MLRI', 'WMLRI'
    Returns: 
        green: the interpolated green channel 
        dif: green residual image
    """

    # raw CFA data
    rawq = np.sum(mosaic, axis=2)

    ### Calculate Horizontal and Vertical Color Differences ###
    # mask
    maskGr, maskGb, _, _ = get_mosaic_masks(rawq,pattern)

    # Algorithm = 'GBTF'
    # Algorithm = 'RI'
    if Algorithm == 'GBTF':
        # This functions implements Algorithm 3
        difh, difv, difh2, difv2 = haresidual(rawq, mask, maskGr, maskGb, mosaic)
    else:
        # This functions implements Algorithm 5
        difh, difv, difh2, difv2 = GuidefilterResidual(rawq, mask, maskGr, maskGb, mosaic, Algorithm)

    ## final color differece estimate (last part of the 3rd step)
    # directional weight. These lines implement line 19 of Algorithm 5
    Kn, Ks, Ke, Kw = DirectsSmooth4Kernel(Algorithm, sigma)
    Wn, Ws, We, Ww = Means4Weights(Algorithm, difh2, difv2)

    # combine directional color differences
    difn = filter2D(difv, Kn)
    difs = filter2D(difv, Ks)
    dife = filter2D(difh, Ke)
    difw = filter2D(difh, Kw)

    Wt = Ww + We + Wn + Ws
    dif = (Wn * difn + Ws * difs + Ww * difw + We * dife) / Wt

    # Calculate Green by adding bayer raw data (4th step)
    green = dif + rawq

    green = green * (1-mask[:, :, 1]) + rawq * mask[:, :, 1]

    # clip to 0-255
    green = np.clip(green, 0, 255)

    return green, dif
