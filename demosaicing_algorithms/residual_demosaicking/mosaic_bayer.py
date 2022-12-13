import numpy as np


def mosaic_bayer(rgb, pattern):
    """
    generate a mosaic from a rgb image
    pattern can be: 'grbg', 'rggb', 'gbrg', 'bggr'
    """
    num = np.zeros(len(pattern))
    pattern_list = list(pattern)
    p = pattern_list.index('r')
    num[p] = 0
    p = [idx for idx, i in enumerate(pattern_list) if i == 'g']
    num[p] = 1
    p = pattern_list.index('b')
    num[p] = 2

    size_rgb = rgb.shape
    mask = np.zeros((size_rgb[0], size_rgb[1], 3))

    # Generate mask
    mask[0::2, 0::2, int(num[0])] = 1
    mask[0::2, 1::2, int(num[1])] = 1
    mask[1::2, 0::2, int(num[2])] = 1
    mask[1::2, 1::2, int(num[3])] = 1

    # Generate mosaic
    mosaic = rgb * mask

    return mosaic, mask



def get_mosaic_masks(mosaic, pattern):
    """
    generate the mosaic masks assuming a given pattern
    returns:  maskGr, maskGb, maskR, maskB
    """
    size_rawq = mosaic.shape
    maskGr = np.zeros((size_rawq[0], size_rawq[1]))
    maskGb = np.zeros((size_rawq[0], size_rawq[1]))
    maskR  = np.zeros((size_rawq[0], size_rawq[1]))
    maskB  = np.zeros((size_rawq[0], size_rawq[1]))

    if pattern == 'grbg':
        maskGr[0::2, 0::2] = 1
        maskGb[1::2, 1::2] = 1
        maskR [0::2, 1::2] = 1
        maskB [1::2, 0::2] = 1
    elif pattern == 'rggb':
        maskGr[0::2, 1::2] = 1
        maskGb[1::2, 0::2] = 1
        maskB [1::2, 1::2] = 1
        maskR [0::2, 0::2] = 1
    elif pattern == 'gbrg':
        maskGb[0::2, 0::2] = 1
        maskGr[1::2, 1::2] = 1
        maskR [1::2, 0::2] = 1
        maskB [0::2, 1::2] = 1
    elif pattern == 'bggr':
        maskGb[0::2, 1::2] = 1
        maskGr[1::2, 0::2] = 1
        maskB [0::2, 0::2] = 1
        maskR [1::2, 1::2] = 1

    return maskGr, maskGb, maskR, maskB