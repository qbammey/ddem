import numpy as np
import math


def impsnr(x, y, peak=255, b=0):
    """
    computes the psnr between images x and y
    peak (default 255) indicates the maximum value of the image
    b (default 0) is used to remove additive bias from the signal
    """
    if b > 0:
        x = x[b:(x.shape[0] - b), b:(x.shape[1] - b), :]
        y = y[b:(y.shape[0] - b), b:(y.shape[1] - b), :]

    dif = x / 1.0 - y / 1.0
    dif = dif * dif

    psnr = []
    for i in range(dif.shape[2]):
        d = dif[:, :, i]
        mse = np.mean(d) + 1e-32
        psnr_val = 10 * math.log10(peak * peak / mse)
        psnr.append(psnr_val)

    return psnr



def imcpsnr(x, y, peak=255, b=0):
    """
    computes the color psnr between images x and y
    peak (default 255) indicates the maximum value of the image
    b (default 0) is used to remove additive bias from the signal
    """
    if b > 0:
        x = x[b:(x.shape[0] - b), b:(x.shape[1] - b), :]
        y = y[b:(y.shape[0] - b), b:(y.shape[1] - b), :]

    dif = x / 1.0 - y / 1.0
    dif = dif * dif
    mse = np.mean(dif) + 1e-32
    cpsnr = 10 * math.log10(peak * peak / mse)

    return cpsnr

