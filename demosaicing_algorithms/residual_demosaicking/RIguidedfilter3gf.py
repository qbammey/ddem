##### GuidedFilter ##################################################################
# This functions implements Algorithm 11
#
#    - guidance image: I
#    - filtering input image: p
#    - binary mask: M
#    - local directional window radius: h, v
#    - regularization parameter: eps
#
#####################################################################################

import numpy as np
from filtertools import filter2D, boxFilter




def guidedfilter3gf(I, p, M, h, v, eps, Algorithm, F):
    """
    implements 3 variants of the Guided Filter (GF) including Minimized-Laplacian Guided Filter (MLGF) 
    which are used by the RI, MLRI, and WMLRI demosaicing algorithms
    Arguments: 
        I: guidance image
        p: input image 
        M: Mask image 
        (h,v) size of the filter
        eps: regularization
        Algorithm: RI,MLRI,WMLRI
        F: laplacian kernel
    Returns: 
        q: filtered version of p 
    """   
    # threshold parameter
    th = 0.00001 * 255 * 255

    # Image size
    I_size = I.shape

    # The number of the sammpled pixels in each local patch
    # In MATLAB, h and v are radii, but in opencv, diameter is required
    boxsz = (2*h+1, 2*v+1)
    N = boxFilter(M, boxsz)
    # this avoids 0/0 in rectangles where the mask is null, and the result should be 0.
    N[N == 0] = 1

    # these are weighted box means because N is computed from M
    mean_I = boxFilter(I * M, boxsz) / N  
    mean_p = boxFilter(p * M, boxsz) / N

    # Algorithm='MLRI'
    if Algorithm == 'MLRI' or Algorithm =='WMLRI':
        difIF = filter2D(I*M, F) 
        difpF = filter2D(p, F)
        mean_Ip = boxFilter(difIF * difpF * M, boxsz) / N
        mean_II = boxFilter(difIF * difIF * M, boxsz) / N
        mean_II[mean_II < th] = th   
        a = mean_Ip / (mean_II + eps)
    else:  # Algorithm='RI'
        mean_Ip = boxFilter(I * p * M, boxsz) / N
        # The covariance of (I, p) in each local patch
        mean_II = boxFilter(I * I * M, boxsz) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I
        var_I[var_I < th] = th

        # linear coefficients
        a = cov_Ip / (var_I + eps)

    b = mean_p - a * mean_I

    if Algorithm =='WMLRI':
        # computes the denominator of line 16 in Algorithm 11
        dif = boxFilter(I * I * M, boxsz) * a * a \
              + b * b * N + boxFilter(p * p * M, boxsz) \
              + 2 * a * b * boxFilter(I * M, boxsz)\
              - 2 * b * boxFilter(p * M, boxsz) \
              - 2 * a * boxFilter(p * I * M, boxsz)
        dif = dif / N
        dif[dif < 0] = 0
        dif[dif < 0.001] = 0.001
        dif = 1 / dif
        wdif = boxFilter(dif, boxsz)
        wdif[wdif < 0.001] = 0.001
        mean_a = boxFilter(a * dif, boxsz) / wdif
        mean_b = boxFilter(b * dif, boxsz) / wdif

    else:
        # The size of each local patch; N=(2h+1)*(2v+1) except for boundary pixels.
        N2 = boxFilter(np.ones((I_size[0], I_size[1])), boxsz)

        mean_a = boxFilter(a, boxsz) / N2
        mean_b = boxFilter(b, boxsz) / N2

    # output
    q = mean_a * I + mean_b

    return q
