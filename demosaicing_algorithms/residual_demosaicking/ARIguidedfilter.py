import numpy as np
from filtertools import filter2D, boxFilter 



def guidedfilter(I, p, M, h, v, eps, direction):
    """
    implements the Guided Filter (GF) used by the ARI demosaicing algorithm
    Arguments: 
        I: guidance image
        p: input image
        M: Mask image
        (h,v) size of the filter
        eps: regularization
        direction: HV (horizontal-vertical) or diag (diagonal, used for Red and Blue)
    Returns: 
        q: filtered version of p
    """
    M = M.astype('float32')

    # horizontal and vertical MLGF guided filtering 
    if direction == 'HV':
        # The number of the sammpled pixels in each local patch
        boxsz = (2*h+1, 2*v+1)   # in matlab boxfilter uses h,v as radius, opencv needs the diameter
        N = boxFilter(M, boxsz ) 
        N[N == 0] = 1

        mean_I = boxFilter(I * M, boxsz ) / N
        mean_p = boxFilter(p, boxsz ) / N
        mean_Ip = boxFilter(I * p, boxsz ) / N

        # The covariance of (I, p) in each local patch
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = boxFilter(I * I * M, boxsz ) / N
        var_I = mean_II - mean_I * mean_I

        # linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        # weighted average
        dif = boxFilter(I * I * M, boxsz ) * a * a \
              + b * b * N + boxFilter(p * p * M, boxsz ) \
              + 2 * a * b * boxFilter(I * M, boxsz ) \
              - 2 * b * boxFilter(p * M, boxsz ) \
              - 2 * a * boxFilter(p * I * M, boxsz )
        dif = dif / N
        dif[dif < 0] = 0
        dif = np.sqrt(dif)
        dif = np.nan_to_num(dif)
        dif[dif < 0.001] = 0.001
        dif = 1 / dif
        wdif = boxFilter(dif, boxsz )

        mean_a = boxFilter(a * dif, boxsz ) / (wdif + 1e-4)
        mean_b = boxFilter(b * dif, boxsz ) / (wdif + 1e-4)

    # diagonal and anti-diagonal MLGF guided filtering 
    else:
        # define the diagonal "boxFilter" window
        r = h + v
        diagBox = np.ones((2 * r + 1, 2 * r + 1))
        w = 2 * r + 1

        for i in range(1, v+1):
            for t in range(1, 2 * i):
                diagBox[t-1, 2*i-t-1] = 0
                diagBox[w+1-t-1, w+1-2*i+t-1] = 0

        for i in range(1, h+1):
            for t in range(1, 2*i):
                diagBox[t-1, w+1-2*i+t-1] = 0
                diagBox[w+1-t-1, 2*i-t-1] = 0 

        tmp = np.zeros((2*r+1, 2*r+1))
        tmp[0::2, 0::2] = 1
        tmp[1::2, 1::2] = 1
        diagBox = diagBox * tmp

        # number of sampled pixels in each local patch
        N = filter2D(M, diagBox)
        N[(-1e-8 < N) == (N < 1e-8)] = 0
        N[N == 0] = 1
    
        mean_I = filter2D(I * M, diagBox) / N
        mean_I[(-1e-8 < mean_I) == (mean_I < 1e-8)] = 0.0
        mean_p = filter2D(p * M, diagBox) / N
        mean_p[(-1e-8 < mean_p) == (mean_p < 1e-8)] = 0.0
        mean_Ip = filter2D(I * p * M, diagBox) / N
        mean_Ip[(-1e-8 < mean_Ip) == (mean_Ip < 1e-8)] = 0.0

        cov_Ip = mean_Ip - mean_I * mean_p
        cov_Ip[(-1e-8 < cov_Ip) == (cov_Ip < 1e-8)] = 0.0
        mean_II = filter2D(I * I * M, diagBox) / N
        mean_II[(-1e-8 < mean_II) == (mean_II < 1e-8)] = 0.0
        var_I = mean_II - mean_I * mean_I
        var_I[(-1e-8 < var_I) == (var_I < 1e-8)] = 0.0

        # linear coefficients
        a = cov_Ip / (var_I + eps)
        a[(-1e-8 < a) == (a < 1e-8)] = 0.0
        b = mean_p - a * mean_I
        b[(-1e-8 < b) == (b < 1e-8)] = 0.0

        # weighted average
        dif = filter2D(I * I * M,diagBox) * a * a \
              + b * b * N + filter2D(p * p * M, diagBox)\
              + 2 * a * b * filter2D(I * M, diagBox)\
              - 2 * b * filter2D(p * M, diagBox)\
              - 2 * a * filter2D(p * I * M, diagBox)

        dif = dif / N
        dif[dif < 1e-8] = 0.0
        dif = dif ** 0.5
        dif = np.nan_to_num(dif)
        dif[dif < 0.001] = 0.001
        dif = 1.0 / dif
        wdif = filter2D(dif, diagBox)

        adif = filter2D(a * dif, diagBox)
        adif[(-1e-8 < adif) == (adif < 1e-8)] = 0
        mean_a = adif / (wdif + 1e-4)

        bdif = filter2D(b * dif, diagBox)
        bdif[(-1e-8 < bdif) == (bdif < 1e-8)] = 0
        mean_b = bdif / (wdif + 1e-4)


    # output
    q = mean_a * I + mean_b

    return q
