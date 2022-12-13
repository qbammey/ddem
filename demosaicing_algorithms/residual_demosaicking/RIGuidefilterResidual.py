import numpy as np
from RIguidedfilter3gf import guidedfilter3gf
from filtertools import filter2D





def GuidefilterResidual(rawq, mask, maskGr, maskGb, mosaic, Algorithm):
    """
    Guided filter processing used for the green channel interpolation by residual 
    interpolation algorithms ('GBTF', 'RI', 'MLRI', 'WMLRI')  
    (Algorithm 5)
    """

    maskR = mask[:, :, 0]
    maskB = mask[:, :, 2]

    # compute horizontal and vertical interpolations

    ## Due to the particular forms for the filters Kv and Kh 
    ## rawh and rawv always mix values from the same channel. 
    ## This implies that the same convolution can be used for 
    ## each channel by multiplying with the mask. The same 
    ## result could be obtained with the following lines but it's less efficient.
    #Guiderh = mosaic[:, :, 0] + filter2D(mosaic[:, :, 0], Kh)  
    #Guidegh = mosaic[:, :, 1] + filter2D(mosaic[:, :, 1], Kh) 
    #Guidebh = mosaic[:, :, 2] + filter2D(mosaic[:, :, 2], Kh)  
    #
    #Guiderv = mosaic[:, :, 0] + filter2D(mosaic[:, :, 0], Kv)  
    #Guidegv = mosaic[:, :, 1] + filter2D(mosaic[:, :, 1], Kv)  
    #Guidebv = mosaic[:, :, 2] + filter2D(mosaic[:, :, 2], Kv)  
    Kh = np.array([[1/2, 0, 1/2]])
    Kv = Kh.T
    rawh = filter2D(rawq, Kh)
    rawv = filter2D(rawq, Kv)

    Guidegh = mosaic[:, :, 1] + rawh * mask[:, :, 0] + rawh * mask[:, :, 2]
    Guiderh = mosaic[:, :, 0] + rawh * maskGr
    Guidebh = mosaic[:, :, 2] + rawh * maskGb

    Guidegv = mosaic[:, :, 1] + rawv * mask[:, :, 0] + rawv * mask[:, :, 2]
    Guiderv = mosaic[:, :, 0] + rawv * maskGb
    Guidebv = mosaic[:, :, 2] + rawv * maskGr




    # tentative image
    # Algorithm = 'WMLRI'
    if Algorithm == 'RI':
        h = 5
        v = 0
    elif Algorithm == 'MLRI':
        h = 3
        v = 3
    elif Algorithm == 'WMLRI':
        h = 3
        v = 3

    eps = 0
    F = np.array([[-1, 0, 2, 0, -1]])
    FT = F.T

    # apply the guided filtering algorithm to each directional inteprolation
    tentativeRh  = guidedfilter3gf(Guidegh, mosaic[:, :, 0]         , maskR , h, v, eps, Algorithm, F)
    tentativeGrh = guidedfilter3gf(Guiderh, mosaic[:, :, 1] * maskGr, maskGr, h, v, eps, Algorithm, F)
    tentativeGbh = guidedfilter3gf(Guidebh, mosaic[:, :, 1] * maskGb, maskGb, h, v, eps, Algorithm, F)
    tentativeBh  = guidedfilter3gf(Guidegh, mosaic[:, :, 2]         , maskB , h, v, eps, Algorithm, F)

    tentativeRv  = guidedfilter3gf(Guidegv, mosaic[:, :, 0]         , maskR , v, h, eps, Algorithm, FT)
    tentativeGrv = guidedfilter3gf(Guiderv, mosaic[:, :, 1] * maskGb, maskGb, v, h, eps, Algorithm, FT)
    tentativeGbv = guidedfilter3gf(Guidebv, mosaic[:, :, 1] * maskGr, maskGr, v, h, eps, Algorithm, FT)
    tentativeBv  = guidedfilter3gf(Guidegv, mosaic[:, :, 2]         , maskB , v, h, eps, Algorithm, FT)

    tentativeGrh = np.clip(tentativeGrh, 0, 255)
    tentativeGrv = np.clip(tentativeGrv, 0, 255)
    tentativeGbh = np.clip(tentativeGbh, 0, 255)
    tentativeGbv = np.clip(tentativeGbv, 0, 255)
    tentativeRh = np.clip(tentativeRh, 0, 255)
    tentativeRv = np.clip(tentativeRv, 0, 255)
    tentativeBh = np.clip(tentativeBh, 0, 255)
    tentativeBv = np.clip(tentativeBv, 0, 255)

    # residual
    residualGrh = (mosaic[:, :, 1] - tentativeGrh) * maskGr
    residualGbh = (mosaic[:, :, 1] - tentativeGbh) * maskGb
    residualRh = (mosaic[:, :, 0] - tentativeRh) * maskR
    residualBh = (mosaic[:, :, 2] - tentativeBh) * maskB
    residualGrv = (mosaic[:, :, 1] - tentativeGrv) * maskGb
    residualGbv = (mosaic[:, :, 1] - tentativeGbv) * maskGr
    residualRv = (mosaic[:, :, 0] - tentativeRv) * maskR
    residualBv = (mosaic[:, :, 2] - tentativeBv) * maskB

    # residual interpolation
    residualGrh = filter2D(residualGrh, Kh)
    residualGbh = filter2D(residualGbh, Kh)
    residualRh = filter2D(residualRh, Kh)
    residualBh = filter2D(residualBh, Kh)

    residualGrv = filter2D(residualGrv, Kv)
    residualGbv = filter2D(residualGbv, Kv)
    residualRv = filter2D(residualRv, Kv)
    residualBv = filter2D(residualBv, Kv)

    # add tentative image
    Grh = (tentativeGrh + residualGrh) * maskR
    Gbh = (tentativeGbh + residualGbh) * maskB
    Rh = (tentativeRh + residualRh) * maskGr
    Bh = (tentativeBh + residualBh) * maskGb
    Grv = (tentativeGrv + residualGrv) * maskR
    Gbv = (tentativeGbv + residualGbv) * maskB
    Rv = (tentativeRv + residualRv) * maskGb
    Bv = (tentativeBv + residualBv) * maskGr

    Grh = np.clip(Grh, 0, 255)
    Grv = np.clip(Grv, 0, 255)
    Gbh = np.clip(Gbh, 0, 255)
    Gbv = np.clip(Gbv, 0, 255)
    Rh = np.clip(Rh, 0, 255)
    Rv = np.clip(Rv, 0, 255)
    Bh = np.clip(Bh, 0, 255)
    Bv = np.clip(Bv, 0, 255)

    # vertical and horizontal color difference
    difh = mosaic[:, :, 1] + Grh + Gbh - mosaic[:, :, 0] - mosaic[:, :, 2] - Rh - Bh
    difv = mosaic[:, :, 1] + Grv + Gbv - mosaic[:, :, 0] - mosaic[:, :, 2] - Rv - Bv


    ###  Combine Vertical and Horizontal Color Differences ###
    # color difference gradient
    Kh = np.array([[1, 0, -1]])
    Kv = Kh.T
    difh2 = abs(filter2D(difh, Kh))
    difv2 = abs(filter2D(difv, Kv))

    return difh, difv, difh2, difv2
