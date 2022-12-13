import numpy as np
from ARIguidedfilter import guidedfilter
from ARIguidedfilter_MLRI import guidedfilter_MLRI
from filtertools import filter2D, getGaussianKernel
from mosaic_bayer import get_mosaic_masks



# This functions implements Algorithm 7 and 8
def ARIgreen_interpolation(mosaic, mask, pattern, eps):
    """
    green interpolation for the ARI (Adaptive Residual Interpolation) demosaicking algorithm
    Arguments: 
        mosaic: 3 channel image containing the R G B mosaic
        mask: 3 channel image indicating where the mosaic is set
        pattern: Bayer pattern 'grbg', 'rggb', 'gbrg', 'bggr'
        eps: regularization parameter (recommended: 1e-10)
    Returns: 
        green: the interpolated green channel 
    """

    # raw CFA data
    rawq = np.sum(mosaic, axis=2)

    # mask 
    maskGr, maskGb, maskR, maskB = get_mosaic_masks(rawq,pattern)

    Mrh = maskR  + maskGr
    Mbh = maskB  + maskGb
    Mrv = maskR  + maskGb
    Mbv = maskB  + maskGr

    # Step (i): iterative directional interpolation (Algo 7 line 8-10)
    ## Due to the particular forms for the filters Kv and Kh 
    ## rawh and rawv always mix values from the same channel. 
    ## This implies that the same filtering can be used for 
    ## each channel by multiplying with the mask.
    Kh = np.array([[1/2, 0, 1/2]])
    Kv = Kh.T

    rawh = filter2D(rawq, Kh)
    rawv = filter2D(rawq, Kv)

    # horizontal direction  (Algo 7 line 8)
    Guidegrh = mosaic[:, :, 1] * maskGr + rawh * maskR 
    Guidegbh = mosaic[:, :, 1] * maskGb + rawh * maskB 
    Guiderh = mosaic[:, :, 0] + rawh * maskGr
    Guidebh = mosaic[:, :, 2] + rawh * maskGb

    # vertical direction
    Guidegrv = mosaic[:, :, 1] * maskGb + rawv * maskR 
    Guidegbv = mosaic[:, :, 1] * maskGr + rawv * maskB 
    Guiderv = mosaic[:, :, 0] + rawv * maskGb
    Guidebv = mosaic[:, :, 2] + rawv * maskGr

    # initial guided filter window size for RI
    h = 2
    v = 1

    # initial guided filter window size for MLRI
    h2 = 4
    v2 = 0

    # maximum iteration number
    itnum = 11

    # initialization of horizontal and vertical iteration criteria (Algo 7 line 11)
    RI_w2h = np.ones(maskGr.shape) * 1e32
    RI_w2v = np.ones(maskGr.shape) * 1e32

    MLRI_w2h = np.ones(maskGr.shape) * 1e32
    MLRI_w2v = np.ones(maskGr.shape) * 1e32

    # initial guide image for RI  (Algo 7 line 8)
    RI_Guidegrh = Guidegrh
    RI_Guidegbh = Guidegbh
    RI_Guiderh = Guiderh
    RI_Guidebh = Guidebh
    RI_Guidegrv = Guidegrv
    RI_Guidegbv = Guidegbv
    RI_Guiderv = Guiderv
    RI_Guidebv = Guidebv

    # initial guide image for MLRI
    MLRI_Guidegrh = Guidegrh
    MLRI_Guidegbh = Guidegbh
    MLRI_Guiderh = Guiderh
    MLRI_Guidebh = Guidebh
    MLRI_Guidegrv = Guidegrv
    MLRI_Guidegbv = Guidegbv
    MLRI_Guiderv = Guiderv
    MLRI_Guidebv = Guidebv

    # initialization of interpolated G values  (Algo 7 line 10)
    RI_Gh = Guidegrh + Guidegbh
    RI_Gv = Guidegrv + Guidegbv
    MLRI_Gh = Guidegrh + Guidegbh
    MLRI_Gv = Guidegrv + Guidegbv

    # Iterative horizontal and vertical interpolation
    for ittime in range(itnum):
        # generate horizontal and vertical tentative estimate (Algo 7 line 17)
        RI_tentativeGrh = guidedfilter(RI_Guiderh, RI_Guidegrh, Mrh, h, v, eps, direction='HV')
        RI_tentativeGbh = guidedfilter(RI_Guidebh, RI_Guidegbh, Mbh, h, v, eps, direction='HV')
        RI_tentativeRh = guidedfilter(RI_Guidegrh, RI_Guiderh, Mrh, h, v, eps, direction='HV')
        RI_tentativeBh = guidedfilter(RI_Guidegbh, RI_Guidebh, Mbh, h, v, eps, direction='HV')
        RI_tentativeGrv = guidedfilter(RI_Guiderv, RI_Guidegrv, Mrv, v, h, eps, direction='HV')
        RI_tentativeGbv = guidedfilter(RI_Guidebv, RI_Guidegbv, Mbv, v, h, eps, direction='HV')
        RI_tentativeRv = guidedfilter(RI_Guidegrv, RI_Guiderv, Mrv, v, h, eps, direction='HV')
        RI_tentativeBv = guidedfilter(RI_Guidegbv, RI_Guidebv, Mbv, v, h, eps, direction='HV')

        # generate horizontal tentative estimate by MLRI (Algo 7 line 17)
        Fh = -np.array([[-1, 0, 2, 0, -1]])
        MLRI_tentativeRh = guidedfilter_MLRI(MLRI_Guidegrh, MLRI_Guiderh, Mrh, maskR , h2, v2, eps, direction='HV', F=Fh)
        MLRI_tentativeBh = guidedfilter_MLRI(MLRI_Guidegbh, MLRI_Guidebh, Mbh, maskB , h2, v2, eps, direction='HV', F=Fh)
        MLRI_tentativeGrh = guidedfilter_MLRI(MLRI_Guiderh, MLRI_Guidegrh, Mrh, maskGr, h2, v2, eps, direction='HV', F=Fh)
        MLRI_tentativeGbh = guidedfilter_MLRI(MLRI_Guidebh, MLRI_Guidegbh, Mbh, maskGb, h2, v2, eps, direction='HV', F=Fh)

        # generate vertical tentative estimate by MLRI (Algo 7 line 17)
        Fv = Fh.T
        MLRI_tentativeRv = guidedfilter_MLRI(MLRI_Guidegrv, MLRI_Guiderv, Mrv, maskR , v2, h2, eps, direction='HV', F=Fv)
        MLRI_tentativeBv = guidedfilter_MLRI(MLRI_Guidegbv, MLRI_Guidebv, Mbv, maskB , v2, h2, eps, direction='HV', F=Fv)
        MLRI_tentativeGrv = guidedfilter_MLRI(MLRI_Guiderv, MLRI_Guidegrv, Mrv, maskGb, v2, h2, eps, direction='HV', F=Fv)
        MLRI_tentativeGbv = guidedfilter_MLRI(MLRI_Guidebv, MLRI_Guidegbv, Mbv, maskGr, v2, h2, eps, direction='HV', F=Fv)

        # calculate residuals of RI and MLRI (Algo 7 line 18)
        RI_residualGrh = (mosaic[:, :, 1] - RI_tentativeGrh) * maskGr
        RI_residualGbh = (mosaic[:, :, 1] - RI_tentativeGbh) * maskGb
        RI_residualRh = (mosaic[:, :, 0] - RI_tentativeRh) * maskR 
        RI_residualBh = (mosaic[:, :, 2] - RI_tentativeBh) * maskB 
        RI_residualGrv = (mosaic[:, :, 1] - RI_tentativeGrv) * maskGb
        RI_residualGbv = (mosaic[:, :, 1] - RI_tentativeGbv) * maskGr
        RI_residualRv = (mosaic[:, :, 0] - RI_tentativeRv) * maskR 
        RI_residualBv = (mosaic[:, :, 2] - RI_tentativeBv) * maskB 
        MLRI_residualGrh = (mosaic[:, :, 1] - MLRI_tentativeGrh) * maskGr
        MLRI_residualGbh = (mosaic[:, :, 1] - MLRI_tentativeGbh) * maskGb
        MLRI_residualRh = (mosaic[:, :, 0] - MLRI_tentativeRh) * maskR 
        MLRI_residualBh = (mosaic[:, :, 2] - MLRI_tentativeBh) * maskB 
        MLRI_residualGrv = (mosaic[:, :, 1] - MLRI_tentativeGrv) * maskGb
        MLRI_residualGbv = (mosaic[:, :, 1] - MLRI_tentativeGbv) * maskGr
        MLRI_residualRv = (mosaic[:, :, 0] - MLRI_tentativeRv) * maskR 
        MLRI_residualBv = (mosaic[:, :, 2] - MLRI_tentativeBv) * maskB 

        # horizontal and vertical linear interpolation of residuals (Algo 7 line 19)
        Kh = np.array([[1 / 2, 1, 1 / 2]])
        RI_residualGrh = filter2D(RI_residualGrh, Kh )
        RI_residualGbh = filter2D(RI_residualGbh, Kh )
        RI_residualRh = filter2D(RI_residualRh, Kh )
        RI_residualBh = filter2D(RI_residualBh, Kh )
        MLRI_residualGrh = filter2D(MLRI_residualGrh, Kh )
        MLRI_residualGbh = filter2D(MLRI_residualGbh, Kh )
        MLRI_residualRh = filter2D(MLRI_residualRh, Kh )
        MLRI_residualBh = filter2D(MLRI_residualBh, Kh )

        Kv = Kh.T
        RI_residualGrv = filter2D(RI_residualGrv, Kv )
        RI_residualGbv = filter2D(RI_residualGbv, Kv )
        RI_residualRv = filter2D(RI_residualRv, Kv )
        RI_residualBv = filter2D(RI_residualBv, Kv )
        MLRI_residualGrv = filter2D(MLRI_residualGrv, Kv )
        MLRI_residualGbv = filter2D(MLRI_residualGbv, Kv )
        MLRI_residualRv = filter2D(MLRI_residualRv, Kv )
        MLRI_residualBv = filter2D(MLRI_residualBv, Kv )

        # add tentative estimate (Algo 7 line 20)
        RI_Grh = (RI_tentativeGrh + RI_residualGrh) * maskR 
        RI_Gbh = (RI_tentativeGbh + RI_residualGbh) * maskB 
        RI_Rh = (RI_tentativeRh + RI_residualRh) * maskGr
        RI_Bh = (RI_tentativeBh + RI_residualBh) * maskGb
        RI_Grv = (RI_tentativeGrv + RI_residualGrv) * maskR 
        RI_Gbv = (RI_tentativeGbv + RI_residualGbv) * maskB 
        RI_Rv = (RI_tentativeRv + RI_residualRv) * maskGb
        RI_Bv = (RI_tentativeBv + RI_residualBv) * maskGr
        MLRI_Grh = (MLRI_tentativeGrh + MLRI_residualGrh) * maskR 
        MLRI_Gbh = (MLRI_tentativeGbh + MLRI_residualGbh) * maskB 
        MLRI_Rh = (MLRI_tentativeRh + MLRI_residualRh) * maskGr
        MLRI_Bh = (MLRI_tentativeBh + MLRI_residualBh) * maskGb
        MLRI_Grv = (MLRI_tentativeGrv + MLRI_residualGrv) * maskR 
        MLRI_Gbv = (MLRI_tentativeGbv + MLRI_residualGbv) * maskB 
        MLRI_Rv = (MLRI_tentativeRv + MLRI_residualRv) * maskGr
        MLRI_Bv = (MLRI_tentativeBv + MLRI_residualBv) * maskGr

        # Step(ii): adaptive selection of iteration at each pixel
        # calculate iteration criteria  (Algo 7 line 4)
        RI_criGrh = (RI_Guidegrh - RI_tentativeGrh) * Mrh
        RI_criGbh = (RI_Guidegbh - RI_tentativeGbh) * Mbh
        RI_criRh = (RI_Guiderh - RI_tentativeRh) * Mrh
        RI_criBh = (RI_Guidebh - RI_tentativeBh) * Mbh
        RI_criGrv = (RI_Guidegrv - RI_tentativeGrv) * Mrv
        RI_criGbv = (RI_Guidegbv - RI_tentativeGbv) * Mbv
        RI_criRv = (RI_Guiderv - RI_tentativeRv) * Mrv
        RI_criBv = (RI_Guidebv - RI_tentativeBv) * Mbv
        MLRI_criGrh = (MLRI_Guidegrh - MLRI_tentativeGrh) * Mrh
        MLRI_criGbh = (MLRI_Guidegbh - MLRI_tentativeGbh) * Mbh
        MLRI_criRh = (MLRI_Guiderh - MLRI_tentativeRh) * Mrh
        MLRI_criBh = (MLRI_Guidebh - MLRI_tentativeBh) * Mbh
        MLRI_criGrv = (MLRI_Guidegrv - MLRI_tentativeGrv) * Mrv
        MLRI_criGbv = (MLRI_Guidegbv - MLRI_tentativeGbv) * Mbv
        MLRI_criRv = (MLRI_Guiderv - MLRI_tentativeRv) * Mrv
        MLRI_criBv = (MLRI_Guidebv - MLRI_tentativeBv) * Mbv

        # calculate gradient of iteration criteria (Algo 8 line 5)
        Fh = np.array([[-1, 0, 1]])
        RI_difcriGrh = abs(filter2D(RI_criGrh, Fh ))
        RI_difcriGbh = abs(filter2D(RI_criGbh, Fh ))
        RI_difcriRh = abs(filter2D(RI_criRh, Fh ))
        RI_difcriBh = abs(filter2D(RI_criBh, Fh ))
        MLRI_difcriGrh = abs(filter2D(MLRI_criGrh, Fh ))
        MLRI_difcriGbh = abs(filter2D(MLRI_criGbh, Fh ))
        MLRI_difcriRh = abs(filter2D(MLRI_criRh, Fh ))
        MLRI_difcriBh = abs(filter2D(MLRI_criBh, Fh ))

        Fv = Fh.T
        RI_difcriGrv = abs(filter2D(RI_criGrv, Fv ))
        RI_difcriGbv = abs(filter2D(RI_criGbv, Fv ))
        RI_difcriRv = abs(filter2D(RI_criRv, Fv ))
        RI_difcriBv = abs(filter2D(RI_criBv, Fv ))
        MLRI_difcriGrv = abs(filter2D(MLRI_criGrv, Fv ))
        MLRI_difcriGbv = abs(filter2D(MLRI_criGbv, Fv ))
        MLRI_difcriRv = abs(filter2D(MLRI_criRv, Fv ))
        MLRI_difcriBv = abs(filter2D(MLRI_criBv, Fv ))

        # absolute value of iteration criteria
        RI_criGrh = abs(RI_criGrh)
        RI_criGbh = abs(RI_criGbh)
        RI_criRh = abs(RI_criRh)
        RI_criBh = abs(RI_criBh)
        RI_criGrv = abs(RI_criGrv)
        RI_criGbv = abs(RI_criGbv)
        RI_criRv = abs(RI_criRv)
        RI_criBv = abs(RI_criBv)
        MLRI_criGrh = abs(MLRI_criGrh)
        MLRI_criGbh = abs(MLRI_criGbh)
        MLRI_criRh = abs(MLRI_criRh)
        MLRI_criBh = abs(MLRI_criBh)
        MLRI_criGrv = abs(MLRI_criGrv)
        MLRI_criGbv = abs(MLRI_criGbv)
        MLRI_criRv = abs(MLRI_criRv)
        MLRI_criBv = abs(MLRI_criBv)

        # add Gr and R (Gb and B) criteria residuals (Algo 8 line 6)
        RI_criGRh = (RI_criGrh + RI_criRh) * Mrh
        RI_criGBh = (RI_criGbh + RI_criBh) * Mbh
        RI_criGRv = (RI_criGrv + RI_criRv) * Mrv
        RI_criGBv = (RI_criGbv + RI_criBv) * Mbv
        MLRI_criGRh = (MLRI_criGrh + MLRI_criRh) * Mrh
        MLRI_criGBh = (MLRI_criGbh + MLRI_criBh) * Mbh
        MLRI_criGRv = (MLRI_criGrv + MLRI_criRv) * Mrv
        MLRI_criGBv = (MLRI_criGbv + MLRI_criBv) * Mbv

        # add Gr and R (Gb and B) gradient of criteria residuals
        RI_difcriGRh = (RI_difcriGrh + RI_difcriRh) * Mrh
        RI_difcriGBh = (RI_difcriGbh + RI_difcriBh) * Mbh
        RI_difcriGRv = (RI_difcriGrv + RI_difcriRv) * Mrv
        RI_difcriGBv = (RI_difcriGbv + RI_difcriBv) * Mbv
        MLRI_difcriGRh = (MLRI_difcriGrh + MLRI_difcriRh) * Mrh
        MLRI_difcriGBh = (MLRI_difcriGbh + MLRI_difcriBh) * Mbh
        MLRI_difcriGRv = (MLRI_difcriGrv + MLRI_difcriRv) * Mrv
        MLRI_difcriGBv = (MLRI_difcriGbv + MLRI_difcriBv) * Mbv

        # directional map of iteration criteria (Algo 8 line 7)
        RI_crih = RI_criGRh + RI_criGBh
        RI_criv = RI_criGRv + RI_criGBv
        MLRI_crih = MLRI_criGRh + MLRI_criGBh
        MLRI_criv = MLRI_criGRv + MLRI_criGBv

        # directional gradient map of iteration criteria
        RI_difcrih = RI_difcriGRh + RI_difcriGBh
        RI_difcriv = RI_difcriGRv + RI_difcriGBv
        MLRI_difcrih = MLRI_difcriGRh + MLRI_difcriGBh
        MLRI_difcriv = MLRI_difcriGRv + MLRI_difcriGBv

        # smoothing of iteration criteria (Algo 8 line 8-9)
        sigma = 2
        Fh = getGaussianKernel(5, sigma) * getGaussianKernel(5, sigma).T
        RI_crih = filter2D(RI_crih, Fh )
        MLRI_crih = filter2D(MLRI_crih, Fh )
        RI_difcrih = filter2D(RI_difcrih, Fh )
        MLRI_difcrih = filter2D(MLRI_difcrih, Fh )

        Fv = Fh 
        RI_criv = filter2D(RI_criv, Fv )
        MLRI_criv = filter2D(MLRI_criv, Fv )
        RI_difcriv = filter2D(RI_difcriv, Fv )
        MLRI_difcriv = filter2D(MLRI_difcriv, Fv )

        # calcualte iteration criteria  (Algo 8 line 10)
        RI_wh = (RI_crih ** 2) * (RI_difcrih)
        RI_wv = (RI_criv ** 2) * (RI_difcriv)
        MLRI_wh = (MLRI_crih ** 2) * (MLRI_difcrih)
        MLRI_wv = (MLRI_criv ** 2) * (MLRI_difcriv)

        # find smaller criteria pixels (criteria used in Algo 7 line 24)
        RI_pih = np.where(RI_wh < RI_w2h)
        RI_piv = np.where(RI_wv < RI_w2v)
        MLRI_pih = np.where(MLRI_wh < MLRI_w2h)
        MLRI_piv = np.where(MLRI_wv < MLRI_w2v)

        # guide updating  (Algo 7 line 22)
        RI_Guidegrh = mosaic[:, :, 1] * maskGr + RI_Grh
        RI_Guidegbh = mosaic[:, :, 1] * maskGb + RI_Gbh
        RI_Guidegh = RI_Guidegrh + RI_Guidegbh
        RI_Guiderh = mosaic[:, :, 0] + RI_Rh
        RI_Guidebh = mosaic[:, :, 2] + RI_Bh
        RI_Guidegrv = mosaic[:, :, 1] * maskGb + RI_Grv
        RI_Guidegbv = mosaic[:, :, 1] * maskGr + RI_Gbv
        RI_Guidegv = RI_Guidegrv + RI_Guidegbv
        RI_Guiderv = mosaic[:, :, 0] + RI_Rv
        RI_Guidebv = mosaic[:, :, 2] + RI_Bv
        MLRI_Guidegrh = mosaic[:, :, 1] * maskGr + MLRI_Grh
        MLRI_Guidegbh = mosaic[:, :, 1] * maskGb + MLRI_Gbh
        MLRI_Guidegh = MLRI_Guidegrh + MLRI_Guidegbh
        MLRI_Guiderh = mosaic[:, :, 0] + MLRI_Rh
        MLRI_Guidebh = mosaic[:, :, 2] + MLRI_Bh
        MLRI_Guidegrv = mosaic[:, :, 1] * maskGb + MLRI_Grv
        MLRI_Guidegbv = mosaic[:, :, 1] * maskGr + MLRI_Gbv
        MLRI_Guidegv = MLRI_Guidegrv + MLRI_Guidegbv
        MLRI_Guiderv = mosaic[:, :, 0] + MLRI_Rv
        MLRI_Guidebv = mosaic[:, :, 2] + MLRI_Bv

        # select smallest iteration criteria at each pixel (Algo 7 line 24)
        RI_Gh[RI_pih[0], RI_pih[1]] = RI_Guidegh[RI_pih[0], RI_pih[1]]
        MLRI_Gh[MLRI_pih[0], MLRI_pih[1]] = MLRI_Guidegh[MLRI_pih[0], MLRI_pih[1]]
        RI_Gv[RI_piv[0], RI_piv[1]] = RI_Guidegv[RI_piv[0], RI_piv[1]]
        MLRI_Gv[MLRI_piv[0], MLRI_piv[1]] = MLRI_Guidegv[MLRI_piv[0], MLRI_piv[1]]

        # update minimum iteration criteria (Algo 7 line 25)
        RI_w2h[RI_pih[0], RI_pih[1]] = RI_wh[RI_pih[0], RI_pih[1]]
        RI_w2v[RI_piv[0], RI_piv[1]] = RI_wv[RI_piv[0], RI_piv[1]]
        MLRI_w2h[MLRI_pih[0], MLRI_pih[1]] = MLRI_wh[MLRI_pih[0], MLRI_pih[1]]
        MLRI_w2v[MLRI_piv[0], MLRI_piv[1]] = MLRI_wv[MLRI_piv[0], MLRI_piv[1]]

        # guided filter window size update (Algo 7 line 26)
        h = h + 1
        v = v + 1
        h2 = h2 + 1
        v2 = v2 + 1

    #  Step(iii): adaptive combining 
    #  combining weight
    RI_w2h = 1 / (RI_w2h + 1e-10)
    RI_w2v = 1 / (RI_w2v + 1e-10)
    MLRI_w2h = 1 / (MLRI_w2h + 1e-10)
    MLRI_w2v = 1 / (MLRI_w2v + 1e-10)
    w = RI_w2h + RI_w2v + MLRI_w2h + MLRI_w2v

    # combining (Algo 7 line 30)
    green = (RI_w2h * RI_Gh + RI_w2v * RI_Gv + MLRI_w2h * MLRI_Gh + MLRI_w2v * MLRI_Gv) / (w + 1e-32)

    # final output
    green = green * (1-mask)[:, :, 1] + mosaic[:, :, 1]
    green = np.clip(green, 0, 255)

    return green
