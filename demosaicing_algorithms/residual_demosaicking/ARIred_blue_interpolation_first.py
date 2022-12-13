import numpy as np
from ARIguidedfilter import guidedfilter
from ARIguidedfilter_MLRI import guidedfilter_MLRI
from filtertools import filter2D, getGaussianKernel



# This functions implements Algorithm 9
def ARIred_blue_interpolation_first(green, mosaic, mask, eps):
    """
    red and blue interpolation for the ARI (Adaptive Residual Interpolation) demosaicking algorithm
    Arguments: 
        green: image containing the interpolated green channel
        mosaic: 3 channel image containing the R G B mosaic
        mask: 3 channel image indicating where the mosaic is set
        eps: regularization parameter (recommended: 1e-10)
    Returns: 
        red,blue: the interpolated red and blue channels    
    """
    # inverse mask
    imask = (mask == 0)
    imask = imask.astype('float32')

    # ##### Iterpolate R at B pixels and B at R pixels
    # Step (i): iterative directional interpolation
    # initial linear interpolation
    F1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]) / 2
    Guider1 = mosaic[:, :, 0] + filter2D(mosaic[:, :, 0], F1 ) * mask[:, :, 2]
    Guideg1 = green * imask[:, :, 1]
    Guideb1 = mosaic[:, :, 2] + filter2D(mosaic[:, :, 2], F1 ) * mask[:, :, 0]

    F2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / 2
    Guider2 = mosaic[:, :, 0] + filter2D(mosaic[:, :, 0], F2 ) * mask[:, :, 2]
    Guideg2 = green * imask[:, :, 1]
    Guideb2 = mosaic[:, :, 2] + filter2D(mosaic[:, :, 2], F2 ) * mask[:, :, 0]

    # initial guided filter window size for RI
    h = 2
    v = 2
    # initial guided filter window size for MLRI
    h2 = 2
    v2 = 0
    # maximum iteration number
    itnum = 2

    # initialization of iteration criteria
    RI_w2R1 = np.ones(mask[:, :, 0].shape) * 1e32
    RI_w2R2 = np.ones(mask[:, :, 0].shape) * 1e32
    MLRI_w2R1 = np.ones(mask[:, :, 0].shape) * 1e32
    MLRI_w2R2 = np.ones(mask[:, :, 0].shape) * 1e32
    RI_w2B1 = np.ones(mask[:, :, 0].shape) * 1e32
    RI_w2B2 = np.ones(mask[:, :, 0].shape) * 1e32
    MLRI_w2B1 = np.ones(mask[:, :, 0].shape) * 1e32
    MLRI_w2B2 = np.ones(mask[:, :, 0].shape) * 1e32

    # initial guide image for RI/MLRI
    RI_Guideg1 = Guideg1
    RI_Guider1 = Guider1
    RI_Guideb1 = Guideb1
    RI_Guideg2 = Guideg2
    RI_Guider2 = Guider2
    RI_Guideb2 = Guideb2
    MLRI_Guideg1 = Guideg1
    MLRI_Guider1 = Guider1
    MLRI_Guideb1 = Guideb1
    MLRI_Guideg2 = Guideg2
    MLRI_Guider2 = Guider2
    MLRI_Guideb2 = Guideb2

    # initialization of interpolated R and B values
    RI_R1 = Guider1
    RI_R2 = Guider2
    MLRI_R1 = Guider1
    MLRI_R2 = Guider2
    RI_B1 = Guideb1
    RI_B2 = Guideb2
    MLRI_B1 = Guideb1
    MLRI_B2 = Guideb2

    # Iterative diagonal interpolation
    for ittime in range(itnum):
        # generate diagonal tentative estimate by RI
        RI_tentativeR1 = guidedfilter(RI_Guideg1, RI_Guider1, imask[:, :, 1], h, v, eps, direction='diag')
        RI_tentativeR2 = guidedfilter(RI_Guideg2, RI_Guider2, imask[:, :, 1], v, h, eps, direction='diag')
        RI_tentativeB1 = guidedfilter(RI_Guideg1, RI_Guideb1, imask[:, :, 1], h, v, eps, direction='diag')
        RI_tentativeB2 = guidedfilter(RI_Guideg2, RI_Guideb2, imask[:, :, 1], v, h, eps, direction='diag')

        # generate diagonal tentative estimate by MLRI
        F1 = np.array([[-1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, -1]])
        MLRI_tentativeR1 = guidedfilter_MLRI(MLRI_Guideg1, MLRI_Guider1, imask[:, :, 1], mask[:, :, 0], h2, v2, eps, direction='diag', F=F1)
        MLRI_tentativeB1 = guidedfilter_MLRI(MLRI_Guideg1, MLRI_Guideb1, imask[:, :, 1], mask[:, :, 2], h2, v2, eps, direction='diag', F=F1)

        F2 = np.array([[0, 0, 0, 0, -1], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0]])
        MLRI_tentativeR2 = guidedfilter_MLRI(MLRI_Guideg2, MLRI_Guider2, imask[:, :, 1], mask[:, :, 0], v2, h2, eps, direction='diag', F=F2)
        MLRI_tentativeB2 = guidedfilter_MLRI(MLRI_Guideg2, MLRI_Guideb2, imask[:, :, 1], mask[:, :, 2], v2, h2, eps, direction='diag', F=F2)

        # calculate residuals of RI and MLRI
        RI_residualR1 = (mosaic[:, :, 0] - RI_tentativeR1) * mask[:, :, 0]
        RI_residualB1 = (mosaic[:, :, 2] - RI_tentativeB1) * mask[:, :, 2]
        RI_residualR2 = (mosaic[:, :, 0] - RI_tentativeR2) * mask[:, :, 0]
        RI_residualB2 = (mosaic[:, :, 2] - RI_tentativeB2) * mask[:, :, 2]
        MLRI_residualR1 = (mosaic[:, :, 0] - MLRI_tentativeR1) * mask[:, :, 0]
        MLRI_residualB1 = (mosaic[:, :, 2] - MLRI_tentativeB1) * mask[:, :, 2]
        MLRI_residualR2 = (mosaic[:, :, 0] - MLRI_tentativeR2) * mask[:, :, 0]
        MLRI_residualB2 = (mosaic[:, :, 2] - MLRI_tentativeB2) * mask[:, :, 2]

        K1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]) / 2
        RI_residualR1 = filter2D(RI_residualR1, K1 )
        RI_residualB1 = filter2D(RI_residualB1, K1 )
        MLRI_residualR1 = filter2D(MLRI_residualR1, K1 )
        MLRI_residualB1 = filter2D(MLRI_residualB1, K1 )

        K2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / 2
        RI_residualR2 = filter2D(RI_residualR2, K2 )
        RI_residualB2 = filter2D(RI_residualB2, K2 )
        MLRI_residualR2 = filter2D(MLRI_residualR2, K2 )
        MLRI_residualB2 = filter2D(MLRI_residualB2, K2 )

        # add tentative estimate
        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask[:, :, 2]
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask[:, :, 0]
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask[:, :, 2]
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask[:, :, 0]
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask[:, :, 2]
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask[:, :, 0]
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask[:, :, 2]
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask[:, :, 0]

        # Step(ii): adaptive selection of iteration at each pixel
        # calculate iteration criteria
        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * imask[:, :, 1]
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * imask[:, :, 1]
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * imask[:, :, 1]
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * imask[:, :, 1]
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * imask[:, :, 1]
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * imask[:, :, 1]
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * imask[:, :, 1]
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * imask[:, :, 1]

        F1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        RI_difcriR1 = abs(filter2D(RI_criR1, F1 ))
        RI_difcriB1 = abs(filter2D(RI_criB1, F1 ))
        MLRI_difcriR1 = abs(filter2D(MLRI_criR1, F1 ))
        MLRI_difcriB1 = abs(filter2D(MLRI_criB1, F1 ))

        F2 = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
        RI_difcriR2 = abs(filter2D(RI_criR2, F2 ))
        RI_difcriB2 = abs(filter2D(RI_criB2, F2 ))
        MLRI_difcriR2 = abs(filter2D(MLRI_criR2, F2 ))
        MLRI_difcriB2 = abs(filter2D(MLRI_criB2, F2 ))

        # absolute value of iteration criteria
        RI_criR1 = abs(RI_criR1)
        RI_criB1 = abs(RI_criB1)
        RI_criR2 = abs(RI_criR2)
        RI_criB2 = abs(RI_criB2)
        MLRI_criR1 = abs(MLRI_criR1)
        MLRI_criB1 = abs(MLRI_criB1)
        MLRI_criR2 = abs(MLRI_criR2)
        MLRI_criB2 = abs(MLRI_criB2)

        # directional map of iteration criteria
        RI_criR1 = RI_criR1 + RI_criB1
        RI_criB1 = RI_criB1 + RI_criR1
        RI_criR2 = RI_criR2 + RI_criB2
        RI_criB2 = RI_criB2 + RI_criR2
        MLRI_criR1 = MLRI_criR1 + MLRI_criB1
        MLRI_criB1 = MLRI_criB1 + MLRI_criR1
        MLRI_criR2 = MLRI_criR2 + MLRI_criB2
        MLRI_criB2 = MLRI_criB2 + MLRI_criR2

        # directional gradient map of iteration criteria
        RI_difcriR1 = RI_difcriR1 + RI_difcriB1
        RI_difcriB1 = RI_difcriB1 + RI_difcriR1
        RI_difcriR2 = RI_difcriR2 + RI_difcriB2
        RI_difcriB2 = RI_difcriB2 + RI_difcriR2
        MLRI_difcriR1 = MLRI_difcriR1 + MLRI_difcriB1
        MLRI_difcriB1 = MLRI_difcriB1 + MLRI_difcriR1
        MLRI_difcriR2 = MLRI_difcriR2 + MLRI_difcriB2
        MLRI_difcriB2 = MLRI_difcriB2 + MLRI_difcriR2

        # smoothing of iteration criteria
        sigma = 2
        F1 = getGaussianKernel(5, sigma) * getGaussianKernel(5, sigma).T
        M1 = filter2D(imask[:, :, 1], F1 )
        RI_criR1 = filter2D(RI_criR1, F1 ) / M1 * imask[:, :, 1]
        MLRI_criR1 = filter2D(MLRI_criR1, F1 ) / M1 * imask[:, :, 1]
        RI_criB1 =  filter2D(RI_criB1, F1 ) / M1 * imask[:, :, 1]
        MLRI_criB1 = filter2D(MLRI_criB1, F1 ) / M1 * imask[:, :, 1]
        RI_difcriR1 = filter2D(RI_difcriR1, F1 ) / M1 * imask[:, :, 1]
        MLRI_difcriR1 = filter2D(MLRI_difcriR1, F1 ) / M1 * imask[:, :, 1]
        RI_difcriB1 = filter2D(RI_difcriB1, F1 ) / M1 * imask[:, :, 1]
        MLRI_difcriB1 = filter2D(MLRI_difcriB1, F1 ) / M1 * imask[:, :, 1]

        F2 = getGaussianKernel(5, sigma) * getGaussianKernel(5, sigma).T
        M2 = filter2D(imask[:, :, 1], F2 )
        RI_criR2 = filter2D(RI_criR2, F2 ) / M2 * imask[:, :, 1]
        MLRI_criR2 = filter2D(MLRI_criR2, F2 ) / M2 * imask[:, :, 1]
        RI_criB2 = filter2D(RI_criB2, F2 ) / M2 * imask[:, :, 1]
        MLRI_criB2 = filter2D(MLRI_criB2, F2 ) / M2 * imask[:, :, 1]
        RI_difcriR2 = filter2D(RI_difcriR2, F2 ) / M2 * imask[:, :, 1]
        MLRI_difcriR2 = filter2D(MLRI_difcriR2, F2 ) / M2 * imask[:, :, 1]
        RI_difcriB2 = filter2D(RI_difcriB2, F2 ) / M2 * imask[:, :, 1]
        MLRI_difcriB2 = filter2D(MLRI_difcriB2, F2 ) / M2 * imask[:, :, 1]

        # calcualte iteration criteria
        RI_wR1 = (RI_criR1 ** 2) * RI_difcriR1
        RI_wR2 = (RI_criR2 ** 2) * RI_difcriR2
        MLRI_wR1 = (MLRI_criR1 ** 2) * MLRI_difcriR1
        MLRI_wR2 = (MLRI_criR2 ** 2) * MLRI_difcriR2
        RI_wB1 = (RI_criB1 ** 2) * RI_difcriB1
        RI_wB2 = (RI_criB2 ** 2) * RI_difcriB2
        MLRI_wB1 = (MLRI_criB1 ** 2) * MLRI_difcriB1
        MLRI_wB2 = (MLRI_criB2 ** 2) * MLRI_difcriB2

        # find smaller criteria pixels
        RI_piR1 = np.where(RI_wR1 < RI_w2R1)
        RI_piR2 = np.where(RI_wR2 < RI_w2R2)
        MLRI_piR1 = np.where(MLRI_wR1 < MLRI_w2R1)
        MLRI_piR2 = np.where(MLRI_wR2 < MLRI_w2R2)
        RI_piB1 = np.where(RI_wB1 < RI_w2B1)
        RI_piB2 = np.where(RI_wB2 < RI_w2B2)
        MLRI_piB1 = np.where(MLRI_wB1 < MLRI_w2B1)
        MLRI_piB2 = np.where(MLRI_wB2 < MLRI_w2B2)

        # guide updating
        RI_Guider1 = mosaic[:, :, 0] + RI_R1
        RI_Guideb1 = mosaic[:, :, 2] + RI_B1
        RI_Guider2 = mosaic[:, :, 0] + RI_R2
        RI_Guideb2 = mosaic[:, :, 2] + RI_B2
        MLRI_Guider1 = mosaic[:, :, 0] + MLRI_R1
        MLRI_Guideb1 = mosaic[:, :, 2] + MLRI_B1
        MLRI_Guider2 = mosaic[:, :, 0] + MLRI_R2
        MLRI_Guideb2 = mosaic[:, :, 2] + MLRI_B2

        # select smallest iteration criteria at each pixel
        RI_R1[RI_piR1[0], RI_piR1[1]] = RI_Guider1[RI_piR1[0], RI_piR1[1]]
        MLRI_R1[MLRI_piR1[0], MLRI_piR1[1]] = MLRI_Guider1[MLRI_piR1[0], MLRI_piR1[1]]
        RI_R2[RI_piR2[0], RI_piR2[1]] = RI_Guider2[RI_piR2[0], RI_piR2[1]]
        MLRI_R2[MLRI_piR2[0], MLRI_piR2[1]] = MLRI_Guider2[MLRI_piR2[0], MLRI_piR2[1]]
        RI_B1[RI_piB1[0], RI_piB1[1]] = RI_Guideb1[RI_piB1[0], RI_piB1[1]]
        MLRI_B1[MLRI_piB1[0], MLRI_piB1[1]] = MLRI_Guideb1[MLRI_piB1[0], MLRI_piB1[1]]
        RI_B2[RI_piB2[0], RI_piB2[1]] = RI_Guideb2[RI_piB2[0], RI_piB2[1]]
        MLRI_B2[MLRI_piB2[0], MLRI_piB2[1]] = MLRI_Guideb2[MLRI_piB2[0], MLRI_piB2[1]]

        # update minimum iteration criteria
        RI_w2R1[RI_piR1[0], RI_piR1[1]] = RI_wR1[RI_piR1[0], RI_piR1[1]]
        RI_w2R2[RI_piR2[0], RI_piR2[1]] = RI_wR2[RI_piR2[0], RI_piR2[1]]
        RI_w2B1[RI_piB1[0], RI_piB1[1]] = RI_wB1[RI_piB1[0], RI_piB1[1]]
        RI_w2B2[RI_piB2[0], RI_piB2[1]] = RI_wB2[RI_piB2[0], RI_piB2[1]]
        MLRI_w2R1[MLRI_piR1[0], MLRI_piR1[1]] = MLRI_wR1[MLRI_piR1[0], MLRI_piR1[1]]
        MLRI_w2R2[MLRI_piR2[0], MLRI_piR2[1]] = MLRI_wR2[MLRI_piR2[0], MLRI_piR2[1]]
        MLRI_w2B1[MLRI_piB1[0], MLRI_piB1[1]] = MLRI_wB1[MLRI_piB1[0], MLRI_piB1[1]]
        MLRI_w2B2[MLRI_piB2[0], MLRI_piB2[1]] = MLRI_wB2[MLRI_piB2[0], MLRI_piB2[1]]

        # guided filter window size update
        h = h + 1
        v = v + 1
        h2 = h2 + 1
        v2 = v2 + 1

    # Step(iii): adaptive combining
    # combining weight
    RI_w2R1 = 1 / (RI_w2R1 + 1e-10)
    RI_w2R2 = 1 / (RI_w2R2 + 1e-10)
    MLRI_w2R1 = 1 / (MLRI_w2R1 + 1e-10)
    MLRI_w2R2 = 1 / (MLRI_w2R2 + 1e-10)
    RI_w2B1 = 1 / (RI_w2B1 + 1e-10)
    RI_w2B2 = 1 / (RI_w2B2 + 1e-10)
    MLRI_w2B1 = 1 / (MLRI_w2B1 + 1e-10)
    MLRI_w2B2 = 1 / (MLRI_w2B2 + 1e-10)

    wR = RI_w2R1 + RI_w2R2 + MLRI_w2R1 + MLRI_w2R2
    wB = RI_w2B1 + RI_w2B2 + MLRI_w2B1 + MLRI_w2B2

    # combining
    pre_red = RI_w2R1 * RI_R1 + RI_w2R2 * RI_R2 + MLRI_w2R1 * MLRI_R1 + MLRI_w2R2 * MLRI_R2
    pre_red[(-1e-8 < pre_red) & (pre_red < 1e-8)] = 0
    red = pre_red / (wR + 1e-32)

    pre_blue = RI_w2B1 * RI_B1 + RI_w2B2 * RI_B2 + MLRI_w2B1 * MLRI_B1 + MLRI_w2B2 * MLRI_B2
    pre_blue[(-1e-8 < pre_blue) & (pre_blue < 1e-8)] = 0
    blue = pre_blue / (wB + 1e-32)

    # output of the first step
    red = red * mask[:, :, 2] + mosaic[:, :, 0]
    blue = blue * mask[:, :, 0] + mosaic[:, :, 2]

    red = np.clip(red, 0, 255)
    blue = np.clip(blue, 0, 255)

    return red, blue

