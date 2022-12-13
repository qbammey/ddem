import numpy as np
from filtertools import filter2D



def haresidual(rawq, mask, maskGr, maskGb, mosaic):
    """
    This functions implements Algorithm 3 
    Hamilton-Adams residual used in the GBTF algorithm
    """
    # (1st step of GBTF: HA interpolation - line 11)
    # The filter f is:  1/2 K_H - 1/4 Delta_H 
    f = np.array([[-1/4, 1/2, 1/2, 1/2, -1/4]])
    rawh = filter2D(rawq, f)
    rawv = filter2D(rawq, f.T)

    maskR = mask[:, :, 0]
    maskB = mask[:, :, 2]

    # tentative image
    Grh = rawh * maskR
    Gbh = rawh * maskB
    Rh = rawh * maskGr
    Bh = rawh * maskGb
    Grv = rawv * maskR
    Gbv = rawv * maskB
    Rv = rawv * maskGb
    Bv = rawv * maskGr

    # vertical and horizontal color difference (2nd step of GBTF - line 12)
    # {Gr,Gb,R,B}h  are \tilde Q in the paper, restricted to the different mosaic phases.
    # mosaic[:,:,i] are  Q in the paper the combination below.
    # Note that the phases of {Gr,Gb,R,B}h/v are not evident from the names, for instance Rh is on maskGr
    difh = (Grh - mosaic[:, :, 0]) + (Gbh - mosaic[:, :, 2]) + (- Rh - Bh + mosaic[:, :, 1])
    difv = (Grv - mosaic[:, :, 0]) + (Gbv - mosaic[:, :, 2]) + (- Rv - Bv + mosaic[:, :, 1])
#    difh = mosaic[:, :, 1] + Grh + Gbh - mosaic[:, :, 0] - mosaic[:, :, 2] - Rh - Bh
#    difv = mosaic[:, :, 1] + Grv + Gbv - mosaic[:, :, 0] - mosaic[:, :, 2] - Rv - Bv

    ### Combine Vertical and Horizontal Color Differences ###
    # color difference gradient (first half of 3rd step of GBTF)
    Kh = np.array([[1, 0, -1]])
    Kv = Kh.T
    AvK = np.array([[1, 1, 1]])
    difh2 = filter2D ( abs(filter2D(difh, Kh)), AvK.T )
    difv2 = filter2D ( abs(filter2D(difv, Kv)), AvK   )

    return difh, difv, difh2, difv2
