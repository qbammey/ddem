import numpy as np
from mosaic_bayer import mosaic_bayer, get_mosaic_masks
from filtertools import filter2D


# This functions implements Algorithm 1
def hagreen_interpolation(mosaic, mask):
    """
    hamilton-adams green channel processing
    """
    Kh = np.array([[1/2, 0, 1/2]])  
    Kv = Kh.T
    Deltah = np.array([[1, 0, -2, 0, 1]])
    Deltav = Deltah.T

    Diffh = np.array([[1, 0, -1]])
    Diffv = Diffh.T

    Diffh =np.array([[1, 0, -1]])
    Diffv = Diffh.T

    rawq = np.sum(mosaic, axis=2) #    get the raw CFA data

    rawh = filter2D( rawq, Kh  ) - filter2D( rawq, Deltah/4 )
    rawv = filter2D( rawq, Kv  ) - filter2D( rawq, Deltav/4  )
    CLh = np.abs( filter2D(rawq, Diffh) ) + np.abs( filter2D(rawq, Deltah) ) 
    CLv = np.abs( filter2D(rawq, Diffv) ) + np.abs( filter2D(rawq, Deltav) ) 

    # this implements the logic assigning rawh  when CLv > CLh
    #                                     rawv  when CLv < CLh;
    #                                     (rawh+rawv)/2 otherwise
    CLlocation = np.sign(CLh - CLv)
    green = (1 + CLlocation) * rawv / 2 + (1 - CLlocation) * rawh / 2

    green = green * (1-mask[:, :, 1]) + rawq * mask[:, :, 1]

    return green




# This functions implements Algorithm 2 (red pixels)
def hared_interpolation(green, mosaic, mask, pattern):
    """
    hamilton-adams red channel processing
    """
    # mask
    maskGr, maskGb, _, maskB = get_mosaic_masks(mosaic,pattern)

    Kh = np.array([[1, 0, 1]])
    Kv = Kh.T
    Kp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    Kn = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

    Deltap = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]])
    Deltan = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]])

    Deltah = np.array([[1, -2, 1]])
    Deltav = Deltah.T

    # these filters are the diagonal filters
    Diffp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])
    Diffn = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])

    mosaicR = mosaic[:,:,0]

    Rh  = maskGr * ( 0.5 * filter2D( mosaicR, Kh ) - 0.25 * filter2D( green, Deltah ))
    Rv  = maskGb * ( 0.5 * filter2D( mosaicR, Kv ) - 0.25 * filter2D( green, Deltav ))
    Rp  = maskB  * ( 0.5 * filter2D( mosaicR, Kp ) - 0.25 * filter2D( green, Deltap ))
    Rn  = maskB  * ( 0.5 * filter2D( mosaicR, Kn ) - 0.25 * filter2D( green, Deltan ))
    CLp = maskB  * (np.abs( filter2D( mosaicR, Diffp )) + np.abs( filter2D( green, Deltap )) )
    CLn = maskB  * (np.abs( filter2D( mosaicR, Diffn )) + np.abs( filter2D( green, Deltan )) )

    CLlocation = np.sign(CLp - CLn)
    red = (1 + CLlocation) * Rn / 2 + (1 - CLlocation) * Rp / 2
    red = red + Rh + Rv + mosaicR

    return red







# This functions implements Algorithm 2 (blue pixels)
def hablue_interpolation(green, mosaic, mask, pattern):
    """
    hamilton-adams blue channel processing
    """
    # masks
    maskGr, maskGb, maskR, _ = get_mosaic_masks(mosaic,pattern)

    Kh = np.array([[1, 0, 1]])
    Kv = Kh.T
    Kp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    Kn = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

    Deltap = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]])
    Deltan = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]])

    Deltah = np.array([[1, -2, 1]])
    Deltav = Deltah.T

    # these filters are the diagonal filters
    Diffp = np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]])
    Diffn = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])

    mosaicB = mosaic[:,:,2]

    Bh  = maskGb * ( 0.5 * filter2D( mosaicB, Kh ) - 0.25 * filter2D( green, Deltah ))
    Bv  = maskGr * ( 0.5 * filter2D( mosaicB, Kv ) - 0.25 * filter2D( green, Deltav ))
    Bp  = maskR  * ( 0.5 * filter2D( mosaicB, Kp ) - 0.25 * filter2D( green, Deltap ))
    Bn  = maskR  * ( 0.5 * filter2D( mosaicB, Kn ) - 0.25 * filter2D( green, Deltan ))
    CLp = maskR  * (np.abs( filter2D( mosaicB, Diffp )) + np.abs( filter2D( green, Deltap )) )
    CLn = maskR  * (np.abs( filter2D( mosaicB, Diffn )) + np.abs( filter2D( green, Deltan )) )

    CLlocation = np.sign(CLp - CLn)
    blue = (1 + CLlocation) * Bn / 2 + (1 - CLlocation) * Bp / 2
    blue = blue + Bh + Bv + mosaicB

    return blue







def demosaic_HA(mosaic, pattern):
    """
    Hamilton-Adams demosaicing main function
    """

    # mosaic and mask (just to generate the mask)
    mosaic, mask = mosaic_bayer(mosaic, pattern)


    # green interpolation (implements Algorithm 1)
    green = hagreen_interpolation(mosaic, mask)
    green = np.clip(green, 0, 255)

    # Red and Blue demosaicing (implements Algorithm 2)
    red = hared_interpolation(green, mosaic, mask, pattern)
    blue = hablue_interpolation(green, mosaic, mask, pattern)
    red = np.clip(red, 0, 255)
    blue = np.clip(blue, 0, 255)

    # result image

    rgb_size = mosaic.shape
    rgb_dem = np.zeros((rgb_size[0], rgb_size[1], 3))
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue

    return rgb_dem





if __name__ == "__main__":
    from skimage.io import imread, imsave
    rgb = imread('Sans_bruit_13.PNG')
    rgb = rgb.astype('float32')
    pattern = 'grbg'

    # generate the mosaic
    mosaic, _ = mosaic_bayer(rgb, pattern)

    # call demosaicing
    rgb_dem = demosaic_HA(mosaic, pattern)
    imsave('test_HA.tiff', rgb_dem.astype('uint8'))
