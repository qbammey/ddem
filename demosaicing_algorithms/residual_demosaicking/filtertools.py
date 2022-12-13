import cv2


def filter2D(im, ker):
    """
    convolve the 2d  image (im) with the 2d kernel (ker) and return a 2d  image
    pads the image to preserve the shape by replicating boundaries
    """
    return cv2.filter2D(im,  -1, kernel=ker, borderType=cv2.BORDER_REPLICATE)


def boxFilter(im, sz):
    """
    convolve the 2d  image (im) with a box filter of diameter sz (tuple)
    pads the image to preserve the shape by replicating boundaries
    """
    return cv2.boxFilter(im,  -1, sz, normalize=False, borderType=cv2.BORDER_CONSTANT)


def getGaussianKernel(sz,sigma):
    """
    returns a 1d Gaussian kernel with standard deviation sigma and support sz 
    """
    return cv2.getGaussianKernel(sz, sigma)
