#！/usr/bin/env python
#-*-coding:utf-8-*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Copyright (c) 2020 Inner Mongolia University and Centre Borelli, ENS Paris-Saclay.
#  All rights reserved.
#
#  Authors:                     Qiyu Jin
#                               Yu Guo
#                               Gabriele Facciolo
#                               Jean-Michel Morel
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# %
# %      Input
# %       - input		: full RGB image
# %       - pattern	: mosaic pattern
# %                       default : 'grbg'
# %                       others  : 'rggb','gbrg','bggr'
# %       - Algorithm	: Demosaicing Algorithm
# %                         default : 'GBTF'
# %                         others  : 'HA', 'RI', 'MLRI', 'WMLRI', 'ARI'
# %
# %      Output
# %       - out   : result image
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from mosaic_bayer import mosaic_bayer
from demosaic_ARI import demosaic_ARI
from demosaic_HA import demosaic_HA
from demosaic_RI import demosaic_RI


def demosaick(rgb, pattern, sigma, Algorithm):
    """
    wrapper for calling different demosaicking algorithms ('ARI', 'HA', 'GBTF', 'RI', 'MLRI', 'WMLRI')
    """

    # mosaic and mask
    mosaic, mask = mosaic_bayer(rgb, pattern)

    if Algorithm == 'ARI':
        rgb_dem = demosaic_ARI(mosaic, pattern)

    elif Algorithm == 'HA':
        rgb_dem = demosaic_HA(mosaic, pattern)

    else: # ('GBTF', 'RI', 'MLRI', 'WMLRI')  
        rgb_dem = demosaic_RI(mosaic, pattern, sigma, Algorithm)

    return rgb_dem

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

def main(args):
    from skimage.io import imread, imsave  # hwc
    import numpy as np
    # read image
    rgb_orig = imread(args.input).astype('float32')

    # add simulated noise
    np.random.seed(2021)
    rgb = rgb_orig + np.random.randn(*rgb_orig.shape) * args.noise_sigma

    # mosaic pattern
    # G R ..
    # B G ..
    # : :
    pattern = args.pattern

    # demosaicking
    #sigma = 1  # sigma : standard deviation of gaussian filter(default : 1) * For Kodak image data set, 1e8 works well.
    Algorithm = args.Algorithm  # 'HA', 'RI' , 'MLRI' , 'WMLRI', 'ARI'
    tic()
    rgb_dem = demosaick(rgb, pattern, args.sigma, Algorithm)
    toc()


    # save output image
    imsave(args.output, rgb_dem.clip(0,255).astype('uint8'))

    # save the mosaic image
    if args.mosaic != "":
        mosaic, _ = mosaic_bayer(rgb, pattern)
        imsave(args.mosaic, mosaic.clip(0,255).astype('uint8'))


    # save the mosaic image
    if args.output_diff != "":
        imsave(args.output_diff, ((rgb_orig - rgb_dem)*10+128. ).clip(0,255).astype('uint8'))

    # calculate PSNR and CPSNR
    psnr = impsnr(rgb_orig, rgb_dem, 255, 10)
    cpsnr = imcpsnr(rgb_orig, rgb_dem, 255, 10)

    print('Red:{:.4f} dB'.format(psnr[0]))
    print('Green:{:.4f} dB'.format(psnr[1]))
    print('Blue:{:.4f} dB'.format(psnr[2]))
    print('CPSNR:{:.4f} dB'.format(cpsnr))

    cpsnr_path = 'cpsnr_out.txt'
    file = open(cpsnr_path, 'w')
    file.write('Red:{:.4f} dB '.format(psnr[0]))
    file.write('Green:{:.4f} dB '.format(psnr[1]))
    file.write('Green:{:.4f} dB '.format(psnr[1]))
    file.write('CPSNR:{:.4f} dB '.format(cpsnr))


if __name__ == "__main__":
        
    from impsnr import impsnr, imcpsnr
    import argparse

    Test_input = 'Sans_bruit_13.PNG'
    Test_out = 'test_GBTF.png'
    Test_pattern = 'grbg'
    Test_Algorithm = 'GBTF'

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=Test_input, help="test input, uses the default test input provided if no argument")
    parser.add_argument("--pattern", default=Test_pattern, help="bayer pattern", type=str)
    parser.add_argument("--Algorithm", default=Test_Algorithm, help="Demosaicing Algorithm", type=str)
    parser.add_argument("--output", default=Test_out, help="output image", type=str)
    parser.add_argument("--output_diff", default="", help="output difference image wrt the noiseless input", type=str)
    parser.add_argument("--noise_sigma", default=0, help="added noise standard devation", type=float)
    parser.add_argument("--mosaic", default="", help="export the noisy mosaic", type=str)
    parser.add_argument("--sigma", default=1, help="standard deviation of the regularization gaussian used in RI, MLRI, WMLRI", type=float)
    

    args = parser.parse_args()
    main(args)
