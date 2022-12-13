A Demosaicking Algorithm with Adaptive Inter-channel Correlation (version 1)

Joan Duran, joan.duran@uib.es, Universitat de les Illes Balears (Spain)
Antoni Buades, toni.buades@uib.es, Universitat de les Illes Balears (Spain)


# OVERVIEW

This C source code accompanies with Image Processing On Line (IPOL) article
"A Demosaicking Algorithm with Adaptive Inter-channel Correlation" at 

    http://www.ipol.im/pub/art/

This code is used by the online IPOL demo:

    http://demo.ipol.im/demo/

This program reads and writes PNG images, but can be easily adapted to any 
other file format. Only 8bit RGB PNG images are handled. 

Two programs are provided:

* ‘demosaicking_ipol' simulates the mosaicked image using Bayer filter array
and fills in the missing components by means of the demosaicking algorithm proposed in [J. Duran, A. Buades, “Self-Similarity and Spectral Correlation Adaptive Algorithm for Color Demosaicking”, IEEE Trans. Image Process., vol. 23(9), pp 4031-4040, 2014]. The method takes advantage of image self-similarity and balances how much inter-channel correlation must be taken into account. The first step of the algorithm consists in deciding a posteriori among a set of four local directionally interpolated images. In a second step, a patch-based algorithm refines the locally interpolated image. In both cases, the process applies on channel differences instead of channels themselves.


# USAGE

Usage: demosaicking_ipol truth.png mosaicked.png demosaicked.png beta

truth.png       :: full color RGB image.
mosaicked.png   :: oberseved mosaicked image.
demosaicked.png :: full color demosaicked image.
beta            :: fixed channel-correlation parameter.

The following parameters are fixed in the main demo function:
epsilon   : thresholding parameter avoiding numerical intrincacies when
            computing local variation of chromatic components.
M         : bounding parameter above which a discontinuity of the luminance
	    gradient is considered.\n");
halfL     : half-size of the support zone where the variance of the chromatic
            components is computed.
reswind   : half-size of research window.
compwind  : half-size of comparison window.
N         : number of most similar pixels for filtering.
redx redy : coordinates of the first red value in CFA.

Usage: imdiff_ipol image1.png image2.png difference.png

image1.png     : first image.
image2.png     : second image.
difference.png : difference image.

This program also provides on the screen the RMSE values.


#LICENSE

Files io_png.c and io_png.h are copyright Nicolas Limare. These files are
distributed under the BSD license conditions described in the corresponding
headers files.

All the other files are distributed under the terms of the LGPLv3 license.


# REQUIREMENTS

The code is written in ANSI C and C++, and should compile on any system with an
ANSI C/C++ compiler.

The libpng header and libraries are required on the system for compilation and
execution. 


# COMPILATION

Simply use the provided makefile, with the command 'make'.


# EXAMPLE

   # Apply demosaicking algorithm with automatic estimation of beta:
    
   ./demosaicking_ipol image.png mosaicked.png demosaicked.png 0

   # Apply demosaicking algorithm with fixed beta:

   ./demosaicking_ipol image.png mosaicked.png demosaicked.png 1.0

   # Compute RMSE and image difference:

   ./imdiff_ipol image.png demosaicked.png difference.png