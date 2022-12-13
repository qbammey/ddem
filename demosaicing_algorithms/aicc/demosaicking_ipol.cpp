/*
 * Copyright 2009-2015 IPOL Image Processing On Line http://www.ipol.im/
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @mainpage A Demosaicking Algorithm with Adaptive Inter-channel Correlation
 *           (version 1)
 *
 * README.txt:
 * @verbinclude README.txt
 */

/**
 * @file   demosaicking_ipol.cpp
 * @brief  Main executable file
 *
 * @author Joan Duran <joan.duran@uib.es>
 */

#include "libdemosaicking.h"
#include "libauxiliar.h"
#include "io_png.h"
#include <string.h>

// Usage: demosaicking_ipol truth.png mosaicked.png demosaicked.png beta

int main(int argc, char **argv)
{
    if(argc < 6)
	{
        printf("usage: demosaicking_ipol truth.png mosaicked.png "
               "demosaicked.png beta redy redx\n\n");
        printf("truth.png       :: full color RGB image.\n");
        printf("mosaicked.png   :: oberseved mosaicked image.\n");
        printf("demosaicked.png :: full color demosaicked image.\n");
        printf("redy redx :: coordinates of the first red value in CFA.\n");
        printf("\n");
        printf("The following parameters are fixed in the main demo "
               "function:\n");
		printf("beta            :: fixed channel-correlation parameter.\n");
        printf("epsilon   :: thresholding parameter avoiding numerical\n"
               "             intrincacies when computing local variation of\n"
               "             chromatic components.\n");
        printf("M         :: bounding parameter above which a discontinuity\n"
               "             of the luminance gradient is considered.\n");
        printf("halfL     :: half-size of the support zone where the variance\n"
               "             of the chromatic components is computed.\n");
        printf("reswind   :: half-size of research window.\n");
        printf("compwind  :: half-size of comparison window.\n");
        printf("N         :: number of most similar pixels for filtering.\n");
	    
        return EXIT_FAILURE;
	}

    // Read full color reference image
    size_t nx, ny, nc;
    float *d_v = NULL;
    
    d_v = io_png_read_f32(argv[1], &nx, &ny, &nc);
    
    if(!d_v)
	{
	    fprintf(stderr, "Error - %s not found  or not a correct png image.\n",
                argv[1]);
        return EXIT_FAILURE;
    }
    
    if(nc > 3)  // Use only RGB image
        nc = 3;
    
    // Variables of reference image
    int width = (int) nx;
    int height = (int) ny;
    int num_channels = (int) nc;
    int dim = width * height;
    int dimchannels = num_channels * dim;
    
    float **truth = new float*[num_channels];
    for(int c = 0; c < num_channels; c++)
        truth[c] = &d_v[c*dim];
    
    // Input parameters
    float beta = 1.0f; //atof(argv[4]);
    
    if((beta < 0.0f) || (beta > 1.0f))
    {
        fprintf(stderr, "Error - beta must be in range (0,1].\n");
        return EXIT_FAILURE;
    }
    
    // Compute h in terms of beta if not automatically determined
    float h = 0.0f;
    
    if(beta != 0.0f)
        h = (310.0f * beta - 214.0f) / 3.0f;
    
    // Fixed parameters
    float epsilon = 0.00000001f;
    float M = 13.0f;
    int halfL = 1;
    int reswind = 10;
    int compwind = 1;
    int N = 10;
    int redx = atoi(argv[4]);
    int redy = atoi(argv[5]);
    
    
    // Simulate mosaicked image
    float **mosaicked = new float*[num_channels];
    for(int c = 0; c < num_channels; c++)
        mosaicked[c] = new float[dim];
    
    if(CFAimage(truth[0], truth[1], truth[2], mosaicked[0], mosaicked[1],
                mosaicked[2], redx, redy, width, height) != 1)
        return EXIT_FAILURE;

    
	// Demosaicking process
    float **demosaicked = new float*[num_channels];
    for(int c = 0; c < num_channels; c++)
        demosaicked[c] = new float[dim];
    
    if(algorithm_chain(mosaicked[0], mosaicked[1], mosaicked[2], demosaicked[0],
                       demosaicked[1], demosaicked[2], beta, h, epsilon, M,
                       halfL, reswind, compwind, N, redx, redy, width,
                       height) != 1)
        return EXIT_FAILURE;
    
    // Save mosaicked image in png format
    float *image_png = new float[dimchannels];
    int k = 0;
    for(int c = 0; c < num_channels; c++)
        for(int i = 0; i < dim; i++)
        {
            image_png[k] = mosaicked[c][i];
            k++;
        }
    
    if(io_png_write_f32(argv[2], image_png, (size_t) width, (size_t) height,
                        (size_t) num_channels) != 0)
    {
        fprintf(stderr, "Error - Failed to save png image %s.\n", argv[2]);
        return EXIT_FAILURE;
    }
    
    // Save demosaicked image in png format
    k = 0;
	for(int c = 0; c < num_channels; c++)
        for(int i = 0; i < dim; i++)
        {
            image_png[k] = demosaicked[c][i];
            k++;
        }
    
    if(io_png_write_f32(argv[3], image_png, (size_t) width, (size_t) height,
                        (size_t) num_channels) != 0)
    {
        fprintf(stderr, "Error - Failed to save png image %s.\n", argv[3]);
        return EXIT_FAILURE;
    }
    
	// Delete allocated memory
    delete[] image_png;
    delete[] truth;
    free(d_v);
    
    for(int c = 0; c < num_channels; c++)
    {
        delete[] mosaicked[c];
        delete[] demosaicked[c];
    }
    
    delete[] mosaicked;
    delete[] demosaicked;
    
	return EXIT_SUCCESS;
}
