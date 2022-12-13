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
 * Visualizes the difference between two images in such a way that
 * the error range is linearly transformed from [-20, 20] to [0, 255] for
 * visualization purposes. Errors outside this range are saturated to 0 and 255,
 * respectively. It also computes the Root Mean Squared Error with boundary
 * elimination of 6 pixels:
 *
 * @f$ RMSE = \frac{1}{3} \sum_{m=1}^{3} \sqrt{\frac{\sum_{p\in I}
 * (u(p)-v(p))^2}{N^2}} @f$,
 *
 * where @f$ I @f$ denotes the discrete grid and @f$ N^2 @f$ is the number of
 * pixels of both images.
 *
 */

/**
 * @file  imdiff_ipol.cpp
 * @brief  Main executable file
 *
 * @author Joan Duran <joan.duran@uib.es>
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "io_png.h"

// Usage: imdiff_ipol image1.png image2.png difference.png

int main(int argc, char **argv)
{
    if(argc < 4) 
	{
        printf("usage: imdiff_ipol image1.png image2.png difference.png\n");
        printf("image1.png     :: first image.\n");
        printf("image2.png     :: second image.\n");
        printf("difference.png :: difference image.\n");
	    
        return EXIT_FAILURE;
	}

    // Read first input
    size_t nx, ny, nc;
    float *d_v = NULL;

    d_v = io_png_read_f32(argv[1], &nx, &ny, &nc);

	if(!d_v)
	{
        fprintf(stderr, "Error - %s not found  or not a correct png image.\n", argv[1]);
        return EXIT_FAILURE;
    }

    if(nc == 2)  // We do not use the alpha channel if grayscale image
	    nc = 1;

    if(nc > 3)   // We do not use the alpha channel if RGB image
	    nc = 3;

    // Read second input
   	size_t nx2, ny2, nc2;
    float *d_v2 = NULL;

	d_v2 = io_png_read_f32(argv[2], &nx2, &ny2, &nc2);

    if(!d_v2)
	{
	    fprintf(stderr, "Error - %s not found  or not a correct png image.\n",
                argv[2]);
        return EXIT_FAILURE;
    }

    if(nc2 == 2)  // We do not use the alpha channel if grayscale image
	    nc2 = 1;

    if(nc2 > 3)   // We do not use the alpha channel if RGB image
        nc2 = 3;

    if(nx != nx2 || ny != ny2)
	{   
        // Check if both images have same size
	    fprintf(stderr, "Error - %s and %s sizes don't match.\n", argv[1],
                argv[2]);
       	return EXIT_FAILURE;
    }

	if(nc != nc2)
	{
        // Check if both images have same number of channels
		fprintf(stderr, "Error - %s and %s channels don't match.\n", argv[1],
                argv[2]);
       	return EXIT_FAILURE;
    }
    
    // Image variables
    int d_w = (int) nx;
    int d_h = (int) ny;
    int d_c = (int) nc;
    int d_wh = d_w * d_h;
    int d_whc = d_c * d_wh;
    
    // Compute image difference. Convert from [-20, 20] to [0,255]
   	float *difference = new float[d_whc];
    
    for(int i = 0; i < d_whc; i++)
    {
        float value = d_v[i] - d_v2[i];
        value =  (value + 20.0f) * 255.0f / 40.0f;
            
        if(value < 0.0)   value = 0.0f;
        if(value > 255.0) value = 255.0f;
            
        difference[i] = value;
    }
    
    // Saving difference as png image
    if (io_png_write_f32(argv[3], difference, (size_t) d_w, (size_t) d_h,
                         (size_t) d_c) != 0)
        fprintf(stderr, "Error - Failed to save png image %s.\n", argv[3]);
    
    // Boundary elimination
    int bound = 6;
    int width = d_w - 2 * bound;
    int height = d_h - 2 * bound;
    int dim = width * height;
    
    float **image1 = new float*[d_c];
    float **image2 = new float*[d_c];
    
    for(int c = 0; c < d_c; c++)
    {
        image1[c] = new float[dim];
        image2[c] = new float[dim];
        
        int n = 0;
        int l = c * d_wh + bound * d_w + bound;
        
        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++, n++, l++)
            {
                image1[c][n] = d_v[l];
                image2[c][n] = d_v2[l];
            }
            
            l += d_w - width;
        }        
    }
    
	// Compute RMSE
    float fRMSE = 0.0f;

    for(int c = 0; c < d_c ;  c++)
    {
        float fDist = 0.0f;
        
        for(int i = 0; i < dim; i++)
		{
          	float dif = image1[c][i] - image2[c][i];
            fDist += dif * dif;
		}
        
        fDist /= (float) dim;
        
        fRMSE += sqrt(fDist);
    }

    fRMSE /= (float) d_c;

    printf("RMSE: %2.2f\n", fRMSE);

    
	// Delete allocated memory
    free(d_v);
    free(d_v2);
    
    for (int c = 0; c < d_c; c++)
    {
        delete[] image1[c];
        delete[] image2[c];
    }
    
    delete[] image1;
    delete[] image2;
	delete[] difference;

    return EXIT_SUCCESS;
}
