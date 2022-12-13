/**
 * @file dmzhangwucli.c 
 * @brief Zhang-Wu LMMSE Image Demosaicking command line program
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * 
 * Copyright (c) 2010-2011, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or 
 * redistribute it under the terms of the simplified BSD License. You 
 * should have received a copy of this license along this program. If 
 * not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * @mainpage
 * @htmlinclude readme.html
 */

#include <math.h>
#include <string.h>
#include <ctype.h>

#include "imageio.h"
#include "dmbilinear.h"
#include "dmzhangwu.h"


/** @brief struct of program parameters */
typedef struct
{
    /** @brief Input file name */
    char *InputFile;
    /** @brief Output file name */
    char *OutputFile;
    /** @brief Quality for saving JPEG images (0 to 100) */
    int JpegQuality;
    /** @brief CFA pattern upperleftmost red pixel x-coordinate */
    int RedX;
    /** @brief CFA pattern upperleftmost red pixel y-coordinate */
    int RedY;
    /** @brief UseZhangCodeEst flag for how to estimate local signal mean */
    int UseZhangCodeEst;
} programparams;


static int ParseParams(programparams *Param, int argc, char *argv[]);


/** @brief Print program usage help message */
static void PrintHelpMessage()
{
    printf("Zhang-Wu LMMSE demosaicing demo, P. Getreuer 2010-2011\n\n");
    printf("Usage: dmzhangwu [options] <input file> <output file>\n\n"
        "Only " READIMAGE_FORMATS_SUPPORTED " images are supported.\n\n");
    printf("Options:\n");
    printf("   -p <pattern>   CFA pattern, choices for <pattern> are\n");
    printf("                  RGGB        upperleftmost red pixel is at (0,0)\n");
    printf("                  GRBG        upperleftmost red pixel is at (1,0)\n");
    printf("                  GBRG        upperleftmost red pixel is at (0,1)\n");
    printf("                  BGGR        upperleftmost red pixel is at (1,1)\n");
    printf("   -e <estimate>  Estimation of the local signal mean, choices for\n");
    printf("                  <estimate> are\n");
    printf("                  paper       as in Zhang and Wu's paper (default)\n");
    printf("                  zhangcode   as in Zhang's MATLAB code\n");
#ifdef LIBJPEG_SUPPORT
    printf("   -q <number>    Quality for saving JPEG images (0 to 100)\n\n");
#endif
    printf("Example:\n"
        "   dmzhangwu -p RGGB frog.bmp frog-dm.bmp\n");
}


int main(int argc, char *argv[])
{
    programparams Param;
    float *Input = NULL, *Output = NULL;
    int Width, Height, Status = 1;
    
    
    if(!ParseParams(&Param, argc, argv))
        return 0;

    /* Read the input image */
    if(!(Input = (float *)ReadImage(&Width, &Height, 
        Param.InputFile, IMAGEIO_FLOAT | IMAGEIO_RGB | IMAGEIO_PLANAR)))
        goto Catch;
    
    if(Width < 4 || Height < 4)
    {
        ErrorMessage("Image is too small (%dx%d).\n", Width, Height);
        goto Catch;
    }
    
    if(!(Output = (float *)Malloc(sizeof(float)*3*
        ((long int)Width)*((long int)Height))))
        goto Catch;
    
    /* Flatten the input to a 2D array */
    CfaFlatten(Input, Input, Width, Height, Param.RedX, Param.RedY);
    
    /* Perform demosaicing */    
    if(!(ZhangWuDemosaic(Output, Input, Width, Height, 
        Param.RedX, Param.RedY, Param.UseZhangCodeEst)))
        goto Catch;
    
    /* Write the output image */
    if(!WriteImage(Output, Width, Height, Param.OutputFile, 
        IMAGEIO_FLOAT | IMAGEIO_RGB | IMAGEIO_PLANAR, Param.JpegQuality))
        goto Catch;
    
    Status = 0; /* Finished successfully, set exit status to zero. */
Catch:
    Free(Output);
    Free(Input);
    return Status;
}


static int ParseParams(programparams *Param, int argc, char *argv[])
{
    static char *DefaultOutputFile = (char *)"out.bmp";
    char *OptionString;
    char OptionChar;
    int i;

    
    if(argc < 2)
    {
        PrintHelpMessage();
        return 0;
    }

    /* Set parameter defaults */
    Param->InputFile = 0;
    Param->OutputFile = DefaultOutputFile;
    Param->JpegQuality = 80;
    Param->RedX = 0;
    Param->RedY = 0;
    Param->UseZhangCodeEst = 0;
    
    for(i = 1; i < argc;)
    {
        if(argv[i] && argv[i][0] == '-')
        {
            if((OptionChar = argv[i][1]) == 0)
            {
                printf("Invalid parameter format.\n");
                return 0;
            }

            if(argv[i][2])
                OptionString = &argv[i][2];
            else if(++i < argc)
                OptionString = argv[i];
            else
            {
                printf("Invalid parameter format.\n");
                return 0;
            }
            
            switch(OptionChar)
            {
            case 'p':
                if(!strcmp(OptionString, "RGGB") 
                    || !strcmp(OptionString, "rggb"))
                {
                    Param->RedX = 0;
                    Param->RedY = 0;
                }
                else if(!strcmp(OptionString, "GRBG") 
                    || !strcmp(OptionString, "grbg"))
                {
                    Param->RedX = 1;
                    Param->RedY = 0;
                }
                else if(!strcmp(OptionString, "GBRG") 
                    || !strcmp(OptionString, "gbrg"))
                {
                    Param->RedX = 0;
                    Param->RedY = 1;
                }
                else if(!strcmp(OptionString, "BGGR") 
                    || !strcmp(OptionString, "bggr"))
                {
                    Param->RedX = 1;
                    Param->RedY = 1;
                }
                else
                    printf("CFA pattern must be RGGB, GRBG, GBRG, or BGGR.\n");
                break;
            case 'e':
                if(!strcmp(OptionString, "paper"))
                    Param->UseZhangCodeEst = 0;
                else if(!strcmp(OptionString, "zhangcode"))
                    Param->UseZhangCodeEst = 1;
                else
                    printf("Estimation must be paper or zhangcode.\n");
                break;
#ifdef LIBJPEG_SUPPORT
            case 'q':
                Param->JpegQuality = atoi(OptionString);

                if(Param->JpegQuality <= 0 || Param->JpegQuality > 100)
                {
                    printf("JPEG quality must be between 0 and 100.\n");
                    return 0;
                }
                break;
#endif
            case '-':
                PrintHelpMessage();
                return 0;
            default:
                if(isprint(OptionChar))
                    printf("Unknown option \"-%c\".\n", OptionChar);
                else
                    printf("Unknown option.\n");

                return 0;
            }

            i++;
        }
        else
        {
            if(!Param->InputFile)
                Param->InputFile = argv[i];
            else
                Param->OutputFile = argv[i];

            i++;
        }
    }
    
    if(!Param->InputFile)
    {
        PrintHelpMessage();
        return 0;
    }
    
    return 1;
}
