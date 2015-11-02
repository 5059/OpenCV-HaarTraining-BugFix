/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
 * createsamples.cpp
 *
 * Create test/training samples
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>

using namespace std;

#include "cvhaartraining.h"

/****************************************************************************/
// DANIEL_BUG_FIX - added new function to create samples from list of images + apply few distortions on each image
int cvCreateTrainingSamplesFromInfo_add_distortion( const char* vecfilename,const char* infoname, int bgcolor, int bgthreshold, const char* bgfilename, 
							  int nReplication, int invert, int maxintensitydev, double maxxangle, double maxyangle, double maxzangle, int showsamples, int winwidth, int winheight);

/****************************************************************************/
int main( int argc, char* argv[] ){
    int i = 0;
    char* nullname   = (char*)"(NULL)";
    char* vecname    = NULL; /* .vec file name */
    char* infoname   = NULL; /* file name with marked up image descriptions */
    char* imagename  = NULL; /* single sample image */
    char* bgfilename = NULL; /* background */
    int num = 1000;
    int bgcolor = 0;
    int bgthreshold = 80;
    int invert = 0;
    int maxintensitydev = 40;
    double maxxangle = 1.1;
    double maxyangle = 1.1;
    double maxzangle = 0.5;
    int showsamples = 0;
    /* the samples are adjusted to this scale in the sample preview window */
    double scale = 4.0;
    int width  = 24;
    int height = 24;
	char interpType = 'c'; // DANIEL_BUG_FIX - added Another parameter to select Daniel's Interpolation method. Default is 'opencv'

    srand((unsigned int)time(0));

    if( argc == 1 )
    {
        printf( "Usage: %s\n  [-info <collection_file_name>]\n"
                "  [-img <image_file_name>]\n"
                "  [-vec <vec_file_name>]\n"
                "  [-bg <background_file_name>]\n  [-num <number_of_samples = %d>]\n"
                "  [-bgcolor <background_color = %d>]\n"
                "  [-inv] [-randinv] [-bgthresh <background_color_threshold = %d>]\n"
                "  [-maxidev <max_intensity_deviation = %d>]\n"
                "  [-maxxangle <max_x_rotation_angle = %f>]\n"
                "  [-maxyangle <max_y_rotation_angle = %f>]\n"
                "  [-maxzangle <max_z_rotation_angle = %f>]\n"
                "  [-show [<scale = %f>]]\n"
                "  [-w <sample_width = %d>]\n  [-h <sample_height = %d>]\n"
				"  [-interp <How to interpolate pixels of the samples: cv/daniel>]\n",
                argv[0], num, bgcolor, bgthreshold, maxintensitydev,
                maxxangle, maxyangle, maxzangle, scale, width, height );
				// DANIEL_BUG_FIX - added Another parameter to select Daniel's Interpolation method

        return 0;
    }

    for( i = 1; i < argc; ++i ){
        if( !strcmp( argv[i], "-info" )){
            infoname = argv[++i];
        }
        else if( !strcmp( argv[i], "-img" )){
            imagename = argv[++i];
        }
        else if( !strcmp( argv[i], "-vec" )){
            vecname = argv[++i];
        }
        else if( !strcmp( argv[i], "-bg" )){
            bgfilename = argv[++i];
        }
        else if( !strcmp( argv[i], "-num" )){
            num = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-bgcolor" )){
            bgcolor = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-bgthresh" )){
            bgthreshold = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-inv" )){
            invert = 1;
        }
        else if( !strcmp( argv[i], "-randinv" )){
            invert = CV_RANDOM_INVERT;
        }
        else if( !strcmp( argv[i], "-maxidev" )){
            maxintensitydev = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-maxxangle" )){
            maxxangle = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-maxyangle" )){
            maxyangle = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-maxzangle" )){
            maxzangle = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-show" )){
            showsamples = 1;
            if( i+1 < argc && strlen( argv[i+1] ) > 0 && argv[i+1][0] != '-' ){
                double d;
                d = strtod( argv[i+1], 0 );
                if( d != -HUGE_VAL && d != HUGE_VAL && d > 0 ) scale = d;
                ++i;
            }
        }
        else if( !strcmp( argv[i], "-w" )){
            width = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-h" )){
            height = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-interp" )){
            interpType = (argv[++i])[0]; // DANIEL_BUG_FIX - added Another parameter to select Daniel's Interpolation method. Default is 'c' - 'opencv', 'd' - 'daniel
        }
    }

    printf( "Info file name: %s\n", ((infoname == NULL) ?   nullname : infoname ) );
    printf( "Img file name: %s\n",  ((imagename == NULL) ?  nullname : imagename ) );
    printf( "Vec file name: %s\n",  ((vecname == NULL) ?    nullname : vecname ) );
    printf( "BG  file name: %s\n",  ((bgfilename == NULL) ? nullname : bgfilename ) );
    printf( "Num: %d\n", num );
    printf( "BG color: %d\n", bgcolor );
    printf( "BG threshold: %d\n", bgthreshold );
    printf( "Invert: %s\n", (invert == CV_RANDOM_INVERT) ? "RANDOM" : ( (invert) ? "TRUE" : "FALSE" ) );
    printf( "Max intensity deviation: %d\n", maxintensitydev );
    printf( "Max x angle: %g\n", maxxangle );
    printf( "Max y angle: %g\n", maxyangle );
    printf( "Max z angle: %g\n", maxzangle );
    printf( "Show samples: %s\n", (showsamples) ? "TRUE" : "FALSE" );
    if (showsamples){
        printf( "Scale: %g\n", scale );
    }
    printf( "Width: %d\n", width );
    printf( "Height: %d\n", height );

    // determine action 
    if( imagename && (!infoname) && vecname ){   // DANIEL_BUG_FIX added (!infoname) so won't be confusion anymore
        printf( "Create training samples from single image applying distortions...\n" );
        cvCreateTrainingSamples( vecname, imagename, bgcolor, bgthreshold, bgfilename,
                                 num, invert, maxintensitydev, maxxangle, maxyangle, maxzangle, showsamples, width, height );
        printf( "Done\n" );
    }
    else if( imagename && bgfilename && infoname && (!vecname)){  // DANIEL_BUG_FIX added (!vecname) so won't be confusion anymore
        printf( "Create test samples from single image applying distortions...\n" );
        cvCreateTestSamples( infoname, imagename, bgcolor, bgthreshold, bgfilename, 
							 num,invert, maxintensitydev, maxxangle, maxyangle, maxzangle, showsamples, width, height );
        printf( "Done\n" );
    }
    else if( (!imagename) && infoname && vecname ){  // DANIEL_BUG_FIX added (!imagename) so won't be confusion anymore
        int total;
        printf( "Create training samples from images collection...\n" );
		printf( "Image interpolation: %s\n", (interpType=='d') ? "daniels" : "built in opencv" ); // DANIEL_BUG_FIX - added Another parameter to select Daniel's Interpolation method. Default is 'c' - 'opencv', 'd' - 'daniel
        total = cvCreateTrainingSamplesFromInfo( infoname, vecname, num, showsamples, width, height, interpType );
        printf( "Done. Created %d samples\n", total );
    }
	// DANIEL_BUG_FIX - call to added new function to create samples from list of images + apply few distortions on each image
    else if( imagename && infoname && vecname ){
        int total;
		// imagename - is not used, it must be supplied to differentiate this case from cvCreateTrainingSamplesFromInfo();
        printf( "Daniels, new function!!!! Create training samples from images collection, applying %d distortions to each image...\n",num);
        total = cvCreateTrainingSamplesFromInfo_add_distortion( vecname, infoname, bgcolor, bgthreshold, bgfilename,
                                 num, invert, maxintensitydev, maxxangle, maxyangle, maxzangle, showsamples, width, height);
        printf( "Done. Created %d samples (%d orig samples x %d replications)\n", total, total/num, num);
    }
	// DANIEL_BUG_FIX - END
	else if( (!imagename) && (!infoname) && vecname ){ // DANIEL_BUG_FIX added (!imagename) && (!infoname) so won't be confusion anymore
        printf( "View samples from vec file (press ESC to exit)...\n" );
        cvShowVecSamples( vecname, width, height, scale );
        printf( "Done\n" );
    }
    else{
        printf( "Illegal arguments, Nothing to do\n" );
    }
    return 0;
}

/****************************************************************************/
