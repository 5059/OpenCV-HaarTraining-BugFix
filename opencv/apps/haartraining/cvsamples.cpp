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
 * cvsamples.cpp
 *
 * support functions for training and test samples creation.
 */

#include "cvhaartraining.h"
#include "_cvhaartraining.h"

/* if ipl.h file is included then iplWarpPerspectiveQ function
   is used for image transformation during samples creation;
   otherwise internal cvWarpPerspective function is used */

//#include <ipl.h>

#include "cv.h"
#include "highgui.h"

/* Calculates coefficients of perspective transformation
 * which maps <quad> into rectangle ((0,0), (w,0), (w,h), (h,0)):
 *
 *      c00*xi + c01*yi + c02
 * ui = ---------------------
 *      c20*xi + c21*yi + c22
 *
 *      c10*xi + c11*yi + c12
 * vi = ---------------------
 *      c20*xi + c21*yi + c22
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
 * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
 * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
 * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
 * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
 * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
 * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
 *
 * where:
 *   (xi, yi) = (quad[i][0], quad[i][1])
 *        cij - coeffs[i][j], coeffs[2][2] = 1
 *   (ui, vi) - rectangle vertices
 */
static void cvGetPerspectiveTransform( CvSize src_size, double quad[4][2],
                                double coeffs[3][3] )
{
    //CV_FUNCNAME( "cvWarpPerspective" );

    __BEGIN__;

    double a[8][8];
    double b[8];

    CvMat A = cvMat( 8, 8, CV_64FC1, a );
    CvMat B = cvMat( 8, 1, CV_64FC1, b );
    CvMat X = cvMat( 8, 1, CV_64FC1, coeffs );

    int i;
    for( i = 0; i < 4; ++i )
    {
        a[i][0] = quad[i][0]; a[i][1] = quad[i][1]; a[i][2] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i][6] = a[i][7] = 0;
        b[i] = 0;
    }
    for( i = 4; i < 8; ++i )
    {
        a[i][3] = quad[i-4][0]; a[i][4] = quad[i-4][1]; a[i][5] = 1;
        a[i][0] = a[i][1] = a[i][2] = a[i][6] = a[i][7] = 0;
        b[i] = 0;
    }

    int u = src_size.width - 1;
    int v = src_size.height - 1;

    a[1][6] = -quad[1][0] * u; a[1][7] = -quad[1][1] * u;
    a[2][6] = -quad[2][0] * u; a[2][7] = -quad[2][1] * u;
    b[1] = b[2] = u;

    a[6][6] = -quad[2][0] * v; a[6][7] = -quad[2][1] * v;
    a[7][6] = -quad[3][0] * v; a[7][7] = -quad[3][1] * v;
    b[6] = b[7] = v;

    cvSolve( &A, &B, &X );

    coeffs[2][2] = 1;

    __END__;
}

/* Warps source into destination by a perspective transform */
static void cvWarpPerspective( CvArr* src, CvArr* dst, double quad[4][2] )
{
    CV_FUNCNAME( "cvWarpPerspective" );

    __BEGIN__;

#ifdef __IPL_H__
    IplImage src_stub, dst_stub;
    IplImage* src_img;
    IplImage* dst_img;
    CV_CALL( src_img = cvGetImage( src, &src_stub ) );
    CV_CALL( dst_img = cvGetImage( dst, &dst_stub ) );
    iplWarpPerspectiveQ( src_img, dst_img, quad, IPL_WARP_R_TO_Q,
                         IPL_INTER_CUBIC | IPL_SMOOTH_EDGE );
#else

    int fill_value = 0;

    double c[3][3]; /* transformation coefficients */
    double q[4][2]; /* rearranged quad */

    int left = 0;
    int right = 0;
    int next_right = 0;
    int next_left = 0;
    double y_min = 0;
    double y_max = 0;
    double k_left, b_left, k_right, b_right;

    uchar* src_data;
    int src_step;
    CvSize src_size;

    uchar* dst_data;
    int dst_step;
    CvSize dst_size;

    double d = 0;
    int direction = 0;
    int i;

    if( !src || (!CV_IS_IMAGE( src ) && !CV_IS_MAT( src )) ||
        cvGetElemType( src ) != CV_8UC1 ||
        cvGetDims( src ) != 2 )
    {
        CV_ERROR( CV_StsBadArg,
            "Source must be two-dimensional array of CV_8UC1 type." );
    }
    if( !dst || (!CV_IS_IMAGE( dst ) && !CV_IS_MAT( dst )) ||
        cvGetElemType( dst ) != CV_8UC1 ||
        cvGetDims( dst ) != 2 )
    {
        CV_ERROR( CV_StsBadArg,
            "Destination must be two-dimensional array of CV_8UC1 type." );
    }

    CV_CALL( cvGetRawData( src, &src_data, &src_step, &src_size ) );
    CV_CALL( cvGetRawData( dst, &dst_data, &dst_step, &dst_size ) );

    CV_CALL( cvGetPerspectiveTransform( src_size, quad, c ) );

    /* if direction > 0 then vertices in quad follow in a CW direction,
       otherwise they follow in a CCW direction */
    direction = 0;
    for( i = 0; i < 4; ++i )
    {
        int ni = i + 1; if( ni == 4 ) ni = 0;
        int pi = i - 1; if( pi == -1 ) pi = 3;

        d = (quad[i][0] - quad[pi][0])*(quad[ni][1] - quad[i][1]) -
            (quad[i][1] - quad[pi][1])*(quad[ni][0] - quad[i][0]);
        int cur_direction = CV_SIGN(d);
        if( direction == 0 )
        {
            direction = cur_direction;
        }
        else if( direction * cur_direction < 0 )
        {
            direction = 0;
            break;
        }
    }
    if( direction == 0 )
    {
        CV_ERROR( CV_StsBadArg, "Quadrangle is nonconvex or degenerated." );
    }

    /* <left> is the index of the topmost quad vertice
       if there are two such vertices <left> is the leftmost one */
    left = 0;
    for( i = 1; i < 4; ++i )
    {
        if( (quad[i][1] < quad[left][1]) ||
            ((quad[i][1] == quad[left][1]) && (quad[i][0] < quad[left][0])) )
        {
            left = i;
        }
    }
    /* rearrange <quad> vertices in such way that they follow in a CW
       direction and the first vertice is the topmost one and put them
       into <q> */
    if( direction > 0 )
    {
        for( i = left; i < 4; ++i )
        {
            q[i-left][0] = quad[i][0];
            q[i-left][1] = quad[i][1];
        }
        for( i = 0; i < left; ++i )
        {
            q[4-left+i][0] = quad[i][0];
            q[4-left+i][1] = quad[i][1];
        }
    }
    else
    {
        for( i = left; i >= 0; --i )
        {
            q[left-i][0] = quad[i][0];
            q[left-i][1] = quad[i][1];
        }
        for( i = 3; i > left; --i )
        {
            q[4+left-i][0] = quad[i][0];
            q[4+left-i][1] = quad[i][1];
        }
    }

    left = right = 0;
    /* if there are two topmost points, <right> is the index of the rightmost one
       otherwise <right> */
    if( q[left][1] == q[left+1][1] )
    {
        right = 1;
    }

    /* <next_left> follows <left> in a CCW direction */
    next_left = 3;
    /* <next_right> follows <right> in a CW direction */
    next_right = right + 1;

    /* subtraction of 1 prevents skipping of the first row */
    y_min = q[left][1] - 1;

    /* left edge equation: y = k_left * x + b_left */
    k_left = (q[left][0] - q[next_left][0]) /
               (q[left][1] - q[next_left][1]);
    b_left = (q[left][1] * q[next_left][0] -
               q[left][0] * q[next_left][1]) /
                 (q[left][1] - q[next_left][1]);

    /* right edge equation: y = k_right * x + b_right */
    k_right = (q[right][0] - q[next_right][0]) /
               (q[right][1] - q[next_right][1]);
    b_right = (q[right][1] * q[next_right][0] -
               q[right][0] * q[next_right][1]) /
                 (q[right][1] - q[next_right][1]);

    for(;;)
    {
        int x, y;

        y_max = MIN( q[next_left][1], q[next_right][1] );

        int iy_min = MAX( cvRound(y_min), 0 ) + 1;
        int iy_max = MIN( cvRound(y_max), dst_size.height - 1 );

        double x_min = k_left * iy_min + b_left;
        double x_max = k_right * iy_min + b_right;

        /* walk through the destination quadrangle row by row */
        for( y = iy_min; y <= iy_max; ++y )
        {
            int ix_min = MAX( cvRound( x_min ), 0 );
            int ix_max = MIN( cvRound( x_max ), dst_size.width - 1 );

            for( x = ix_min; x <= ix_max; ++x )
            {
                /* calculate coordinates of the corresponding source array point */
                double div = (c[2][0] * x + c[2][1] * y + c[2][2]);
                double src_x = (c[0][0] * x + c[0][1] * y + c[0][2]) / div;
                double src_y = (c[1][0] * x + c[1][1] * y + c[1][2]) / div;

                int isrc_x = cvFloor( src_x );
                int isrc_y = cvFloor( src_y );
                double delta_x = src_x - isrc_x;
                double delta_y = src_y - isrc_y;

                uchar* s = src_data + isrc_y * src_step + isrc_x;

                int i00, i10, i01, i11;
                i00 = i10 = i01 = i11 = (int) fill_value;

                /* linear interpolation using 2x2 neighborhood */
                if( isrc_x >= 0 && isrc_x <= src_size.width &&
                    isrc_y >= 0 && isrc_y <= src_size.height )
                {
                    i00 = s[0];
                }
                if( isrc_x >= -1 && isrc_x < src_size.width &&
                    isrc_y >= 0 && isrc_y <= src_size.height )
                {
                    i10 = s[1];
                }
                if( isrc_x >= 0 && isrc_x <= src_size.width &&
                    isrc_y >= -1 && isrc_y < src_size.height )
                {
                    i01 = s[src_step];
                }
                if( isrc_x >= -1 && isrc_x < src_size.width &&
                    isrc_y >= -1 && isrc_y < src_size.height )
                {
                    i11 = s[src_step+1];
                }

                double i0 = i00 + (i10 - i00)*delta_x;
                double i1 = i01 + (i11 - i01)*delta_x;

                ((uchar*)(dst_data + y * dst_step))[x] = (uchar) (i0 + (i1 - i0)*delta_y);
            }
            x_min += k_left;
            x_max += k_right;
        }

        if( (next_left == next_right) ||
            (next_left+1 == next_right && q[next_left][1] == q[next_right][1]) )
        {
            break;
        }

        if( y_max == q[next_left][1] )
        {
            left = next_left;
            next_left = left - 1;

            k_left = (q[left][0] - q[next_left][0]) /
                       (q[left][1] - q[next_left][1]);
            b_left = (q[left][1] * q[next_left][0] -
                       q[left][0] * q[next_left][1]) /
                         (q[left][1] - q[next_left][1]);
        }
        if( y_max == q[next_right][1] )
        {
            right = next_right;
            next_right = right + 1;

            k_right = (q[right][0] - q[next_right][0]) /
                       (q[right][1] - q[next_right][1]);
            b_right = (q[right][1] * q[next_right][0] -
                       q[right][0] * q[next_right][1]) /
                         (q[right][1] - q[next_right][1]);
        }
        y_min = y_max;
    }
#endif /* #ifndef __IPL_H__ */

    __END__;
}

static
void icvRandomQuad( int width, int height, double quad[4][2],
                    double maxxangle,
                    double maxyangle,
                    double maxzangle )
{
    double distfactor = 3.0;
    double distfactor2 = 1.0;

    double halfw, halfh;
    int i;

    double rotVectData[3];
    double vectData[3];
    double rotMatData[9];

    CvMat rotVect;
    CvMat rotMat;
    CvMat vect;

    double d;

    rotVect = cvMat( 3, 1, CV_64FC1, &rotVectData[0] );
    rotMat = cvMat( 3, 3, CV_64FC1, &rotMatData[0] );
    vect = cvMat( 3, 1, CV_64FC1, &vectData[0] );

    rotVectData[0] = maxxangle * (2.0 * rand() / RAND_MAX - 1.0);
    rotVectData[1] = ( maxyangle - fabs( rotVectData[0] ) )
        * (2.0 * rand() / RAND_MAX - 1.0);
    rotVectData[2] = maxzangle * (2.0 * rand() / RAND_MAX - 1.0);
    d = (distfactor + distfactor2 * (2.0 * rand() / RAND_MAX - 1.0)) * width;

/*
    rotVectData[0] = maxxangle;
    rotVectData[1] = maxyangle;
    rotVectData[2] = maxzangle;

    d = distfactor * width;
*/

    cvRodrigues2( &rotVect, &rotMat );

    halfw = 0.5 * width;
    halfh = 0.5 * height;

    quad[0][0] = -halfw;
    quad[0][1] = -halfh;
    quad[1][0] =  halfw;
    quad[1][1] = -halfh;
    quad[2][0] =  halfw;
    quad[2][1] =  halfh;
    quad[3][0] = -halfw;
    quad[3][1] =  halfh;

    for( i = 0; i < 4; i++ )
    {
        rotVectData[0] = quad[i][0];
        rotVectData[1] = quad[i][1];
        rotVectData[2] = 0.0;
        cvMatMulAdd( &rotMat, &rotVect, 0, &vect );
        quad[i][0] = vectData[0] * d / (d + vectData[2]) + halfw;
        quad[i][1] = vectData[1] * d / (d + vectData[2]) + halfh;

        /*
        quad[i][0] += halfw;
        quad[i][1] += halfh;
        */
    }
}

// DANIEL_BUG_FIX - added icvStartSampleDistortion_onImg()
int icvStartSampleDistortion_onImg( const IplImage* img        , int bgcolor, int bgthreshold, CvSampleDistortionData* data ){
    memset( data, 0, sizeof( *data ) );
    data->src = cvCloneImage(img);
    if( data->src != NULL && data->src->nChannels == 1
        && data->src->depth == IPL_DEPTH_8U )
    {
        int r, c;
        uchar* pmask;
        uchar* psrc;
        uchar* perode;
        uchar* pdilate;
        uchar dd, de;

        data->dx = data->src->width / 2;
        data->dy = data->src->height / 2;
        data->bgcolor = bgcolor;

        data->mask = cvCloneImage( data->src );
        data->erode = cvCloneImage( data->src );
        data->dilate = cvCloneImage( data->src );

        /* make mask image */
        for( r = 0; r < data->mask->height; r++ )
        {
            for( c = 0; c < data->mask->width; c++ )
            {
                pmask = ( (uchar*) (data->mask->imageData + r * data->mask->widthStep)
                        + c );
                if( bgcolor - bgthreshold <= (int) (*pmask) &&
                    (int) (*pmask) <= bgcolor + bgthreshold )
                {
                    *pmask = (uchar) 0;
                }
                else
                {
                    *pmask = (uchar) 255;
                }
            }
        }

        /* extend borders of source image */
        cvErode( data->src, data->erode, 0, 1 );
        cvDilate( data->src, data->dilate, 0, 1 );
        for( r = 0; r < data->mask->height; r++ )
        {
            for( c = 0; c < data->mask->width; c++ )
            {
                pmask = ( (uchar*) (data->mask->imageData + r * data->mask->widthStep)
                        + c );
                if( (*pmask) == 0 )
                {
                    psrc = ( (uchar*) (data->src->imageData + r * data->src->widthStep)
                           + c );
                    perode =
                        ( (uchar*) (data->erode->imageData + r * data->erode->widthStep)
                                + c );
                    pdilate =
                        ( (uchar*)(data->dilate->imageData + r * data->dilate->widthStep)
                                + c );
                    de = (uchar)(bgcolor - (*perode));
                    dd = (uchar)((*pdilate) - bgcolor);
                    if( de >= dd && de > bgthreshold )
                    {
                        (*psrc) = (*perode);
                    }
                    if( dd > de && dd > bgthreshold )
                    {
                        (*psrc) = (*pdilate);
                    }
                }
            }
        }

        data->img = cvCreateImage( cvSize( data->src->width + 2 * data->dx,
                                           data->src->height + 2 * data->dy ),
                                   IPL_DEPTH_8U, 1 );
        data->maskimg = cvCloneImage( data->img );

        return 1;
    }

    return 0;
}
// DANIEL_BUG_FIX - End
int icvStartSampleDistortion( const char* imgfilename, int bgcolor, int bgthreshold, CvSampleDistortionData* data ){
    memset( data, 0, sizeof( *data ) );
    data->src = cvLoadImage( imgfilename, 0 );
    if( data->src != NULL && data->src->nChannels == 1
        && data->src->depth == IPL_DEPTH_8U )
    {
        int r, c;
        uchar* pmask;
        uchar* psrc;
        uchar* perode;
        uchar* pdilate;
        uchar dd, de;

        data->dx = data->src->width / 2;
        data->dy = data->src->height / 2;
        data->bgcolor = bgcolor;

        data->mask = cvCloneImage( data->src );
        data->erode = cvCloneImage( data->src );
        data->dilate = cvCloneImage( data->src );

        /* make mask image */
        for( r = 0; r < data->mask->height; r++ )
        {
            for( c = 0; c < data->mask->width; c++ )
            {
                pmask = ( (uchar*) (data->mask->imageData + r * data->mask->widthStep)
                        + c );
                if( bgcolor - bgthreshold <= (int) (*pmask) &&
                    (int) (*pmask) <= bgcolor + bgthreshold )
                {
                    *pmask = (uchar) 0;
                }
                else
                {
                    *pmask = (uchar) 255;
                }
            }
        }

        /* extend borders of source image */
        cvErode( data->src, data->erode, 0, 1 );
        cvDilate( data->src, data->dilate, 0, 1 );
        for( r = 0; r < data->mask->height; r++ )
        {
            for( c = 0; c < data->mask->width; c++ )
            {
                pmask = ( (uchar*) (data->mask->imageData + r * data->mask->widthStep)
                        + c );
                if( (*pmask) == 0 )
                {
                    psrc = ( (uchar*) (data->src->imageData + r * data->src->widthStep)
                           + c );
                    perode =
                        ( (uchar*) (data->erode->imageData + r * data->erode->widthStep)
                                + c );
                    pdilate =
                        ( (uchar*)(data->dilate->imageData + r * data->dilate->widthStep)
                                + c );
                    de = (uchar)(bgcolor - (*perode));
                    dd = (uchar)((*pdilate) - bgcolor);
                    if( de >= dd && de > bgthreshold )
                    {
                        (*psrc) = (*perode);
                    }
                    if( dd > de && dd > bgthreshold )
                    {
                        (*psrc) = (*pdilate);
                    }
                }
            }
        }

        data->img = cvCreateImage( cvSize( data->src->width + 2 * data->dx,
                                           data->src->height + 2 * data->dy ),
                                   IPL_DEPTH_8U, 1 );
        data->maskimg = cvCloneImage( data->img );

        return 1;
    }

    return 0;
}

void icvPlaceDistortedSample( CvArr* background,
                              int inverse, int maxintensitydev,
                              double maxxangle, double maxyangle, double maxzangle,
                              int inscribe, double maxshiftf, double maxscalef,
                              CvSampleDistortionData* data )
{
    double quad[4][2];
    int r, c;
    uchar* pimg;
    uchar* pbg;
    uchar* palpha;
    uchar chartmp;
    int forecolordev;
    float scale;
    IplImage* img;
    IplImage* maskimg;
    CvMat  stub;
    CvMat* bgimg;

    CvRect cr;
    CvRect roi;

    double xshift, yshift, randscale;

    icvRandomQuad( data->src->width, data->src->height, quad,
                   maxxangle, maxyangle, maxzangle );
    quad[0][0] += (double) data->dx;
    quad[0][1] += (double) data->dy;
    quad[1][0] += (double) data->dx;
    quad[1][1] += (double) data->dy;
    quad[2][0] += (double) data->dx;
    quad[2][1] += (double) data->dy;
    quad[3][0] += (double) data->dx;
    quad[3][1] += (double) data->dy;

    cvSet( data->img, cvScalar( data->bgcolor ) );
    cvSet( data->maskimg, cvScalar( 0.0 ) );

    cvWarpPerspective( data->src, data->img, quad );
    cvWarpPerspective( data->mask, data->maskimg, quad );

    cvSmooth( data->maskimg, data->maskimg, CV_GAUSSIAN, 3, 3 );

    bgimg = cvGetMat( background, &stub );

    cr.x = data->dx;
    cr.y = data->dy;
    cr.width = data->src->width;
    cr.height = data->src->height;

    if( inscribe )
    {
        /* quad's circumscribing rectangle */
        cr.x = (int) MIN( quad[0][0], quad[3][0] );
        cr.y = (int) MIN( quad[0][1], quad[1][1] );
        cr.width  = (int) (MAX( quad[1][0], quad[2][0] ) + 0.5F ) - cr.x;
        cr.height = (int) (MAX( quad[2][1], quad[3][1] ) + 0.5F ) - cr.y;
    }

    xshift = maxshiftf * rand() / RAND_MAX;
    yshift = maxshiftf * rand() / RAND_MAX;

    cr.x -= (int) ( xshift * cr.width  );
    cr.y -= (int) ( yshift * cr.height );
    cr.width  = (int) ((1.0 + maxshiftf) * cr.width );
    cr.height = (int) ((1.0 + maxshiftf) * cr.height);

    randscale = maxscalef * rand() / RAND_MAX;
    cr.x -= (int) ( 0.5 * randscale * cr.width  );
    cr.y -= (int) ( 0.5 * randscale * cr.height );
    cr.width  = (int) ((1.0 + randscale) * cr.width );
    cr.height = (int) ((1.0 + randscale) * cr.height);

    scale = MAX( ((float) cr.width) / bgimg->cols, ((float) cr.height) / bgimg->rows );

    roi.x = (int) (-0.5F * (scale * bgimg->cols - cr.width) + cr.x);
    roi.y = (int) (-0.5F * (scale * bgimg->rows - cr.height) + cr.y);
    roi.width  = (int) (scale * bgimg->cols);
    roi.height = (int) (scale * bgimg->rows);

    img = cvCreateImage( cvSize( bgimg->cols, bgimg->rows ), IPL_DEPTH_8U, 1 );
    maskimg = cvCreateImage( cvSize( bgimg->cols, bgimg->rows ), IPL_DEPTH_8U, 1 );

    cvSetImageROI( data->img, roi );
    cvResize( data->img, img );
    cvResetImageROI( data->img );
    cvSetImageROI( data->maskimg, roi );
    cvResize( data->maskimg, maskimg );
    cvResetImageROI( data->maskimg );

    forecolordev = (int) (maxintensitydev * (2.0 * rand() / RAND_MAX - 1.0));

    for( r = 0; r < img->height; r++ )
    {
        for( c = 0; c < img->width; c++ )
        {
            pimg = (uchar*) img->imageData + r * img->widthStep + c;
            pbg = (uchar*) bgimg->data.ptr + r * bgimg->step + c;
            palpha = (uchar*) maskimg->imageData + r * maskimg->widthStep + c;
            chartmp = (uchar) MAX( 0, MIN( 255, forecolordev + (*pimg) ) );
            if( inverse )
            {
                chartmp ^= 0xFF;
            }
            *pbg = (uchar) (( chartmp*(*palpha )+(255 - (*palpha) )*(*pbg) ) / 255);
        }
    }

    cvReleaseImage( &img );
    cvReleaseImage( &maskimg );
}

void icvEndSampleDistortion( CvSampleDistortionData* data )
{
    if( data->src )
    {
        cvReleaseImage( &data->src );
    }
    if( data->mask )
    {
        cvReleaseImage( &data->mask );
    }
    if( data->erode )
    {
        cvReleaseImage( &data->erode );
    }
    if( data->dilate )
    {
        cvReleaseImage( &data->dilate );
    }
    if( data->img )
    {
        cvReleaseImage( &data->img );
    }
    if( data->maskimg )
    {
        cvReleaseImage( &data->maskimg );
    }
}

void icvWriteVecHeader( FILE* file, int count, int width, int height )
{
    int vecsize;
    short tmp;

    /* number of samples */
    fwrite( &count, sizeof( count ), 1, file );
    /* vector size */
    vecsize = width * height;
    fwrite( &vecsize, sizeof( vecsize ), 1, file );
    /* min/max values */
    tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, file );
    fwrite( &tmp, sizeof( tmp ), 1, file );
}

void icvWriteVecSample( FILE* file, CvArr* sample )
{
    CvMat* mat, stub;
    int r, c;
    short tmp;
    uchar chartmp;

    mat = cvGetMat( sample, &stub );
    chartmp = 0;
    fwrite( &chartmp, sizeof( chartmp ), 1, file );
    for( r = 0; r < mat->rows; r++ )
    {
        for( c = 0; c < mat->cols; c++ )
        {
            tmp = (short) (CV_MAT_ELEM( *mat, uchar, r, c ));
            fwrite( &tmp, sizeof( tmp ), 1, file );
        }
    }
}



/****************************************************************************/
// DANIEL_BUG_FIX - start
// Functions Which implement a fast rescaling proprietery (extremly fast) algorithm, developed by Daniel.
typedef unsigned char INTENSITY;
typedef struct {
	INTENSITY	*I;	
	int			iRows;
	int			iCols;
} IMAGE;
#define ABS(a)				((a) >= 0 ? (a) : (-(a)))

void IPLtoDaniel(    IMAGE *dIm, IplImage* iplIm) {
	dIm->iCols = (iplIm)->width; 
	dIm->iRows = (iplIm)->height; 
	dIm->I = (INTENSITY*)(iplIm)->imageData; 

	if (iplIm->widthStep != (iplIm)->width){
		// Fuck - openCV uses padding. Remove it..
		int i;
		INTENSITY* curPixSrc = dIm->I + iplIm->widthStep;
		INTENSITY* curPixDst = dIm->I + (iplIm)->width;
		for (i=0; i<(iplIm)->height-1; i++){
			memmove(curPixDst,curPixSrc,(iplIm)->width);
			curPixSrc += iplIm->widthStep;
			curPixDst += iplIm->width;
		}
		CV_Assert(curPixDst-dIm->I==dIm->iCols*dIm->iRows);
		CV_Assert(curPixSrc-dIm->I==iplIm->imageSize);
	}
}
void DanielBackToIPL(IMAGE *dIm, IplImage* iplIm) {
	if (iplIm->widthStep != (iplIm)->width){
		// Fuck - openCV uses padding. Add It
		int i;
		INTENSITY* curPixSrc = dIm->I + dIm->iCols*dIm->iRows - iplIm->width;
		INTENSITY* curPixDst = dIm->I + iplIm->imageSize      - iplIm->widthStep;
		for (i=0; i<(iplIm)->height-1; i++){
			memmove(curPixDst,curPixSrc,iplIm->width);
			curPixSrc -= iplIm->width;
			curPixDst -= iplIm->widthStep;
		}
		CV_Assert(curPixDst==dIm->I);
		CV_Assert(curPixSrc==dIm->I);
	}
}
void ImagePaint(IMAGE *pImg, INTENSITY itsVal){
	memset(pImg->I,itsVal,pImg->iCols*pImg->iRows);
}
void ImageCopy( IMAGE *pimgDst, const IMAGE *imgSrc){
	memcpy(pimgDst->I,imgSrc->I,imgSrc->iCols*imgSrc->iRows);
	pimgDst->iCols = imgSrc->iCols;
	pimgDst->iRows = imgSrc->iRows;
}

void ImageDownSample(const IMAGE *pImgSrc, IMAGE *pImgDst){

	const INTENSITY *sPix;					// Pointers to current pixel in source image
		  INTENSITY *dPix = pImgDst->I;		//			and destination image pixel

	int sRows = pImgSrc->iRows;		// Sizes of images
	int sCols = pImgSrc->iCols;
	int dRows = pImgDst->iRows;
	int dCols = pImgDst->iCols;

	int xD,yD,xS,yS,xF,yF;					// Indices of pixels for 'for' loops. 
											// (xD,yD) is current processed pixel on dst image. (xS,yS) on source.
											// (xF,yF) is current processed pixel in the smoothing filter
	// Optimized version, only integers

	int	scX,scY;							// Location of (xD,yD) Pixel in source image coordinates.

	int	srcMinX,srcMaxX;					// Area around (scX,scY) that should be averaged as interpolation
	int	srcMinY,srcMaxY;				  

	#define MAX_BLOC_SIZE (128)
	int	Dx[MAX_BLOC_SIZE],Dy[MAX_BLOC_SIZE];// Distance of (xS,yS) from (scX,scY). Needed for producing weights for averaging
	int	W;									// Weight of (xS,yS) in the overall sum

	int	curPixVal;							// Sum of W * src[xS,yS] over entire interpolation area
	int curPixSumWeights;					// Sum of weights, for normalization

	const int PRECISION      = 8;			// We use last 8 bits as fraction. Accuracy of ~99.6%
	const int HIGH_PRECISION = 10;			// Additional high precision
	const int one = (1<<PRECISION);			// Numer 1 in fixed point precision

	int stepRowI = (int)(sRows << PRECISION)/dRows;	// Each 1 pixel in dst image covers a block of pixels of 	
	int stepColI = (int)(sCols << PRECISION)/dCols;	//        size (stepRow,stepCol) in source image	

	int stepRowI_H = (int)((sRows<<HIGH_PRECISION)<< PRECISION)/dRows; // High accuracy. We need it when incrementing
	int stepColI_H = (int)((sCols<<HIGH_PRECISION)<< PRECISION)/dCols; // Many times the variable, so accuracy is lost.
	
	if ((stepRowI >> PRECISION)*(stepColI >> PRECISION) > MAX_BLOC_SIZE*MAX_BLOC_SIZE){
		// Error. Protect against overflow. Return black image
		ImagePaint(pImgDst,0);
		return;
	}

	// Do calculation for each pixel in dest image (start on selecting a row).
	for (yD = 0; yD<dRows; yD++){
		// Find the location of Pixel dst[xD,yD] in source image:
		scY     = ((stepRowI-one)/2 + ((yD*stepRowI_H)>>HIGH_PRECISION));	// Center of block + N blocks. (step-1)/2 since block is in pixels [0..step-1] 

		// Select area around (scX,scY) that would be averaged. It must be at least half block from each
		// Side so each pixel will average all it's components.
		srcMinY = ((scY-stepRowI/2)+one-1)	>> PRECISION;		// Ceil
		srcMaxY =  (scY+stepRowI/2)			>> PRECISION;		// Flor

		// Varify that the area is not out of src image.
		srcMinY = MAX(srcMinY,0);
		srcMaxY = MIN(srcMaxY,sRows-1);

		// Calculate relative distances of pixels in the area from (scX,scY). 
		// Use reversed distances as weights, As: (areaSize-dx)/areaSize
		for (yF=0,yS = srcMinY; yS <= srcMaxY; yF++,yS++)
			Dy[yF] = ((stepRowI - ABS((yS<<PRECISION)-scY))<<PRECISION)/stepRowI;

		for (xD = 0; xD<dCols; xD++){	
			// Same as above but for 'X' axis.
			scX		= ((stepColI-one)/2 + ((xD*stepColI_H)>>HIGH_PRECISION));		
			srcMinX	= ((scX-stepColI/2)+one-1) >> PRECISION;	// Ceil
			srcMaxX =  (scX+stepColI/2)		   >> PRECISION;	// Floor
			srcMinX = MAX(srcMinX,0);
			srcMaxX = MIN(srcMaxX,sCols-1);

			for (xF=0,xS = srcMinX; xS <= srcMaxX; xF++,xS++)
				Dx[xF] = ((stepColI - ABS((xS<<PRECISION)-scX))<<PRECISION)/stepColI;	

			// -----------------------------------------
			// Average the pixels in the area (central pixels get more weight)
			curPixVal		 = 0;
			curPixSumWeights = 0;
			for (yF=0,yS = srcMinY; yS <= srcMaxY; yF++,yS++){
				sPix = pImgSrc->I + yS*sCols+srcMinX;					// Initialize pointer to beggining of current row
				for (xF=0,xS = srcMinX; xS <= srcMaxX; xF++,xS++,sPix++){
					W  = Dx[xF]*Dy[yF] >> PRECISION;					// Due to multiplication we got double precision
					curPixVal		+= *(sPix)*W;
					curPixSumWeights+= W;				
				}
			}

			// Put the pixels
			*(dPix++) = (INTENSITY)(curPixVal/curPixSumWeights);
		}
	}
}
// Inverse of down sample, using bilinear interpolation [2x2] mask
void ImageUpSample(const IMAGE *pImgSrc, IMAGE *pImgDst){

	const INTENSITY *sPix;					// Pointers to current pixel in source image
		  INTENSITY *dPix = pImgDst->I;		//			and destination image pixel

	int sRows = pImgSrc->iRows;		// Sizes of images
	int sCols = pImgSrc->iCols;
	int dRows = pImgDst->iRows;
	int dCols = pImgDst->iCols;

	int xD,yD,xS,yS,xF,yF;					// Indices of pixels for 'for' loops. 
											// (xD,yD) is current processed pixel on dst image. (xS,yS) on source.
											// (xF,yF) is current processed pixel in the smoothing filter
	// Optimized version, only integers

	int	scX,scY;							// Location of (xD,yD) Pixel in source image coordinates.

	int	srcMinX,srcMaxX;					// Area around (scX,scY) that should be averaged as interpolation
	int	srcMinY,srcMaxY;				  

	int	Dx[3],Dy[3];						// Distance of (xS,yS) from (scX,scY). Needed for producing weights for averaging
	int	W;									// Weight of (xS,yS) in the overall sum

	int	curPixVal;							// Sum of W * src[xS,yS] over entire interpolation area
	int curPixSumWeights;					// Sum of weights, for normalization

	const int HIGH_PRECISION = 10;			// We use high precision, since we operate with fractions. Must be less than 16 to avoid overflow
	const int one  = (1<<HIGH_PRECISION);	// Numer 1 in fixed point precision
	const int half = (one>>1);

	// Each 1 pixel in dst image covers a fraction of source pixel. This fraction is (stepRow,stepCol)
	// Unforunately we cannot calculate this fraction when upscaling by large factor since in a small 
	// precision error of 'step' will yield an accomulative error in for loops. 
	// Large upscale factor reduces the effect of HIGH_PRECISION.
	int stepRowI = (int)(sRows << HIGH_PRECISION)/dRows;	
	int stepColI = (int)(sCols << HIGH_PRECISION)/dCols;	

	// Do calculation for each pixel in dest image (start on selecting a row).
	for (yD = 0; yD<dRows; yD++){
		// Find the sub pixel location of Pixel dst[xD,yD] in source image:
		scY     = yD*stepRowI + (stepRowI>>1) - half;		// (yd+0.5[dstPixel])*stepRowI - 0.5[srcPixel].

		// Select area around (scX,scY) that would be averaged. It must be at least half original pixel from each side
		srcMinY = (scY)			>> HIGH_PRECISION;	// Floor
		srcMaxY = (scY+one-1)	>> HIGH_PRECISION;	// Ceil		

		// Varify that the area is not out of src image.
		srcMinY = MAX(srcMinY,0);
		srcMaxY = MIN(srcMaxY,sRows-1);

		// Calculate relative distances of pixels in the area from (scX,scY). 
		// Use reversed distances as weights, As: (areaSize-dx)/areaSize
		for (yF=0,yS = srcMinY; yS <= srcMaxY; yF++,yS++){
			Dy[yF] = one - ABS((yS<<HIGH_PRECISION)-scY);
		}

		for (xD = 0; xD<dCols; xD++){	
			// Same as above but for 'X' axis.
			scX     = xD*stepColI + (stepColI>>1) - half;		
			srcMinX = (scX)			>> HIGH_PRECISION;	// Floor
			srcMaxX = (scX+one-1)	>> HIGH_PRECISION;	// Ceil	
			srcMinX = MAX(srcMinX,0);
			srcMaxX = MIN(srcMaxX,sCols-1);

			for (xF=0,xS = srcMinX; xS <= srcMaxX; xF++,xS++){
				Dx[xF] = one - ABS((xS<<HIGH_PRECISION)-scX);
			}

			// -----------------------------------------
			// Average the pixels in the area (central pixels get more weight)
			curPixVal		 = 0;
			curPixSumWeights = 0;
			for (yF=0,yS = srcMinY; yS <= srcMaxY; yF++,yS++){
				sPix = pImgSrc->I + yS*sCols+srcMinX;					// Initialize pointer to beggining of current row
				for (xF=0,xS = srcMinX; xS <= srcMaxX; xF++,xS++,sPix++){
					W  = Dx[xF]*Dy[yF] >> HIGH_PRECISION;				// Due to multiplication we got double precision
					curPixVal		+= *(sPix)*W;
					curPixSumWeights+= W;				
				}
			}

			// Put the pixels
			*(dPix++) = (INTENSITY)(curPixVal/curPixSumWeights);
		}
	}
}
int ImageResize_Daniel(const IplImage* src, IplImage* dst){
	IplImage* tmp = cvCreateImage(cvGetSize(src),src->depth,src->nChannels);
	cvCopy(src, tmp, NULL);

	IMAGE dSrc, dDst;
	IPLtoDaniel(&dSrc,tmp);
	IPLtoDaniel(&dDst,dst);
	int colDiff = (dDst.iCols-dSrc.iCols);
	int rowDiff = (dDst.iRows-dSrc.iRows);
	// If images are of the same size than just copy
	if (!(colDiff|rowDiff)){
		ImageCopy(&dDst,&dSrc);
	}
	else if (colDiff*rowDiff < 0){
		// Not supported yet. Upscale in one direction, down scale in another.
 		cvReleaseImage(&tmp);
		return -1;
	}
	else if ((colDiff > 0)||(rowDiff > 0))
		ImageUpSample(&dSrc,&dDst);
	else
		ImageDownSample(&dSrc,&dDst);
 	DanielBackToIPL(&dDst,dst);
 	cvReleaseImage(&tmp);
	return 0;
}
// DANIEL_BUG_FIX - end
/****************************************************************************/


int cvCreateTrainingSamplesFromInfo( const char* infoname, const char* vecfilename,
                                     int num,
                                     int showsamples,
                                     int winwidth, int winheight, char interpType)
{
    char fullname[PATH_MAX];
    char* filename;

    FILE* info;
    FILE* vec;
    IplImage* src=0;
    IplImage* sample;
    int line;
    int error;
    int i;
    int x, y, width, height;
    int total;

    assert( infoname != NULL );
    assert( vecfilename != NULL );

    total = 0;
    if( !icvMkDir( vecfilename ) )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to create directory hierarchy: %s\n", vecfilename );
#endif /* CV_VERBOSE */

        return total;
    }

    info = fopen( infoname, "r" );
    if( info == NULL )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to open file: %s\n", infoname );
#endif /* CV_VERBOSE */

        return total;
    }

    vec = fopen( vecfilename, "wb" );
    if( vec == NULL )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to open file: %s\n", vecfilename );
#endif /* CV_VERBOSE */

        fclose( info );

        return total;
    }

    sample = cvCreateImage( cvSize( winwidth, winheight ), IPL_DEPTH_8U, 1 );

    icvWriteVecHeader( vec, num, sample->width, sample->height );

    if( showsamples )
    {
        cvNamedWindow( "Sample", CV_WINDOW_AUTOSIZE );
    }

    strcpy( fullname, infoname );
    filename = strrchr( fullname, '\\' );
    if( filename == NULL )
    {
        filename = strrchr( fullname, '/' );
    }
    if( filename == NULL )
    {
        filename = fullname;
    }
    else
    {
        filename++;
    }

    for( line = 1, error = 0, total = 0; total < num ;line++ )
    {
        int count;

        error = ( fscanf( info, "%s %d", filename, &count ) != 2 );
        if( !error )
        {
            src = cvLoadImage( fullname, 0 );
            error = ( src == NULL );
            if( error )
            {

#if CV_VERBOSE
                fprintf( stderr, "Unable to open image: %s\n", fullname );
#endif /* CV_VERBOSE */

            }
        }
        for( i = 0; (i < count) && (total < num); i++, total++ )
        {
            error = ( fscanf( info, "%d %d %d %d", &x, &y, &width, &height ) != 4 );
            if( error ) break;
            cvSetImageROI( src, cvRect( x, y, width, height ) );
			// DANIEL_BUG_FIX - added Another parameter to select Daniel's Interpolation method. Default is 'c' - 'opencv', 'd' - 'daniel
			if (interpType=='d'){
				if (ImageResize_Daniel(src,sample) < 0){
					// Could not resize
					cvResize( src, sample, width >= sample->width && height >= sample->height ? CV_INTER_AREA : CV_INTER_LINEAR );
				}
			}
			else
				cvResize( src, sample, width >= sample->width && height >= sample->height ? CV_INTER_AREA : CV_INTER_LINEAR );

            if( showsamples )
            {
                cvShowImage( "Sample", sample );
                if( cvWaitKey( 0 ) == 27 )
                {
                    showsamples = 0;
                }
            }
            icvWriteVecSample( vec, sample );
        }

        if( src )
        {
            cvReleaseImage( &src );
        }

        if( error )
        {

#if CV_VERBOSE
			// DANIEL_BUG_FIX - Test whether we reached EOF. If so than we do not have parsing error. Probably an EOL symbol or \n or space
			if (!feof(info))
               fprintf( stderr, "%s(%d) : parse error", infoname, line );
			//else
			//   fprintf( stderr, "%s(%d) : strange symbol before EOF, no samples skipped", infoname, line );
			// DANIEL_BUG_FIX - End
#endif /* CV_VERBOSE */

            break;
        }
    }

    if( sample )
    {
        cvReleaseImage( &sample );
    }

    fclose( vec );
    fclose( info );

    return total;
}


void cvShowVecSamples( const char* filename, int winwidth, int winheight,
                       double scale )
{
    CvVecFile file;
    short tmp;
    int i;
    CvMat* sample;

    tmp = 0;
    file.input = fopen( filename, "rb" );

    if( file.input != NULL )
    {
        size_t elements_read1 = fread( &file.count, sizeof( file.count ), 1, file.input );
        size_t elements_read2 = fread( &file.vecsize, sizeof( file.vecsize ), 1, file.input );
        size_t elements_read3 = fread( &tmp, sizeof( tmp ), 1, file.input );
        size_t elements_read4 = fread( &tmp, sizeof( tmp ), 1, file.input );
        CV_Assert(elements_read1 == 1 && elements_read2 == 1 && elements_read3 == 1 && elements_read4 == 1);

        if( file.vecsize != winwidth * winheight )
        {
            int guessed_w = 0;
            int guessed_h = 0;

            fprintf( stderr, "Warning: specified sample width=%d and height=%d "
                "does not correspond to .vec file vector size=%d.\n",
                winwidth, winheight, file.vecsize );
            if( file.vecsize > 0 )
            {
                guessed_w = cvFloor( sqrt( (float) file.vecsize ) );
                if( guessed_w > 0 )
                {
                    guessed_h = file.vecsize / guessed_w;
                }
            }

            if( guessed_w <= 0 || guessed_h <= 0 || guessed_w * guessed_h != file.vecsize)
            {
                fprintf( stderr, "Error: failed to guess sample width and height\n" );
                fclose( file.input );

                return;
            }
            else
            {
                winwidth = guessed_w;
                winheight = guessed_h;
                fprintf( stderr, "Guessed width=%d, guessed height=%d\n",
                    winwidth, winheight );
            }
        }

        if( !feof( file.input ) && scale > 0 )
        {
            CvMat* scaled_sample = 0;

            file.last = 0;
            file.vector = (short*) cvAlloc( sizeof( *file.vector ) * file.vecsize );
            sample = scaled_sample = cvCreateMat( winheight, winwidth, CV_8UC1 );
            if( scale != 1.0 )
            {
                scaled_sample = cvCreateMat( MAX( 1, cvCeil( scale * winheight ) ),
                                             MAX( 1, cvCeil( scale * winwidth ) ),
                                             CV_8UC1 );
            }
            cvNamedWindow( "Sample", CV_WINDOW_AUTOSIZE );
            for( i = 0; i < file.count; i++ )
            {
                icvGetHaarTraininDataFromVecCallback( sample, &file );
                if( scale != 1.0 ) cvResize( sample, scaled_sample, CV_INTER_LINEAR);
                cvShowImage( "Sample", scaled_sample );
                if( cvWaitKey( 0 ) == 27 ) break;
            }
            if( scaled_sample && scaled_sample != sample ) cvReleaseMat( &scaled_sample );
            cvReleaseMat( &sample );
            cvFree( &file.vector );
        }
        fclose( file.input );
    }
}


/* End of file. */
