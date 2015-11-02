#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "cv.h"
#include "cascadeclassifier.h"

using namespace std;

int main( int argc, char* argv[] )
{
    CvCascadeClassifier classifier;
    String cascadeDirName, vecName, bgName;
    int numPos    = 2000;
    int numNeg    = 1000;
    int numStages = 20;
    int precalcValBufSize = 256,
        precalcIdxBufSize = 256;
    bool baseFormatSave = false;

    CvCascadeParams cascadeParams;
    CvCascadeBoostParams stageParams;
    Ptr<CvFeatureParams> featureParams[] = { Ptr<CvFeatureParams>(new CvHaarFeatureParams),
                                             Ptr<CvFeatureParams>(new CvLBPFeatureParams),
                                             Ptr<CvFeatureParams>(new CvHOGFeatureParams)
                                           };
    int fc = sizeof(featureParams)/sizeof(featureParams[0]);
    if( argc == 1 )
    {
        cout << "Usage: " << argv[0] << endl;
        cout << "  -data <cascade_dir_name>" << endl;
        cout << "  -vec <vec_file_name>" << endl;
        cout << "  -bg <background_file_name>" << endl;
        cout << "  [-numPos <number_of_positive_samples = " << numPos << ">]" << endl;
        cout << "  [-numNeg <number_of_negative_samples = " << numNeg << ">]" << endl;
        cout << "  [-numStages <number_of_stages = " << numStages << ">]" << endl;
        cout << "  [-precalcValBufSize <precalculated_vals_buffer_size_in_Mb = " << precalcValBufSize << ">]" << endl;
        cout << "  [-precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb = " << precalcIdxBufSize << ">]" << endl;
        cout << "  [-baseFormatSave]" << endl;
        cascadeParams.printDefaults();
        stageParams.printDefaults();
        for( int fi = 0; fi < fc; fi++ )
            featureParams[fi]->printDefaults();
        return 0;
    }

    for( int i = 1; i < argc; i++ )
    {
        bool set = false;
        if( !strcmp( argv[i], "-data" ) )
        {
            cascadeDirName = argv[++i];
        }
        else if( !strcmp( argv[i], "-vec" ) )
        {
            vecName = argv[++i];
        }
        else if( !strcmp( argv[i], "-bg" ) )
        {
            bgName = argv[++i];
        }
        else if( !strcmp( argv[i], "-numPos" ) )
        {
            numPos = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-numNeg" ) )
        {
            numNeg = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-numStages" ) )
        {
            numStages = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-precalcValBufSize" ) )
        {
            precalcValBufSize = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-precalcIdxBufSize" ) )
        {
            precalcIdxBufSize = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-baseFormatSave" ) )
        {
            baseFormatSave = true;
        }
        else if ( cascadeParams.scanAttr( argv[i], argv[i+1] ) ) { i++; }
        else if ( stageParams.scanAttr( argv[i], argv[i+1] ) ) { i++; }
        else if ( !set )
        {
            for( int fi = 0; fi < fc; fi++ )
            {
                set = featureParams[fi]->scanAttr(argv[i], argv[i+1]);
                if ( !set )
                {
                    i++;
                    break;
                }
            }
        }
    }

	#if 1
    classifier.train( cascadeDirName,
                      vecName,
                      bgName,
                      numPos, numNeg,
                      precalcValBufSize, precalcIdxBufSize,
                      numStages,
                      cascadeParams,
                      *featureParams[cascadeParams.featureType],
                      stageParams,
                      baseFormatSave );
	#endif
    
	// DANIEL_BUG_FIX: After training, Run cascade and display results
	#if 0
	{
		CascadeClassifier c;
		string cascadeXML = cascadeDirName + "/cascade.xml";
		if (!c.load(cascadeXML))
			return -1;
		
		//IplImage* im = cvLoadImage("P:/Temp/v.bmp",CV_LOAD_IMAGE_UNCHANGED);
		//IplImage* im = cvLoadImage("P:/Temp/v_multiDetect.bmp",CV_LOAD_IMAGE_UNCHANGED);
		IplImage* im = cvLoadImage("P:\\Resources\\testing\\cc01_preproc.jpg",1);
		vector<Rect> objects;
		//c.detectMultiScale(im,objects,1.1,1,0,Size(20,30),Size(30,45));
		//c.detectMultiScale(im,objects,1.1,0,0,Size(16,24),Size(16,24)); //     Use zero neighbours when test a stand alone sample image from a Vec file.
		c.detectMultiScale(im,objects,1.1,0,0,Size(16,24),Size(16,24)); //     CV_WRAP virtual void detectMultiScale( const Mat& image,CV_OUT vector<Rect>& objects, vector<int>& rejectLevels, vector<double>& levelWeights, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size(), bool outputRejectLevels=false );
		int nRes = (int)objects.size();
		nRes = MIN(nRes,100);
		Rect curRect;
		int i;
		for (i=0; i<nRes; i++){
			curRect = objects.at(i);
			cvRectangle(im, Point2i(curRect.x,curRect.y),  Point2i(curRect.x+curRect.width,curRect.y+curRect.height),  Scalar(255,255,0));
			cvCircle( im, Point2i((int)(curRect.x + curRect.width/2), (int)(curRect.y +curRect.height/2)), 3, Scalar(255,255,0), 1 );
		}
		cvShowImage("input", im);
		waitKey(-1);
		cvReleaseImage(&im);
		return 0;
	}
	#endif
	return 0;
}
