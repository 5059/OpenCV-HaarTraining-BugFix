# OpenCV Haar Training BugFix

openCV source code has a few crucial bugs in the Haar Training source code. This repository includes the fixed version of the files, by me.
This repository cannot be compiled as standalone. Rather overwrite original openCV source files (as described below) with this version and recompile openCV

### List of changes:
- I use openCV 2.4.3
- I didn't add any new algorithms.
- I just changed/fixed few issues in openCV files regarding Haar training.

### There are 2 main goals for those changes:
1. Bug fixes - For example: [InfiniteLoop]
2. Adding functionality for Haar training of applying distortion to list of positive samples. 
- In current version 'createsamples.cpp' take only one sample (single face) and applies geometrical distortions thus making classifer robust to rotation and displacement.
- In another option it is possible to take a list of faces and convert them to vec files. But there is no option combining the two above. I mean. Take a list of faces, apply distortion to each face and save the results to vec file
- Before my fix, people were using workarounds like saving each face to a 'vec' file and then uniting list of 'vec' files. Read about those workaround here: http://note.sonots.com/SciSoftware/haartraining.html

______________________
Q: How to find my changes if you don't have file compare software?

A: All Daniel's changes are sorrounded by remark    DANIEL_BUG_FIX. Just Search for it and you will find the fix 
- 1 line below or 
- comment explaining the chage or
- a whole new function added or
- code block denoted by DANIEL_BUG_FIX - Begin and DANIEL_BUG_FIX- end
------------------
### Changes: _cvhaartraining.h
1. Added a header to new function for distorting already loaded image and not image file path. 
2. openCV had a function: icvStartSampleDistortion()
3. I added: int icvStartSampleDistortion_onImg( const IplImage* img        , int bgcolor, int bgthreshold, CvSampleDistortionData* data );
4. I think that icvStartSampleDistortion() should be also simplified to load a file from disk and activate icvStartSampleDistortion_onImg(), thus avoiding code duplication in the 2 functions

### Changes: cvsamples.cpp
1. Implementation of the above described icvStartSampleDistortion_onImg() function.
2. Added few assist functions to implement daniel's fast pixels interpolation method. Instead of cvResize() those mothods can be used. This change is optional.

### Changes: cvhaartraining.h and cvhaartraining.cpp
1. Added new function for creating distorting samples described by infoname (list of images) rather than a single image
This function performs as following:
* Calculate the amount of samples in all the images. Let us denote it as 'origSamples'
* nTotalSamples = origSamples * nReplications   (Replication is done by applying few geometrical transformations in a loop)
* For each image in infoname
* For each sample in current image:
* For each different distortion: Write the result sample to vec file
* function returns the number of total written samples + many printf's for documentation.

int cvCreateTrainingSamplesFromInfo_add_distortion( const char* vecfilename,const char* infoname, int bgcolor, int bgthreshold, const char* bgfilename, int nReplication, int invert, int maxintensitydev, double maxxangle, double maxyangle, double maxzangle, int showsamples, int winwidth, int winheight);

2. Bug fix: In cvCreateTestSamples()  - count is set to MIN between count and number of background images. But sometimes we want to reuse backgrounds if we don't have enough of them.
   So the following line was removed: count = MIN( count, cvbgdata->count )

3. Few other bag fixes, well documented, marked with the same 'DANIEL_BUG_FIX'

4. Added 'interpType' parameter to few functions. This allows to select the type of interpolation for image rescaling. 

### Changes: createsamples.cpp

1. Few changes in the code allowing calling convention to 4 different original samples function + Daniel's new one.
2. Control on interpolation type of samples pixels.


### IMPORTANT!!!! Bug in haartraining.cpp that I havent fixed but is very crucial
cvCreateTreeCascadeClassifier() gets a memory size parameter for precalculating features. This number says how many features and images are used
But in haartraining.cpp it is describet to user as  [-mem <memory_in_MB = %d>]
This is a lie. People think that they give 300MB which is allot but actually openCV uses only 300 features*images which uses only a tiny fraction 
of 300MB and causes a very slow execution (training takes FEW DAYS instead of FEW HOURS !!!!).

   [InfiniteLoop]: <http://stackoverflow.com/questions/14041943/what-is-the-solution-for-opencv-haar-training-infinite-loop-when-overfitting>
