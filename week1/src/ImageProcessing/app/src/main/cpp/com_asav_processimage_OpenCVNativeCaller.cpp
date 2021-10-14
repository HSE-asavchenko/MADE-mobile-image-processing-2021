//https://pullrequest.opencv.org/buildbot/export/opencv_releases/master-contrib_pack-contrib-android/20200821-041002--11257/
#include <com_asav_processimage_OpenCVNativeCaller.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include<opencv2/features2d/features2d.hpp>

#include <string>
#include <vector>

#include <android/log.h>

#define LOG_TAG "ImageProcessing"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

JNIEXPORT void JNICALL Java_com_asav_processimage_OpenCVNativeCaller_niBlackThreshold
        (JNIEnv *, jclass, jlong addrMatIn, jlong addrMatOut){
    LOGD("Java_com_asav_processimage_MainActivity_niBlackThreshold -- BEGIN");
    cv::Mat& mIn = *(cv::Mat*)addrMatIn;
    cv::Mat grayImg;
    cv::cvtColor(mIn,grayImg,cv::COLOR_RGBA2GRAY);
    cv::Mat& mOut = *(cv::Mat*)addrMatOut;
    cv::ximgproc::niBlackThreshold(grayImg,mOut,255, cv::THRESH_BINARY, 25,cv::ximgproc::BINARIZATION_NIBLACK  );
    LOGD("Java_com_asav_processimage_MainActivity_niBlackThreshold -- END");
}

