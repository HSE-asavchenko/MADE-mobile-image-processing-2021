#include <com_asav_processimage_GAPITester.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include<opencv2/features2d/features2d.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>

#include <string>
#include <vector>
#include <chrono>

#include <android/log.h>

#define LOG_TAG "EdgeDetector"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

class GApiTester{
public:
    GApiTester():kernel(cv::gapi::core::fluid::kernels()){
        cv::GMat in;
        cv::GMat vga = cv::gapi::resize(in, cv::Size(), 0.5, 0.5);
        cv::GMat gray = cv::gapi::RGB2Gray(vga);
        cv::GMat blurred = cv::gapi::blur(gray, cv::Size(5, 5));
        cv::GMat edges = cv::gapi::Canny(blurred, 32, 128, 3);
        cv::GMat b, g, r;
        std::tie(r, g, b) = cv::gapi::split3(vga);
        cv::GMat out = cv::gapi::merge3(r, g | edges, b);
        pac=new cv::GComputation(in, out);
    }
    ~GApiTester(){
        if(pac!=0)
            delete pac;
    }
    void processImage(cv::Mat& mIn,cv::Mat& mOut){
        if(pac!=0) {
            //int c=mIn.channels();
            pac->apply(mIn, mOut, cv::compile_args(kernel));
        }
    }
private:
    cv::GComputation* pac;
    cv::gapi::GKernelPackage kernel;
};
JNIEXPORT jlong JNICALL Java_com_asav_processimage_GAPITester_gapiInit
        (JNIEnv *, jclass){
    LOGD("Java_com_asav_processimage_GAPITester_gapiInit");
    jlong result = (jlong)new GApiTester();
    LOGD("Java_com_asav_processimage_GAPITester_gapiInit end");
    return result;
}

JNIEXPORT void JNICALL Java_com_asav_processimage_GAPITester_gapiTest
  (JNIEnv *, jclass, jlong thiz, jlong addrMatIn, jlong addrMatOut){
    LOGD("Java_com_asav_processimage_GAPITester_gapiTest");
    auto t1 = std::chrono::high_resolution_clock::now();
    if(thiz != 0)
    {
        cv::Mat& mIn = *(cv::Mat*)addrMatIn;
        cv::Mat& mOut = *(cv::Mat*)addrMatOut;
        ((GApiTester*)thiz)->processImage(mIn,mOut);
    }
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1);
    LOGD("Java_com_asav_processimage_GAPITester_gapiTest end total time: %d ms",int_ms);
}



