LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#include $(PREBUILT_STATIC_LIBRARY)

#opencv
OPENCVROOT:= D:/src_code/opencv/opencv/build_android/OpenCV-android-sdk
OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=STATIC
include ${OPENCVROOT}/sdk/native/jni/OpenCV.mk

LOCAL_SRC_FILES := com_asav_processimage_GAPITester.cpp
LOCAL_CFLAGS += -mfloat-abi=softfp -mfpu=neon -std=c++11 #-march=armv64
LOCAL_ARM_NEON  := true
LOCAL_LDLIBS += -llog -L${OPENCVROOT}/sdk/native/3rdparty/libs/$(TARGET_ARCH_ABI)
LOCAL_MODULE := ImageProcessLib

include $(BUILD_SHARED_LIBRARY)