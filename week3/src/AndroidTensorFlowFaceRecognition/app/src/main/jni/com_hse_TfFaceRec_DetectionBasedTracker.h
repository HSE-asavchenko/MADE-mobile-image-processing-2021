/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_hse_TfFaceRec_DetectionBasedTracker */

#ifndef _Included_com_hse_TfFaceRec_DetectionBasedTracker
#define _Included_com_hse_TfFaceRec_DetectionBasedTracker
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_hse_TfFaceRec_DetectionBasedTracker
 * Method:    nativeCreateObject
 * Signature: (Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_com_hse_TfFaceRec_DetectionBasedTracker_nativeCreateObject
  (JNIEnv *, jclass, jstring, jint);

/*
 * Class:     com_hse_TfFaceRec_DetectionBasedTracker
 * Method:    nativeDestroyObject
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_hse_TfFaceRec_DetectionBasedTracker_nativeDestroyObject
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_hse_TfFaceRec_DetectionBasedTracker
 * Method:    nativeStart
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_hse_TfFaceRec_DetectionBasedTracker_nativeStart
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_hse_TfFaceRec_DetectionBasedTracker
 * Method:    nativeStop
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_hse_TfFaceRec_DetectionBasedTracker_nativeStop
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_hse_TfFaceRec_DetectionBasedTracker
 * Method:    nativeSetFaceSize
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_hse_TfFaceRec_DetectionBasedTracker_nativeSetFaceSize
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     com_hse_TfFaceRec_DetectionBasedTracker
 * Method:    nativeDetect
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_hse_TfFaceRec_DetectionBasedTracker_nativeDetect
  (JNIEnv *, jclass, jlong, jlong, jlong);

#ifdef __cplusplus
}
#endif
#endif
