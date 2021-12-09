/*
 *    Copyright (C) 2017 MINDORKS NEXTGEN PRIVATE LIMITED
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.hse.TfFaceRec;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.os.*;
import android.widget.Toast;

import androidx.appcompat.app.*;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.util.*;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static final String TAG = "Tensorflow example";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    /*
    private static final int INPUT_SIZE = 227;//224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "prob";//"output";
    private static final String MODEL_FILE = "file:///android_asset/age_net.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/age_comp_graph_label_strings.txt";
*/
    private Classifier classifier=null;

    private static final int INPUT_SIZE = 224;
    private static final String INPUT_NAME = "input_1";
    private static final String[] OUTPUT_NAMES = {"global_pooling/Mean","age_pred/Softmax","gender_pred/Sigmoid"};
    private static final String MODEL_FILE = "file:///android_asset/age_gender_tf2_new-01-0.14-0.92.pb";
    //private static final String MODEL_FILE = "file:///android_asset/quantized_graph.pb";
    private TensorFlowImageFeatureExtractor featureExtractor=null;
    private float[] prevFeatures=null;

    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private ImageView imageViewResult;

    private CameraBridgeViewBase   mOpenCvCameraView;
    private Mat mRgba,mGray;
    private DetectionBasedTracker  mNativeDetector;
    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;
    private static final Size FACE_SIZE=new Size(INPUT_SIZE,INPUT_SIZE);


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    System.loadLibrary("OpenCvDetectionLib");
                    Log.i(TAG, "After loading all libraries" );


                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File tmpDir = getDir("cascade", Context.MODE_PRIVATE);
                        File cascadeFile = new File(tmpDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(cascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mNativeDetector = new DetectionBasedTracker(cascadeFile.getAbsolutePath(), 0);

                        cascadeFile.delete();
                        tmpDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load models. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.setCameraPermissionGranted();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    private void setupOpencv(){
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        //OpenCVLoader.initDebug();
    }

    private void requestPermission(String permission) {
        if (ContextCompat.checkSelfPermission(MainActivity.this, permission)
                != PackageManager.PERMISSION_GRANTED) {

            // Give first an explanation, if needed.
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, permission)) {

                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.

            } else {

                // No explanation needed, we can request the permission.

                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{permission},
                        1);
            }
        }
    }
    private void requestPermissions(){
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP_MR1) {
            requestPermission(Manifest.permission.CAMERA);
            requestPermission(Manifest.permission.READ_EXTERNAL_STORAGE);
            requestPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cameraView);
        imageViewResult = null;//(ImageView) findViewById(R.id.imageViewResult);
        textViewResult = (TextView) findViewById(R.id.textViewResult);
        textViewResult.setMovementMethod(new ScrollingMovementMethod());

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else
            init();
    }
    private void init(){
        mOpenCvCameraView.setCameraIndex(0);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        initTensorFlowAndLoadModel();
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status=ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    protected void onResume() {
        super.onResume();
        //cameraView.start();
        setupOpencv();

    }

    @Override
    protected void onPause() {
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        //cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                if(classifier!=null)
                    classifier.close();
                if(featureExtractor!=null)
                    featureExtractor.close();
            }
        });
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (allGranted) {
                    // All Permissions Granted
                    init();
                } else {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    private Mat rotateMat(Mat mat){
        if(true)
            return mat;
        Mat res=mat.t();
        Core.flip(res, res, -1);
        Imgproc.resize(res, res, mat.size());
        return res;
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        //mGray = inputFrame.gray();
        mRgba=rotateMat(mRgba);
        Imgproc.cvtColor(mRgba, mGray, Imgproc.COLOR_BGR2GRAY);

        Mat bgr=new Mat();
        Imgproc.cvtColor(mRgba, bgr, Imgproc.COLOR_RGB2BGR);

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mNativeDetector != null)
            mNativeDetector.detect(mGray, faces);

        Rect[] facesArray = faces.toArray();
        Log.d(TAG, "mNativeDetector="+mNativeDetector+" facesArray.length="+facesArray.length);
        StringBuilder str=new StringBuilder();
        if(facesArray.length==0) {
            str.append("No faces found");
            //facesArray=new Rect[]{new Rect(10,10,mGray.cols()-20,mGray.rows()-20)};
        }
        for (int i = 0; i < facesArray.length; i++) {
            int dw=facesArray[i].width/8;
            int dh=facesArray[i].height/8;
            int x=facesArray[i].x-dw;
            if(x<0)
                x=0;
            int y=facesArray[i].y-dh;
            if(y<0)
                y=0;
            int w=facesArray[i].width+2*dw;
            if(x+w>=mRgba.cols())
                w=mRgba.cols()-x-1;
            int h=facesArray[i].height+2*dh;
            if(y+h>=mRgba.rows())
                h=mRgba.rows()-y-1;
            facesArray[i]=new Rect(x,y,w,h);


            Mat face=mRgba.submat(facesArray[i]);
            Imgproc.resize(face, face,FACE_SIZE);
            final Bitmap resultBitmap = Bitmap.createBitmap(face.cols(),  face.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(face, resultBitmap);
/*
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    imageViewResult.setImageBitmap(resultBitmap);
                }
            });
*/
            /*final List<Classifier.Recognition> results = classifier.recognizeImage(resultBitmap);
            str.append(results.toString()).append("\n");*/

            float[][] cnn_outputs=featureExtractor.recognizeImage(resultBitmap);
            float[] features=cnn_outputs[0];
            if(prevFeatures!=null){
                float dist=0;
                for(int fi=0;fi<features.length;++fi)
                    dist+=(features[fi]-prevFeatures[fi])*(features[fi]-prevFeatures[fi]);
                dist/=features.length;
                str.append("distance=").append(dist).append("\n");
            }
            else
                prevFeatures=new float[features.length];
            System.arraycopy( features, 0, prevFeatures, 0, features.length );

            //age
            final float[] age_features=cnn_outputs[1];
            ArrayList<Integer> indices = new ArrayList<>();
            for (int j=0;j<age_features.length;++j){
                indices.add(j);
            }
            Collections.sort(indices, new Comparator<Integer>() {
                @Override
                public int compare(Integer idx1, Integer idx2) {
                    if (age_features[idx1]==age_features[idx2])
                        return 0;
                    else if (age_features[idx1]>age_features[idx2])
                        return -1;
                    else
                        return 1;
                }
            });
            int max_index=2;
            double[] probabs=new double[max_index];
            double sum=0;
            for(int j=0;j<max_index;++j){
                probabs[j]=age_features[indices.get(j)];
                sum+=probabs[j];
            }
            double age=0;
            for(int j=0;j<max_index;++j) {
                age+=(indices.get(j)+0.5)*probabs[j]/sum;
            }
            str.append("age=").append((int) Math.round(age));

            //gender
            float gender=cnn_outputs[2][0];
            str.append(gender>=0.6?" male":" female").append("\n");
            Imgproc.rectangle(bgr, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

            Log.i(TAG,"age="+age+" gender="+gender);

        }
        final String result=str.toString();
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                textViewResult.setText(result);
            }
        });

        return bgr;
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    /*classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            IMAGE_MEAN,
                            IMAGE_STD,
                            INPUT_NAME,
                            OUTPUT_NAME);*/
                    featureExtractor=new TensorFlowImageFeatureExtractor(
                            getAssets(),
                            MODEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAMES);
                    //makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }
/*
    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnDetectObject.setVisibility(View.VISIBLE);
            }
        });
    }
    */
}
