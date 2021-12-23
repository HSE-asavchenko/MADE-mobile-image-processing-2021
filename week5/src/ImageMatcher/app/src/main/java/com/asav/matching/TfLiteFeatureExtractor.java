package com.asav.matching;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class TfLiteFeatureExtractor {
    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteFeatureExtractor";

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = null;
    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private int imageSizeX=224,imageSizeY=224;
    protected ByteBuffer imgData = null;
    private float[][][] all_descriptors =null;
    Map<Integer, Object> outputMap = new HashMap<>();

    public TfLiteFeatureExtractor(final AssetManager assetManager) throws IOException {
        String model_path="superpoint.tflite";
        Interpreter.Options options = (new Interpreter.Options()).setNumThreads(4);//.addDelegate(delegate);
        /*if (false) {
            GpuDelegate.Options opt=new GpuDelegate.Options();
            opt.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            GpuDelegate delegate = new GpuDelegate();
            options.addDelegate(delegate);
        }*/

        tflite = new Interpreter(loadModelFile(assetManager,model_path),options);
        tflite.allocateTensors();
        int[] inputShape=tflite.getInputTensor(0).shape();
        imageSizeX=inputShape[1];
        imageSizeY=inputShape[2];

        intValues = new int[imageSizeX * imageSizeY];
        imgData =ByteBuffer.allocateDirect(imageSizeX*imageSizeY* inputShape[3]*getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());

        int outputCount=tflite.getOutputTensorCount();
        outputCount=1;//Only positions of keypoints
        for(int i = 0; i< outputCount; ++i) {
            int[] shape=tflite.getOutputTensor(i).shape();
            int numOFFeatures = 1;
            for(int j=0;j<shape.length;++j)
                numOFFeatures*=shape[j];
            Log.i(TAG, "Read output layer size is " + numOFFeatures);
            ByteBuffer ith_output = ByteBuffer.allocateDirect( numOFFeatures* getNumBytesPerChannel());  // Float tensor, shape 3x2x4
            ith_output.order(ByteOrder.nativeOrder());
            outputMap.put(i, ith_output);
            if(i==1)
                all_descriptors = new float[shape[1]][shape[2]][shape[3]];
        }
    }
    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(final AssetManager assetManager, String model_path) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(model_path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    protected void addPixelValue(int val) {
        int r=(val >> 16) & 0xFF;
        int b=(val >> 8) & 0xFF;
        int g=val & 0xFF;

        imgData.putFloat((0.299f*r+0.587f*g+0.114f*b)/255);
    }

    public Mat processImage(Mat image,MatOfKeyPoint keyPoints) {
        Mat resizedImage=new Mat();
        Imgproc.resize(image,resizedImage, new Size(imageSizeX,imageSizeY));
        Bitmap bitmap = Bitmap.createBitmap(resizedImage.cols(),
                resizedImage.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(resizedImage, bitmap);
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        if (imgData == null) {
            return null;
        }
        imgData.rewind();
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
        long startTime = SystemClock.uptimeMillis();
        Object[] inputs = {imgData};
        tflite.runForMultipleInputsOutputs(inputs, outputMap);

        //descriptors
        ByteBuffer ith_output=(ByteBuffer)outputMap.get(0);
        ith_output.rewind();
        final float threshold=0.3f;
        ArrayList<KeyPoint> lkp=new ArrayList<>();
        for (int j = 0; j < imageSizeY; ++j) {
            for (int i = 0; i < imageSizeX; ++i) {
                float prob = ith_output.getFloat();
                if(prob>threshold){
                    lkp.add(new KeyPoint(((float)i)/imageSizeX*image.cols(),((float)j)/imageSizeY*image.rows(),1));
                }
            }
        }
        ith_output.rewind();
        keyPoints.fromList(lkp);

        Mat descriptors=null;
        /*
        ith_output=(ByteBuffer)outputMap.get(1);
        ith_output.rewind();
        for (int j = 0; j < imageSizeY; ++j) {
            for (int i = 0; i < imageSizeX; ++i) {
                for (int k = 0; k < all_descriptors[i][j].length; ++k)
                    all_descriptors[i][j][k] = ith_output.getFloat();
            }
        }
        ith_output.rewind();
        int num_features=all_descriptors[0][0].length;
        descriptors=new Mat(new Size(num_features,lkp.size()),CvType.CV_32FC1);
        int row=0;
        for (int j = 0; j < imageSizeY; ++j) {
            for (int i = 0; i < imageSizeX; ++i) {
                float prob = ith_output.getFloat();
                if(prob>threshold){
                    for(int f=0;f<num_features;++f)
                        descriptors.put(row,f,all_descriptors[i][j][f]);
                    ++row;
                }
            }
        }
        ith_output.rewind();*/

        long endTime = SystemClock.uptimeMillis();
        Log.i(TAG, "tf lite timecost to run model inference: " + Long.toString(endTime - startTime));

        return descriptors;
    }

    public void close() {
        tflite.close();
    }

    protected int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }

}
