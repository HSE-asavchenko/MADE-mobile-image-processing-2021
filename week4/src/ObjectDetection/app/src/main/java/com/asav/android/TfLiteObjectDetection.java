package com.asav.android;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Adapted version of https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/android/app/src/main/java/org/tensorflow/demo/TFLiteObjectDetectionAPIModel.java

 */
public class TfLiteObjectDetection  implements ObjectDetector{
    private static final String TAG = "TFLiteObjectDetection";

    // Only return this many results.
    private static final int NUM_DETECTIONS = 30;
    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    // Config values.
    private int inputSize;
    // Pre-allocated buffers.
    private ArrayList<String> labels = new ArrayList<String>();

    private int[] intValues;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the topScores of detected boxes
    private float[][] outputScores;

    private ByteBuffer imgData;

    private Interpreter tfLite;

    private static final String MODEL_FILE ="tflite_graph.tflite";
    private static final String TF_OD_API_LABELS_FILE ="oid_classes.txt";

    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    /**
    * @param assetManager The asset manager to be used to load assets.
    */

    public TfLiteObjectDetection(final AssetManager assetManager) throws IOException {
        InputStream labelsInput = null;
        labelsInput = assetManager.open(TF_OD_API_LABELS_FILE);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            line=line.toLowerCase().split("#")[0].trim();
            String[] categoryInfo=line.split("=");
            String category=categoryInfo[0];
            labels.add(category);
        }
        br.close();

        inputSize = 300;

        Interpreter.Options options = (new Interpreter.Options()).setNumThreads(NUM_THREADS);//.addDelegate(delegate);
        CompatibilityList compatList = new CompatibilityList();
        boolean hasGPU=compatList.isDelegateSupportedOnThisDevice();
        if (hasGPU) {
            org.tensorflow.lite.gpu.GpuDelegate.Options opt=new org.tensorflow.lite.gpu.GpuDelegate.Options();
            opt.setInferencePreference(org.tensorflow.lite.gpu.GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            org.tensorflow.lite.gpu.GpuDelegate delegate = new org.tensorflow.lite.gpu.GpuDelegate(opt);
            options.addDelegate(delegate);
        }
        tfLite = new Interpreter(loadModelFile(assetManager,MODEL_FILE),options);

        // Pre-allocate buffers.
        int numBytesPerChannel=4;
        imgData = ByteBuffer.allocateDirect(1 *  inputSize *  inputSize * 3 * numBytesPerChannel);
        imgData.order(ByteOrder.nativeOrder());
        intValues = new int[ inputSize *  inputSize];
    }

    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(final AssetManager assetManager, String model_path) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(model_path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public List<DetectorData> detectObjects(final Bitmap bitmap) {

        Bitmap resizedBitmap=resizeBitmap(bitmap);
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                // Float model
                imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }

        // Copy the input data into TensorFlow.
        outputLocations = new float[1][NUM_DETECTIONS][4];
        outputClasses = new float[1][NUM_DETECTIONS];
        outputScores = new float[1][NUM_DETECTIONS];
        float[] numDetections = new float[1];

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);

        // Run the inference call.
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        // Show the best detections.
        // after scaling them back to the input size.
        final ArrayList<DetectorData> recognitions = new ArrayList<DetectorData>();
        for (int i = 0; i < Math.min((int)numDetections[0],outputScores[0].length); ++i) {
        //for (int i = 0; i < outputScores[0].length; ++i) {
            if (outputScores[0][i]>0.4) {
                final RectFloat detection =
                        new RectFloat(
                                outputLocations[0][i][1],
                                outputLocations[0][i][0],
                                outputLocations[0][i][3],
                                outputLocations[0][i][2]);
                recognitions.add(
                        new DetectorData(labels.get((int) outputClasses[0][i] + 1), outputScores[0][i], detection));
            }
        }
        return recognitions;
    }

    public void close() {
        tfLite.close();
    }


    public Bitmap resizeBitmap(Bitmap bitmap) {
        Bitmap resizedBitmap=bitmap;
        if(bitmap.getWidth()!=inputSize || bitmap.getHeight()!=inputSize){
            resizedBitmap=Bitmap.createScaledBitmap(bitmap,inputSize, inputSize,false);
        }
        return resizedBitmap;
    }

}
