package com.asav.android;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;

import java.io.BufferedReader;
import java.io.FileInputStream;
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

/**
 * Adapted version of https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/android/app/src/main/java/org/tensorflow/demo/TFLiteObjectDetectionAPIModel.java

 */
public class TfLiteYoloObjectDetection implements ObjectDetector{
    private static final String TAG = "TFLiteYoloObjectDetection";

    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private ArrayList<String> labels = new ArrayList<String>();

    private int inputSize=TorchObjectDetection.mInputSize;
    private int[] intValues;
    private float[][][] outputs;

    private ByteBuffer imgData;

    private Interpreter tfLite;

    private static final String MODEL_FILE ="yolov5s-fp16.tflite";
    private static final String LABELS_FILE ="coco_classes.txt";
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 255.0f;

    public TfLiteYoloObjectDetection(final AssetManager assetManager) throws IOException {
        InputStream labelsInput = null;
        labelsInput = assetManager.open(LABELS_FILE);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }
        br.close();

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
        outputs=new float[1][TorchObjectDetection.mOutputRow][TorchObjectDetection.mOutputColumn];
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
    private ArrayList<DetectorData> outputsToNMSPredictions(float[][] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<DetectorData> DetectorDatas = new ArrayList<>();
        int numClasses=TorchObjectDetection.mOutputColumn-5;
        for (int i = 0; i< outputs.length; i++) {
            if (outputs[i][4] > TorchObjectDetection.mThreshold) {
                float x = outputs[i][0];
                float y = outputs[i][1];
                float w = outputs[i][2];
                float h = outputs[i][3];

                float left = imgScaleX * (x - w/2);
                float top = imgScaleY * (y - h/2);
                float right = imgScaleX * (x + w/2);
                float bottom = imgScaleY * (y + h/2);

                float max = 0;
                int clsInd = 0;
                for (int j = 0; j < numClasses; j++) {
                    if (outputs[i][5+j] > max) {
                        max = outputs[i][5+j];
                        clsInd = j;
                    }
                }

                RectFloat rect = new RectFloat(startX+ivScaleX*left, startY+top*ivScaleY, startX+ivScaleX*right, startY+ivScaleY*bottom);
                DetectorData DetectorData = new DetectorData(labels.get(clsInd), outputs[i][4], rect);
                DetectorDatas.add(DetectorData);
            }
        }
        return TorchObjectDetection.nonMaxSuppression(DetectorDatas, TorchObjectDetection.mNmsLimit, TorchObjectDetection.mThreshold);
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

        // Run the inference call.
        tfLite.run(imgData,outputs);
        float scale=1.0f;/// inputSize;
        final ArrayList<DetectorData> recognitions = outputsToNMSPredictions(outputs[0], scale, scale, 1, 1, 0, 0);
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
