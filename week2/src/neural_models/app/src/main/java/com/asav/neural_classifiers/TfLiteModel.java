package com.asav.neural_classifiers;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
/*import org.tensorflow.lite.support.common.*;
import org.tensorflow.lite.support.image.*;
import org.tensorflow.lite.support.image.ops.*;*/

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by avsavchenko.
 */
public class TfLiteModel implements DeepModel{

    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteModel";

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;
    private Random rnd=new Random();

    private static final boolean useTensorImage=true;
    private int[] intValues = null;
    protected ByteBuffer imgData = null;
    private int imageSizeX=224,imageSizeY=224, numChannels=3;
    private int[] outputTensorShape;
    private float[][][] outputs;
    Map<Integer, Object> outputMap = new HashMap<>();
    private boolean isInt8Input=false, isInt8Output=false;
    private boolean efficientnet_preprocess =true;

    public TfLiteModel(final Context context, String model_path, boolean gpuFlag) throws IOException {
         Interpreter.Options options = (new Interpreter.Options()).setNumThreads(4);//.addDelegate(delegate);
         if (gpuFlag) {//change to false for emulator
             org.tensorflow.lite.gpu.GpuDelegate.Options opt=new org.tensorflow.lite.gpu.GpuDelegate.Options();
             opt.setInferencePreference(org.tensorflow.lite.gpu.GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
             org.tensorflow.lite.gpu.GpuDelegate delegate = new org.tensorflow.lite.gpu.GpuDelegate();
             options.addDelegate(delegate);
         }
        MappedByteBuffer tfliteModel= loadModelFile(context, model_path);
        tflite = new Interpreter(tfliteModel,options);
        tflite.allocateTensors();
        Tensor inputTensor = tflite.getInputTensor(0);
        int[] inputShape= inputTensor.shape();
        imageSizeX=inputShape[1];
        imageSizeY=inputShape[2];
        numChannels=inputShape[3];
        int numBytesPerChannel=4;
        if (inputTensor.dataType()== DataType.UINT8) {
            numBytesPerChannel = 1;
            isInt8Input=true;
        }
        intValues = new int[imageSizeX * imageSizeY];
        imgData = ByteBuffer.allocateDirect(imageSizeX*imageSizeY* numChannels*numBytesPerChannel);
        imgData.order(ByteOrder.nativeOrder());

        outputTensorShape = tflite.getOutputTensor(0).shape();
        int outputCount=tflite.getOutputTensorCount();
        outputs=new float[outputCount][1][];
        for(int i = 0; i< outputCount; ++i) {
            Tensor outputTensor=tflite.getOutputTensor(i);
            int[] shape=tflite.getOutputTensor(i).shape();
            int numOFFeatures = shape[1];
            numBytesPerChannel=4;
            if (outputTensor.dataType()== DataType.UINT8) {
                numBytesPerChannel = 1;
                isInt8Output = true;
            }
            outputs[i][0] = new float[numOFFeatures];
            ByteBuffer ith_output = ByteBuffer.allocateDirect( numOFFeatures* numBytesPerChannel);  // Float tensor, shape 3x2x4
            ith_output.order(ByteOrder.nativeOrder());
            outputMap.put(i, ith_output);
        }

        efficientnet_preprocess = model_path.toLowerCase().contains("efficientnet");
    }
    private static MappedByteBuffer loadModelFile(Context context, String modelFile) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        MappedByteBuffer retFile = inputStream.getChannel().map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
        fileDescriptor.close();
        return retFile;
    }


    public Pair<Long,float[]> classifyImage(Bitmap bitmap) {
        bitmap=Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, false);
        Object[] inputs={null};
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
        inputs[0] = imgData;
        long startTime = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, outputMap);
        for(int i = 0; i< outputs.length; ++i) {
            ByteBuffer ith_output=(ByteBuffer)outputMap.get(i);
            ith_output.rewind();
            int len=outputs[i][0].length;
            for(int j=0;j<len;++j){
                if(isInt8Output)
                    outputs[i][0][j]=ith_output.get();
                else
                    outputs[i][0][j]=ith_output.getFloat();
            }
            ith_output.rewind();
        }
        long endTime = SystemClock.uptimeMillis();
        Log.i(TAG, "tf lite timecost to run model inference: " + Long.toString(endTime - startTime));

        return new Pair<Long,float[]>(endTime - startTime,outputs[0][0]);
    }

    public void close() {
        tflite.close();
    }

    protected void addPixelValue(int val) {
        if(isInt8Input){
            imgData.put((byte)((val >> 16) & 0xFF));
            imgData.put((byte)((val >> 8) & 0xFF));
            imgData.put((byte)(val & 0xFF));
        }
        else {
            if(efficientnet_preprocess){
                imgData.putFloat((((val >> 16) & 0xFF) / 255.0f - 0.485f) / 0.229f);
                imgData.putFloat((((val >> 8) & 0xFF) / 255.0f - 0.456f) / 0.224f);
                imgData.putFloat(((val & 0xFF) / 255.0f - 0.406f) / 0.225f);
            }
            else {
                float std = 127.5f; //mobilenet
                imgData.putFloat(((val >> 16) & 0xFF) / std - 1.0f);
                imgData.putFloat(((val >> 8) & 0xFF) / std - 1.0f);
                imgData.putFloat((val & 0xFF) / std - 1.0f);
            }
        }
    }
}
