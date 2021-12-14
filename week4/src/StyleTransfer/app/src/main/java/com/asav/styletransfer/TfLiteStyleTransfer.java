package com.asav.styletransfer;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.*;
import org.tensorflow.lite.support.image.*;
import org.tensorflow.lite.support.image.ops.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by avsavchenko.
 */
public class TfLiteStyleTransfer {

    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteStyleTransfer";

    protected Interpreter interpreterPredict, interpreterTransform;
    private final static int STYLE_IMAGE_SIZE = 256;
    private final static int CONTENT_IMAGE_SIZE = 384;
    private final static int BOTTLENECK_SIZE = 100;
    private final static String STYLE_PREDICT_FLOAT16_MODEL = "magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite";
    private final static String STYLE_TRANSFER_FLOAT16_MODEL = "magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.tflite";

    public TfLiteStyleTransfer(final Context context) throws IOException {
        interpreterPredict=getInterpreter(context,STYLE_PREDICT_FLOAT16_MODEL);
        interpreterTransform =getInterpreter(context,STYLE_TRANSFER_FLOAT16_MODEL);
    }
    private static Interpreter getInterpreter(final Context context,String modelFile) throws IOException {
        Interpreter.Options options = (new Interpreter.Options()).setNumThreads(4);//.addDelegate(delegate);
        CompatibilityList compatList = new CompatibilityList();
        boolean hasGPU=compatList.isDelegateSupportedOnThisDevice();
        if (hasGPU) {
            org.tensorflow.lite.gpu.GpuDelegate.Options opt=new org.tensorflow.lite.gpu.GpuDelegate.Options();
            opt.setInferencePreference(org.tensorflow.lite.gpu.GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            org.tensorflow.lite.gpu.GpuDelegate delegate = new org.tensorflow.lite.gpu.GpuDelegate(opt);
            options.addDelegate(delegate);
        }

        MappedByteBuffer tfliteModel= loadModelFile(context,modelFile);
        Interpreter tflite = new Interpreter(tfliteModel,options);
        return tflite;
    }
    private static MappedByteBuffer loadModelFile(Context context, String modelFile) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        MappedByteBuffer retFile = inputStream.getChannel().map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
        fileDescriptor.close();
        return retFile;
    }

    private ByteBuffer bitmapToByteBuffer(Bitmap bitmapIn, int width, int height, float mean, float std){
        Bitmap bitmap = Bitmap.createScaledBitmap(bitmapIn, width, height, false);

        ByteBuffer inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4);
        inputImage.order(ByteOrder.nativeOrder());
        inputImage.rewind();

        int intValues[] = new int[width * height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                final int val = intValues[pixel++];
                inputImage.putFloat((((val >> 16) & 0xFF) - mean) / std);
                inputImage.putFloat((((val >> 8) & 0xFF) - mean) / std);
                inputImage.putFloat(((val & 0xFF) - mean) / std);
            }
        }
        inputImage.rewind();
        return inputImage;
    }

    public Bitmap execute(Bitmap contentImage,Bitmap styleBitmap){
        ByteBuffer contentArray =bitmapToByteBuffer(contentImage, CONTENT_IMAGE_SIZE, CONTENT_IMAGE_SIZE, 0.0f, 255.0f);
        ByteBuffer input =bitmapToByteBuffer(styleBitmap, STYLE_IMAGE_SIZE, STYLE_IMAGE_SIZE, 0.0f, 255.0f);
        Object[] inputsForPredict={input};
        Map<Integer, Object> outputsForPredict = new HashMap<>();
        float styleBottleneck[][][][]=new float[1][1][1][BOTTLENECK_SIZE];
        outputsForPredict.put(0, styleBottleneck);
        // The results of this inference could be reused given the style does not change
        // That would be a good practice in case this was applied to a video stream.
        interpreterPredict.runForMultipleInputsOutputs(inputsForPredict, outputsForPredict);

        Object[] inputsForStyleTransfer={contentArray, styleBottleneck};
        Map<Integer, Object> outputsForStyleTransfer = new HashMap<>();
        float outputImage[][][][]=new float[1][CONTENT_IMAGE_SIZE][CONTENT_IMAGE_SIZE][3];
        outputsForStyleTransfer.put(0, outputImage);

        interpreterTransform.runForMultipleInputsOutputs(inputsForStyleTransfer,outputsForStyleTransfer);

        Bitmap styledImage = Bitmap.createBitmap(CONTENT_IMAGE_SIZE, CONTENT_IMAGE_SIZE, Bitmap.Config.ARGB_8888 );

        for (int x=0;x<CONTENT_IMAGE_SIZE;++x) {
            for (int y=0;y<CONTENT_IMAGE_SIZE;++y) {
                // this y, x is in the correct order!!!
                styledImage.setPixel(y, x, Color.rgb(
                        ((int)(outputImage[0][x][y][0] * 255)),
                        ((int)(outputImage[0][x][y][1] * 255)),
                        (int)(outputImage[0][x][y][2] * 255)));
            }
        }
        return styledImage;
    }

    public void close() {
        interpreterPredict.close();
        interpreterTransform.close();
    }

    protected int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }
}
