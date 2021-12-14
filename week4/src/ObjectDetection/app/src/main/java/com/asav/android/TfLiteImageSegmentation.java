/*
borrowed from https://github.com/tensorflow/examples/blob/master/lite/examples/image_segmentation
 */
package com.asav.android;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import androidx.core.graphics.ColorUtils;

import org.tensorflow.lite.Interpreter;

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
public class TfLiteImageSegmentation {

    /** Tag for the {@link Log}. */
    private static final String TAG = "TfLiteImageSegmentation";

    private ByteBuffer segmentationMasks;
    private Interpreter interpreter;
    private static final String DEEP_LAB_MODEL = "deeplabv3_1_default_1.tflite";
    private static final int imageSize = 257;
    private static final int NUM_CLASSES = 21;
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;

    String labelsArrays []= {
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
            "person", "potted plant", "sheep", "sofa", "train", "tv"
    };
    private int colors[] = new int[NUM_CLASSES];

    public TfLiteImageSegmentation(final Context context) throws IOException {
        interpreter = getInterpreter(context, DEEP_LAB_MODEL);
        segmentationMasks = ByteBuffer.allocateDirect(1 * imageSize * imageSize * NUM_CLASSES * 4);
        segmentationMasks.order(ByteOrder.nativeOrder());

        colors[0] = Color.TRANSPARENT;
        for (int i = 1; i < NUM_CLASSES; ++i) {
            float hue=i*1.0f/NUM_CLASSES;
            colors[i] = HSBtoRGB(hue,1.0f,1.0f,128);
        }
    }
    public static int HSBtoRGB(float hue, float saturation, float brightness, int alpha) {
        int r = 0, g = 0, b = 0;
        if (saturation == 0) {
            r = g = b = (int) (brightness * 255.0f + 0.5f);
        } else {
            float h = (hue - (float)Math.floor(hue)) * 6.0f;
            float f = h - (float)java.lang.Math.floor(h);
            float p = brightness * (1.0f - saturation);
            float q = brightness * (1.0f - saturation * f);
            float t = brightness * (1.0f - (saturation * (1.0f - f)));
            switch ((int) h) {
                case 0:
                    r = (int) (brightness * 255.0f + 0.5f);
                    g = (int) (t * 255.0f + 0.5f);
                    b = (int) (p * 255.0f + 0.5f);
                    break;
                case 1:
                    r = (int) (q * 255.0f + 0.5f);
                    g = (int) (brightness * 255.0f + 0.5f);
                    b = (int) (p * 255.0f + 0.5f);
                    break;
                case 2:
                    r = (int) (p * 255.0f + 0.5f);
                    g = (int) (brightness * 255.0f + 0.5f);
                    b = (int) (t * 255.0f + 0.5f);
                    break;
                case 3:
                    r = (int) (p * 255.0f + 0.5f);
                    g = (int) (q * 255.0f + 0.5f);
                    b = (int) (brightness * 255.0f + 0.5f);
                    break;
                case 4:
                    r = (int) (t * 255.0f + 0.5f);
                    g = (int) (p * 255.0f + 0.5f);
                    b = (int) (brightness * 255.0f + 0.5f);
                    break;
                case 5:
                    r = (int) (brightness * 255.0f + 0.5f);
                    g = (int) (p * 255.0f + 0.5f);
                    b = (int) (q * 255.0f + 0.5f);
                    break;
            }
        }
        return (alpha<<24) | (r << 16) | (g << 8) | (b << 0);
    }

    private static Interpreter getInterpreter(final Context context,String modelFile) throws IOException {
        Interpreter.Options options = (new Interpreter.Options()).setNumThreads(4);//.addDelegate(delegate);
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

    public Bitmap execute(Bitmap data){
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(data,imageSize, imageSize, false);
        ByteBuffer contentArray = bitmapToByteBuffer(scaledBitmap, imageSize, imageSize, 0.0f, 255.0f);
        interpreter.run(contentArray, segmentationMasks);
        Bitmap maskImageApplied = convertBytebufferMaskToBitmap(segmentationMasks, imageSize, imageSize, scaledBitmap);
        return maskImageApplied;
    }
    private Bitmap convertBytebufferMaskToBitmap(ByteBuffer inputBuffer,int imageWidth, int imageHeight,Bitmap scaledBackgroundImage)
    {
        Bitmap maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        Bitmap resultBitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        inputBuffer.rewind();
        HashMap<String,Integer> foundClasses=new HashMap<>();
        for (int y=0;y<imageSize;++y) {
            for (int x=0;x<imageSize;++x) {
                float maxVal = 0f;
                int segmentBit = 0;

                for (int c=0; c<NUM_CLASSES;++c) {
                    float value = inputBuffer.getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4);
                    if (c == 0 || value > maxVal) {
                        maxVal = value;
                        segmentBit = c;
                    }
                }
                foundClasses.put(labelsArrays[segmentBit],foundClasses.getOrDefault(labelsArrays[segmentBit],0)+1);
                int newPixelColor = ColorUtils.compositeColors(
                        colors[segmentBit],
                        scaledBackgroundImage.getPixel(x, y)
                );
                resultBitmap.setPixel(x, y, newPixelColor);
                maskBitmap.setPixel(x, y, colors[segmentBit]);
            }
        }

        Log.i(TAG,"New segmentation results");
        for(String foundClass : foundClasses.keySet())
            Log.i(TAG,"segmentation result: "+foundClass+" ("+foundClasses.get(foundClass)+")");
        return resultBitmap;
        //return maskBitmap;
    }
    public void close() {
        interpreter.close();
    }
}
