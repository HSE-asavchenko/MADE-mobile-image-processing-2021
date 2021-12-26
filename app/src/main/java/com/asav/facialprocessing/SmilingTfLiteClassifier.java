package com.asav.facialprocessing;

import android.content.Context;
import android.util.Log;

import java.io.IOException;

/**
 * Created by avsavchenko.
 */
public class SmilingTfLiteClassifier extends TfLiteClassifier{

    /** Tag for the {@link Log}. */
    private static final String TAG = "SmilingTfLite";

    private static final String MODEL_FILE = "smiling_mobilenet.tflite";

    public SmilingTfLiteClassifier(final Context context) throws IOException {
        super(context,MODEL_FILE);
    }

    protected void addPixelValue(int val) {
        imgData.putFloat((val & 0xFF) - 103.939f);
        imgData.putFloat(((val >> 8) & 0xFF) - 116.779f);
        imgData.putFloat(((val >> 16) & 0xFF) - 123.68f);
    }

    protected ClassifierResult getResults(float[][][] outputs) {
        final float[] emotions_scores = outputs[0][0];
        SmilingData res=new SmilingData(emotions_scores);
        return res;
    }
}
