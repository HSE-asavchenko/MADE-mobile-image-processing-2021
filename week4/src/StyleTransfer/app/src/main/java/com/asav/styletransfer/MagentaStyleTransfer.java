/*
Copied from https://github.com/ajeet-repos/Artistic-Style-Transfer-Using-Tensorflow-on-Android
 */
package com.asav.styletransfer;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.Arrays;

public class MagentaStyleTransfer {
    private static final String TAG = "MagentaStyleTransfer";
    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/stylize_quantized.pb";

    private static final String INPUT_NODE = "input";
    private static final String STYLE_NODE = "style_num";
    private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";

    public static final int NUM_STYLES = 26;
    private float[] styleVals =new float[NUM_STYLES];

    private int desiredSize = 256;
    private int[] intValues = new int[desiredSize * desiredSize];

    private float[] floatValues = new float[desiredSize * desiredSize * 3];;

    public MagentaStyleTransfer(AssetManager context){
        try {
            inferenceInterface = new TensorFlowInferenceInterface(context, MODEL_FILE);
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing TensorFlowInferenceInterface!", e);
        }
    }

    public Bitmap stylizeImage(Bitmap inputBitmap, int style) {
        if(inferenceInterface==null)
            return null;
        Bitmap bitmap = Bitmap.createScaledBitmap(inputBitmap, desiredSize, desiredSize, false);

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
        }
        Arrays.fill(styleVals,0.0f);
        styleVals[style]=1.0f;
        //styleVals[11] = 1.0f;
        //styleVals[19] = 1.0f;
        //styleVals[24] = 1.0f;

        // Copy the input data into TensorFlow.
        inferenceInterface.feed(INPUT_NODE, floatValues, 1, bitmap.getWidth(), bitmap.getHeight(), 3);
        inferenceInterface.feed(STYLE_NODE, styleVals, NUM_STYLES);

        // Execute the output node's dependency sub-graph.
        inferenceInterface.run(new String[] {OUTPUT_NODE}, false);

        // Copy the data from TensorFlow back into our array.
        inferenceInterface.fetch(OUTPUT_NODE, floatValues);

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (floatValues[i * 3] * 255)) << 16)
                            | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                            | ((int) (floatValues[i * 3 + 2] * 255));
        }

        bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        return bitmap;
    }

}
