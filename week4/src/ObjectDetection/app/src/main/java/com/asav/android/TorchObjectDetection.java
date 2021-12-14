package com.asav.android;

import android.content.Context;
import android.graphics.Bitmap;

import org.pytorch.*;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Adapted version of https://github.com/pytorch/android-demo-app/blob/master/ObjectDetection/app/src/main/java/org/pytorch/demo/objectdetection/PrePostProcessor.java
 */
public class TorchObjectDetection implements ObjectDetector{
    private static final String TAG = "TorchObjectDetection";

    private ArrayList<String> labels = new ArrayList<String>();
    
    private static final String MODEL_FILE = "yolov5s.torchscript.ptl";
    private static final String TORCH_LABELS_FILE ="coco_classes.txt";

    private Module mModule=null;

    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    static int mInputSize = 640;
    // model output is of size 25200*(num_of_class+5)
    static int mOutputRow = 25200; // as decided by the YOLOv5 model for input image of size 640*640
    static int mOutputColumn = 85; // left, top, right, bottom, score and 80 class probability
    static float mThreshold = 0.30f; // score above which a detection is generated
    static int mNmsLimit = 15;

    public TorchObjectDetection(final Context context) throws IOException {
        mModule = LiteModuleLoader.load(assetFilePath(context,MODEL_FILE ));
        BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(TORCH_LABELS_FILE)));
        String line;
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }
        br.close();
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    // The two methods nonMaxSuppression and IOU below are ported from https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     - Parameters:
     - boxes: an array of bounding boxes and their scores
     - limit: the maximum number of boxes that will be selected
     - threshold: used to decide whether boxes overlap too much
     */
    static ArrayList<DetectorData> nonMaxSuppression(ArrayList<DetectorData> boxes, int limit, float threshold) {

        // Do an argsort on the confidence scores, from high to low.
        Collections.sort(boxes,
                new Comparator<DetectorData>() {
                    @Override
                    public int compare(DetectorData o1, DetectorData o2) {
                        return o1.confidence.compareTo(o2.confidence);
                    }
                });

        ArrayList<DetectorData> selected = new ArrayList<>();
        boolean[] active = new boolean[boxes.size()];
        Arrays.fill(active, true);
        int numActive = active.length;

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        boolean done = false;
        for (int i=0; i<boxes.size() && !done; i++) {
            if (active[i]) {
                DetectorData boxA = boxes.get(i);
                selected.add(boxA);
                if (selected.size() >= limit) break;

                for (int j=i+1; j<boxes.size(); j++) {
                    if (active[j]) {
                        DetectorData boxB = boxes.get(j);
                        if (IOU(boxA.location, boxB.location) > threshold) {
                            active[j] = false;
                            numActive -= 1;
                            if (numActive <= 0) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return selected;
    }

    /**
     Computes intersection-over-union overlap between two bounding boxes.
     */
    private static float IOU(RectFloat a, RectFloat b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0.0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0.0) return 0.0f;

        float intersectionMinX = Math.max(a.left, b.left);
        float intersectionMinY = Math.max(a.top, b.top);
        float intersectionMaxX = Math.min(a.right, b.right);
        float intersectionMaxY = Math.min(a.bottom, b.bottom);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    private ArrayList<DetectorData> outputsToNMSPredictions(float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<DetectorData> DetectorDatas = new ArrayList<>();
        for (int i = 0; i< mOutputRow; i++) {
            if (outputs[i* mOutputColumn +4] > mThreshold) {
                float x = outputs[i* mOutputColumn];
                float y = outputs[i* mOutputColumn +1];
                float w = outputs[i* mOutputColumn +2];
                float h = outputs[i* mOutputColumn +3];

                float left = imgScaleX * (x - w/2);
                float top = imgScaleY * (y - h/2);
                float right = imgScaleX * (x + w/2);
                float bottom = imgScaleY * (y + h/2);

                float max = outputs[i* mOutputColumn +5];
                int clsInd = 0;
                for (int j = 0; j < mOutputColumn -5; j++) {
                    if (outputs[i* mOutputColumn +5+j] > max) {
                        max = outputs[i* mOutputColumn +5+j];
                        clsInd = j;
                    }
                }

                RectFloat rect = new RectFloat(startX+ivScaleX*left, startY+top*ivScaleY, startX+ivScaleX*right, startY+ivScaleY*bottom);
                DetectorData DetectorData = new DetectorData(labels.get(clsInd), outputs[i*mOutputColumn+4], rect);
                DetectorDatas.add(DetectorData);
            }
        }
        return nonMaxSuppression(DetectorDatas, mNmsLimit, mThreshold);
    }

    public List<DetectorData> detectObjects(final Bitmap bitmap) {
        Bitmap resizedBitmap=resizeBitmap(bitmap);
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, NO_MEAN_RGB, NO_STD_RGB);
        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] outputs = outputTensor.getDataAsFloatArray();
        float scale=1.0f/ mInputSize;
        final ArrayList<DetectorData> recognitions = outputsToNMSPredictions(outputs, scale, scale, 1, 1, 0, 0);
        return recognitions;
    }

    public Bitmap resizeBitmap(Bitmap bitmap) {
        Bitmap resizedBitmap=bitmap;
        if(bitmap.getWidth()!= mInputSize || bitmap.getHeight()!= mInputSize){
            resizedBitmap=Bitmap.createScaledBitmap(bitmap, mInputSize, mInputSize,false);
        }
        return resizedBitmap;
    }

}
