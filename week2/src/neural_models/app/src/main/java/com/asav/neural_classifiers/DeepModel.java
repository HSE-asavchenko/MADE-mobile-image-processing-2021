package com.asav.neural_classifiers;

import android.graphics.Bitmap;
import android.util.Pair;

public interface DeepModel {
    public Pair<Long,float[]> classifyImage(Bitmap bitmap);
}
