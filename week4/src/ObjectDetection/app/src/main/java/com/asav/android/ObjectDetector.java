package com.asav.android;

import android.graphics.Bitmap;

import java.util.List;

public interface ObjectDetector {
    List<DetectorData> detectObjects(final Bitmap bitmap);
    Bitmap resizeBitmap(Bitmap bitmap);
}
