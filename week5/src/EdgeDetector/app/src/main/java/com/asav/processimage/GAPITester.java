package com.asav.processimage;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class GAPITester {
    public GAPITester(){
        handler=gapiInit();
    }
    public Mat processImage(Mat in){
        Mat out=new Mat();
        if (handler != 0) {
            Mat rgb=new Mat();
            Imgproc.cvtColor(in, rgb, Imgproc.COLOR_RGBA2RGB);
            gapiTest(handler,rgb.getNativeObjAddr(),out.getNativeObjAddr());
        }
        return out;
    }

    private static native long gapiInit();
    private static native void gapiTest(long handler, long matAddrIn, long matAddrOut);

    private long handler=0;
}
