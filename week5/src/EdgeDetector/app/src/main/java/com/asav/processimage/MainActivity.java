package com.asav.processimage;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import androidx.exifinterface.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowMetrics;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opencv.core.Core.DFT_SCALE;
import static org.opencv.core.CvType.CV_8U;

public class MainActivity extends AppCompatActivity {
    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private ImageView imageView;
    private Mat sampledImage=null;
    private GAPITester tester=null;
    private ArrayList<org.opencv.core.Point> corners=new ArrayList<org.opencv.core.Point>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        imageView=(ImageView)findViewById(R.id.inputImageView);
        imageView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent event) {
                Log.i(TAG, "event.getX(), event.getY(): " + event.getX() +" "+ event.getY());
                if(sampledImage!=null) {
                    Log.i(TAG, "sampledImage.width(), sampledImage.height(): " + sampledImage.width() +" "+ sampledImage.height());
                    Log.i(TAG, "view.getWidth(), view.getHeight(): " + view.getWidth() +" "+ view.getHeight());
                    int left=(view.getWidth()-sampledImage.width())/2;
                    int top=(view.getHeight()-sampledImage.height())/2;
                    int right=(view.getWidth()+sampledImage.width())/2;
                    int bottom=(view.getHeight()+sampledImage.height())/2;
                    Log.i(TAG, "left: " + left +" right: "+ right +" top: "+ top +" bottom:"+ bottom);
                    if(event.getX()>=left && event.getX()<=right && event.getY()>=top && event.getY()<=bottom) {
                        int projectedX = (int)event.getX()-left;
                        int projectedY = (int)event.getY()-top;
                        org.opencv.core.Point corner = new org.opencv.core.Point(projectedX, projectedY);
                        corners.add(corner);
                        if(corners.size()>4)
                            corners.remove(0);
                        Mat sampleImageCopy=sampledImage.clone();
                        for(org.opencv.core.Point c : corners)
                            Imgproc.circle(sampleImageCopy, c, (int) 5, new Scalar(0, 0, 255), 2);
                        displayImage(sampleImageCopy);
                    }
                }
                return false;
            }
        });

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else
            init();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_menu, menu);
        return true;
    }
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    System.loadLibrary("ImageProcessLib");
                    if(tester==null)
                        tester=new GAPITester();
                    Log.i(TAG, "After loading all libraries" );
                    Toast.makeText(getApplicationContext(),
                            "OpenCV loaded successfully",
                            Toast.LENGTH_SHORT).show();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                    Toast.makeText(getApplicationContext(),
                            "OpenCV error",
                            Toast.LENGTH_SHORT).show();
                } break;
            }
        }
    };
    private void init(){
    }
    @Override
    public void onResume()
    {
        super.onResume();
        if(tester==null) {
            if (!OpenCVLoader.initDebug()) {
                Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
            } else {
                Log.d(TAG, "OpenCV library found inside package. Using it!");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }
        }
    }
    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status= ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (allGranted) {
                    // All Permissions Granted
                    init();
                } else {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    private static final int SELECT_PICTURE = 1;
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_openGallery:
                corners.clear();
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"Select Picture"),
                        SELECT_PICTURE);
                return true;

            case R.id.action_edgedetector:
                if(isImageLoaded()) {
                    edgedetector();
                }
                return true;
            case R.id.action_canny:
                if(isImageLoaded()) {
                    canny();
                }
                return true;
            case R.id.action_linesdetector:
                if(isImageLoaded()) {
                    linesDetector();
                }
                return true;
            case R.id.action_circlesdetector:
                if(isImageLoaded()) {
                    circlesDetector();
                }
                return true;
            case R.id.action_transformer:
                if(isImageLoaded()) {
                    perspectiveTransform();
                }
                return true;
            case R.id.action_gapi_test:
                if(isImageLoaded()) {
                    test_gapi();
                }
                return true;

            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                return super.onOptionsItemSelected(item);
        }
    }
    private boolean isImageLoaded(){
        if(sampledImage==null)
            Toast.makeText(getApplicationContext(),
                    "It is necessary to open image firstly",
                    Toast.LENGTH_SHORT).show();
        return sampledImage!=null;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == SELECT_PICTURE && resultCode == RESULT_OK) {
            Uri selectedImageUri = data.getData(); //The uri with the location of the file
            Log.d(TAG,"uri"+selectedImageUri);
            convertToMat(selectedImageUri);
        }
    }
    private void convertToMat(Uri selectedImageUri)
    {
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            Bitmap bmp=BitmapFactory.decodeStream(ims);
            Mat rgbImage=new Mat();
            Utils.bitmapToMat(bmp, rgbImage);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    1);
            switch (orientation)
            {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    //get the mirrored image
                    rgbImage=rgbImage.t();
                    //flip on the y-axis
                    Core.flip(rgbImage, rgbImage, 1);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    //get up side down image
                    rgbImage=rgbImage.t();
                    //Flip on the x-axis
                    Core.flip(rgbImage, rgbImage, 0);
                    break;
            }

            Display display = getWindowManager().getDefaultDisplay();
            android.graphics.Point size = new android.graphics.Point();
            display.getSize(size);
            int width = size.x;
            int height = size.y;
            double downSampleRatio= calculateSubSampleSize(rgbImage,width,height);
            sampledImage=new Mat();
            Imgproc.resize(rgbImage, sampledImage, new
                    Size(),downSampleRatio,downSampleRatio,Imgproc.INTER_AREA);
            displayImage(sampledImage);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
            sampledImage=null;
        }
    }
    private static double calculateSubSampleSize(Mat srcImage, int reqWidth,
                                                 int reqHeight) {
        final int height = srcImage.height();
        final int width = srcImage.width();
        double inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            final double heightRatio = (double) reqHeight / (double) height;
            final double widthRatio = (double) reqWidth / (double) width;
            inSampleSize = heightRatio<widthRatio ? heightRatio :widthRatio;
        }
        return inSampleSize;
    }

    private void edgedetector(){
        Mat grayImage=new Mat();
        Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);

        Mat xFirstDerivative = new Mat(),yFirstDerivative =new Mat();
        Imgproc.Scharr(grayImage, xFirstDerivative,-1 , 1,0);
        Imgproc.Scharr(grayImage, yFirstDerivative,-1 , 0,1);
        Mat absXD=new Mat(),absYD=new Mat();
        Core.convertScaleAbs(xFirstDerivative, absXD);
        Core.convertScaleAbs(yFirstDerivative, absYD);
        Mat edgeImage=new Mat();
        double alpha=0.5;
        Core.addWeighted(absXD, alpha, absYD, 1-alpha, 0, edgeImage);
        displayImage(edgeImage);
    }
    private void canny(){
        Mat grayImage=new Mat();
        Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);
        Mat edgeImage=new Mat();
        Imgproc.Canny(grayImage, edgeImage, 100, 200);
        displayImage(edgeImage);
    }

    private void test_gapi(){
        Mat outImage =null;
        long startTime = SystemClock.uptimeMillis();
        if(false) {
            if (tester != null)
                outImage = tester.processImage(sampledImage);
        }
        else{
            outImage =new Mat();
            Imgproc.resize(sampledImage, outImage,new Size(),0.5,0.5);
            Mat tmpImg=new Mat();
            Imgproc.cvtColor(outImage,tmpImg,Imgproc.COLOR_RGB2GRAY);
            Imgproc.blur(tmpImg,tmpImg,new Size(5,5));
            Mat edges=new Mat();
            Imgproc.Canny(tmpImg,edges, 32, 128,3);
            ArrayList<Mat> rgb_list = new ArrayList(3);
            Core.split(outImage,rgb_list);
            Core.bitwise_or(rgb_list.get(1),edges,edges);
            rgb_list.set(1,edges);
            Core.merge(rgb_list,outImage);

        }
        //Mat outImage = new Mat();
        //gapiTest(rgb.getNativeObjAddr(),outImage.getNativeObjAddr());
        Log.i(TAG, "Timecost to process image in G-API: " + Long.toString(SystemClock.uptimeMillis() - startTime));
        displayImage(outImage);
    }

    private void linesDetector(){
        Mat grayImage=new Mat();
        Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Canny(grayImage, grayImage, 100, 200);
        Mat lines = new Mat();
        if(true) {
            int threshold = 150;
            Imgproc.HoughLinesP(grayImage, lines, 2, 2*Math.PI / 180, threshold);
            Imgproc.cvtColor(grayImage, grayImage, Imgproc.COLOR_GRAY2RGB);
            Log.w(TAG, "rows:" + lines.rows() + " cols:" + lines.cols());
            for (int i = 0; i < lines.rows(); i++) {
                double[] line = lines.get(i, 0);
                double xStart = line[0],
                        yStart = line[1],
                        xEnd = line[2],
                        yEnd = line[3];
                org.opencv.core.Point lineStart = new org.opencv.core.Point(xStart,
                        yStart);
                org.opencv.core.Point lineEnd = new org.opencv.core.Point(xEnd, yEnd);
                Imgproc.line(grayImage, lineStart, lineEnd, new Scalar(0, 255, 0), 3);
            }
        }
        else{
            int threshold = 200;
            Imgproc.HoughLines(grayImage, lines, 2, 2*Math.PI / 180, threshold);
            Imgproc.cvtColor(grayImage, grayImage, Imgproc.COLOR_GRAY2RGB);
            Log.w(TAG, "rows:" + lines.rows() + " cols:" + lines.cols());
            for (int i = 0; i < lines.rows(); i++) {
                double[] data = lines.get(i, 0);
                double rho1 = data[0];
                double theta1 = data[1];
                double cosTheta = Math.cos(theta1);
                double sinTheta = Math.sin(theta1);
                double x0 = cosTheta * rho1;
                double y0 = sinTheta * rho1;
                int line_len=100;
                Point pt1 = new Point(x0 + line_len * (-sinTheta), y0 + line_len * cosTheta);
                Point pt2 = new Point(x0 - line_len * (-sinTheta), y0 - line_len * cosTheta);
                Imgproc.line(grayImage, pt1, pt2, new Scalar(0, 255, 0), 2);
            }
        }
        displayImage(grayImage);
    }
    private void circlesDetector(){
        new Thread(new Runnable() {
            @Override
            public void run() {
                final Mat grayImage=new Mat();
                Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
                double minDist=10;
                Mat circles = new Mat();
                Imgproc.HoughCircles(grayImage, circles, Imgproc.CV_HOUGH_GRADIENT, 0.5,
                        minDist,150, 50, 10, 150);
                Imgproc.cvtColor(grayImage, grayImage, Imgproc.COLOR_GRAY2RGB);
                for (int i = 0; i < circles.rows(); i++)
                {
                    double[] circle = circles.get(0, i);
                    double centerX = circle[0],
                            centerY = circle[1],
                            radius = circle[2];
                    org.opencv.core.Point center = new org.opencv.core.Point(centerX,
                            centerY);
                    Imgproc.circle(grayImage, center, (int) radius, new Scalar(0,255,0),3);
                }
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        displayImage(grayImage);
                    }
                });
            }
        }).start();
    }

    private void displayImage(Mat image)
    {
        Bitmap bitmap = Bitmap.createBitmap(image.cols(),
                image.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(image, bitmap);
        displayImage(bitmap);
    }
    private void displayImage(Bitmap bitmap)
    {
        imageView.setImageBitmap(bitmap);
    }
    private void perspectiveTransform(){
        if(corners.size()<4){
            Toast.makeText(getApplicationContext(),
                    "It is necessary to choose 4 corners",
                    Toast.LENGTH_SHORT).show();
            return;
        }
        org.opencv.core.Point centroid=new org.opencv.core.Point(0,0);
        for(org.opencv.core.Point point:corners)
        {
            centroid.x+=point.x;
            centroid.y+=point.y;
        }
        centroid.x/=corners.size();
        centroid.y/=corners.size();

        sortCorners(corners,centroid);
        Mat correctedImage=new Mat(sampledImage.rows(),sampledImage.cols(),sampledImage.type());
        Mat srcPoints= Converters.vector_Point2f_to_Mat(corners);

        Mat destPoints=Converters.vector_Point2f_to_Mat(Arrays.asList(new org.opencv.core.Point[]{
                new org.opencv.core.Point(0, 0),
                new org.opencv.core.Point(correctedImage.cols(), 0),
                new org.opencv.core.Point(correctedImage.cols(),correctedImage.rows()),
                new org.opencv.core.Point(0,correctedImage.rows())}));

        Mat transformation=Imgproc.getPerspectiveTransform(srcPoints, destPoints);
        Imgproc.warpPerspective(sampledImage, correctedImage, transformation, correctedImage.size());

        corners.clear();
        displayImage(correctedImage);
    }
    void sortCorners(ArrayList<Point> corners, org.opencv.core.Point center)
    {
        ArrayList<org.opencv.core.Point> top=new ArrayList<org.opencv.core.Point>();
        ArrayList<org.opencv.core.Point> bottom=new ArrayList<org.opencv.core.Point>();

        for (int i = 0; i < corners.size(); i++)
        {
            if (corners.get(i).y < center.y)
                top.add(corners.get(i));
            else
                bottom.add(corners.get(i));
        }

        double topLeft=top.get(0).x;
        int topLeftIndex=0;
        for(int i=1;i<top.size();i++)
        {
            if(top.get(i).x<topLeft)
            {
                topLeft=top.get(i).x;
                topLeftIndex=i;
            }
        }

        double topRight=0;
        int topRightIndex=0;
        for(int i=0;i<top.size();i++)
        {
            if(top.get(i).x>topRight)
            {
                topRight=top.get(i).x;
                topRightIndex=i;
            }
        }

        double bottomLeft=bottom.get(0).x;
        int bottomLeftIndex=0;
        for(int i=1;i<bottom.size();i++)
        {
            if(bottom.get(i).x<bottomLeft)
            {
                bottomLeft=bottom.get(i).x;
                bottomLeftIndex=i;
            }
        }

        double bottomRight=bottom.get(0).x;
        int bottomRightIndex=0;
        for(int i=1;i<bottom.size();i++)
        {
            if(bottom.get(i).x>bottomRight)
            {
                bottomRight=bottom.get(i).x;
                bottomRightIndex=i;
            }
        }

        org.opencv.core.Point topLeftPoint = top.get(topLeftIndex);
        org.opencv.core.Point topRightPoint = top.get(topRightIndex);
        org.opencv.core.Point bottomLeftPoint = bottom.get(bottomLeftIndex);
        org.opencv.core.Point bottomRightPoint = bottom.get(bottomRightIndex);

        corners.clear();
        corners.add(topLeftPoint);
        corners.add(topRightPoint);
        corners.add(bottomRightPoint);
        corners.add(bottomLeftPoint);
    }
}