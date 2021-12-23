package com.asav.matching;

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
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private ImageView imageView;
    private Mat sampledImage=null;
    private TfLiteFeatureExtractor featureExtractor=null;
    private boolean useSuperpoint=false;

    private static native void extractPointsOfInterest(long matAddrIn, long matAddrOut);
    private static native void stitchImages(long matAddrIn1,long matAddrIn2, long matAddrOut);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        imageView=(ImageView)findViewById(R.id.inputImageView);

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
        try {
            featureExtractor=new TfLiteFeatureExtractor(getAssets());
        } catch (IOException e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
            featureExtractor=null;
            useSuperpoint=false;
        }
    }
    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
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
    private static final int SELECT_TEMPLATE_PICTURE_MATCH = 2;
    private static final int SELECT_PICTURE_STITCHING = 3;
    private void openImageFile(int requestCode){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent,"Select Picture"),requestCode);
    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_openGallery:
                openImageFile(SELECT_PICTURE);
                return true;

            case R.id.action_extractpoi:
                if(isImageLoaded()) {
                    extractFeatures();
                }
                return true;
            case R.id.action_matchimages:
                if(isImageLoaded()) {
                    openImageFile(SELECT_TEMPLATE_PICTURE_MATCH);
                }
                return true;
            case R.id.action_stitchimages:
                if(isImageLoaded()) {
                    openImageFile(SELECT_PICTURE_STITCHING);
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
        if(resultCode==RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Log.d(TAG, "uri" + selectedImageUri);
                sampledImage=convertToMat(selectedImageUri);
                if(sampledImage!=null)
                    displayImage(sampledImage);
            }
            else if(requestCode==SELECT_TEMPLATE_PICTURE_MATCH){
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Mat imageToMatch=convertToMat(selectedImageUri);
                matchImages(imageToMatch);
            }
            else if(requestCode==SELECT_PICTURE_STITCHING){
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Mat image2=convertToMat(selectedImageUri);
                Mat resImage;
                if(false){
                    resImage=new Mat();
                    long startTime = System.nanoTime();
                    stitchImages(sampledImage.getNativeObjAddr(),image2.getNativeObjAddr(),resImage.getNativeObjAddr());
                    long elapsedTime = System.nanoTime() - startTime;
                    elapsedTime = elapsedTime / 1000000; // Milliseconds (1:1000000)
                    Log.i(this.getClass().getSimpleName(), "OpenCV Stitching 2 images took " + elapsedTime + "ms");
                    if(resImage.rows()<=0 || resImage.cols()<=0) {
                        Toast.makeText(getApplicationContext(),
                                "Panorama not found",
                                Toast.LENGTH_SHORT).show();
                        return;
                    }
                }
                else
                    resImage=createPanorama(sampledImage,image2);
                displayImage(resImage);
            }
        }
    }
    private Mat convertToMat(Uri selectedImageUri)
    {
        Mat resImage=null;
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
            resImage=new Mat();
            Imgproc.resize(rgbImage, resImage, new
                    Size(),downSampleRatio,downSampleRatio,Imgproc.INTER_AREA);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
            resImage=null;
        }
        return resImage;
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

    private final boolean isBinaryDetector=false;
    private Feature2D getDetector(){
        Feature2D detector;
        if(isBinaryDetector){
            //detector =FastFeatureDetector.create(50);
            detector=ORB.create();
            //detector= BriefDescriptorExtractor.create(256);
        }
        else {
            //detector= HarrisLaplaceFeatureDetector.create(1,0.02f);
            detector = SIFT.create();
            //detector=BRISK.create();
        }
        return detector;
    }
    private void extractFeatures(){
        Mat resImage=sampledImage.clone();
        long startTime = SystemClock.uptimeMillis();
        if(false)
            extractPointsOfInterest(sampledImage.getNativeObjAddr(),resImage.getNativeObjAddr());
        else{
            Mat grayImage=new Mat();
            Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);
            MatOfKeyPoint keyPoints=new MatOfKeyPoint();
            if(useSuperpoint) {
                Mat descriptors=featureExtractor.processImage(grayImage,keyPoints);
            }
            else {
                Feature2D detector = getDetector();
                detector.detect(grayImage, keyPoints);
            }
            Features2d.drawKeypoints(sampledImage, keyPoints, resImage,new Scalar(0,255,0));
        }
        Log.i(TAG, "Timecost to extractPointsOfInterest: " + Long.toString(SystemClock.uptimeMillis() - startTime));
        displayImage(resImage);
    }
    private void matchImages(Mat imageToMatch){
        //https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
        long startTime = SystemClock.uptimeMillis();
        Mat imgScene=new Mat();
        Imgproc.cvtColor(sampledImage,imgScene, Imgproc.COLOR_RGB2GRAY);
        Mat imgObject=new Mat();
        Imgproc.cvtColor(imageToMatch,imgObject, Imgproc.COLOR_RGB2GRAY);

        Feature2D detector=getDetector();
        MatOfKeyPoint keypointsScene=new MatOfKeyPoint();

        Mat descriptors;
        descriptors=new Mat();
        detector.detectAndCompute(imgScene, new Mat(),keypointsScene,descriptors);

        MatOfKeyPoint keypointsObject=new MatOfKeyPoint();
        Mat descriptorsToMatch;
        descriptorsToMatch=new Mat();
        detector.detectAndCompute(imgObject, new Mat(),keypointsObject,descriptorsToMatch);

        DescriptorMatcher matcher =null;
        if(isBinaryDetector){
            matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        }
        else{
            matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        }
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptorsToMatch,descriptors, knnMatches, 2);

        ArrayList<DMatch> listOfGoodMatches =new ArrayList<DMatch>();
        //-- Filter matches using the Lowe's ratio test
        float ratioThresh = 0.75f;
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }
        MatOfDMatch goodEnough=new MatOfDMatch();
        goodEnough.fromList(listOfGoodMatches );
        Mat imgMatches=new Mat();
        Features2d.drawMatches(imgObject, keypointsObject, imgScene,
                keypointsScene, goodEnough, imgMatches,Scalar.all(-1),Scalar.all(-1),new
                        MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

        //-- Localize the object
        List<Point> obj = new ArrayList<>();
        List<Point> scene = new ArrayList<>();
        List<KeyPoint> listOfKeypointsObject = keypointsObject.toList();
        List<KeyPoint> listOfKeypointsScene = keypointsScene.toList();
        for (int i = 0; i < listOfGoodMatches.size(); i++) {
            //-- Get the keypoints from the good matches
            obj.add(listOfKeypointsObject.get(listOfGoodMatches.get(i).queryIdx).pt);
            scene.add(listOfKeypointsScene.get(listOfGoodMatches.get(i).trainIdx).pt);
        }
        MatOfPoint2f objMat = new MatOfPoint2f(), sceneMat = new MatOfPoint2f();
        objMat.fromList(obj);
        sceneMat.fromList(scene);
        double ransacReprojThreshold = 3.0;
        Mat H = Calib3d.findHomography(objMat, sceneMat, Calib3d.RANSAC, ransacReprojThreshold);
        //-- Get the corners from the image_1 ( the object to be "detected" )
        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2), sceneCorners = new Mat();
        float[] objCornersData = new float[(int) (objCorners.total() * objCorners.channels())];
        objCorners.get(0, 0, objCornersData);
        objCornersData[0] = 0;
        objCornersData[1] = 0;
        objCornersData[2] = imgObject.cols();
        objCornersData[3] = 0;
        objCornersData[4] = imgObject.cols();
        objCornersData[5] = imgObject.rows();
        objCornersData[6] = 0;
        objCornersData[7] = imgObject.rows();
        objCorners.put(0, 0, objCornersData);
        Core.perspectiveTransform(objCorners, sceneCorners, H);
        Log.i(TAG, "Timecost to matchImages: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        float[] sceneCornersData = new float[(int) (sceneCorners.total() * sceneCorners.channels())];
        sceneCorners.get(0, 0, sceneCornersData);
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        Imgproc.line(imgMatches, new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]),
                new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]),
                new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]),
                new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]),
                new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]), new Scalar(0, 255, 0), 4);
        displayImage(imgMatches);
    }
    private Mat createPanorama(Mat...arg0) {
        // Base code extracted from: https://stackoverflow.com/questions/49357079/trying-to-get-opencv-to-work-in-java-sample-code-to-stitch-2-photos-together
        if (arg0.length != 2) {
            return null;
        }
        long startTime = System.nanoTime();
        // Convert the two bitmaps to OpenCV mats...
        Mat img1 = arg0[0];
        Mat img2 = arg0[1];

        Mat gray_image1 = new Mat();
        Mat gray_image2 = new Mat();
        Imgproc.cvtColor(img1, gray_image1, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(img2, gray_image2, Imgproc.COLOR_RGB2GRAY);

        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        Feature2D detector=getDetector();
        detector.detectAndCompute(gray_image1, new Mat(),keyPoints1, descriptors1);
        detector.detectAndCompute(gray_image2, new Mat(),keyPoints2, descriptors2);

        MatOfDMatch matches = new MatOfDMatch();
        DescriptorMatcher matcher =null;
        if(isBinaryDetector){
            matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        }
        else{
            matcher=DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        }
        matcher.match(descriptors1, descriptors2, matches);

        double max_dist = 0; double min_dist = 100;
        List<DMatch> listMatches = matches.toList();

        for( int i = 0; i < listMatches.size(); i++ ) {
            double dist = listMatches.get(i).distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // Reduce the list of matching keypoints to a list of good matches...
        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
        MatOfDMatch goodMatches = new MatOfDMatch();
        for(int i = 0; i < listMatches.size(); i++) {
            if(listMatches.get(i).distance < 2*min_dist) {
                good_matches.addLast(listMatches.get(i));
            }
        }
        goodMatches.fromList(good_matches);

        // Calculate the homograohy between the two images...
        LinkedList<Point> imgPoints1List = new LinkedList<Point>();
        LinkedList<Point> imgPoints2List = new LinkedList<Point>();
        List<KeyPoint> keypoints1List = keyPoints1.toList();
        List<KeyPoint> keypoints2List = keyPoints2.toList();

        for(int i = 0; i<good_matches.size(); i++) {
            imgPoints1List.addLast(keypoints1List.get(good_matches.get(i).queryIdx).pt);
            imgPoints2List.addLast(keypoints2List.get(good_matches.get(i).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(imgPoints1List);
        MatOfPoint2f scene = new MatOfPoint2f();
        scene.fromList(imgPoints2List);

        Mat H = Calib3d.findHomography(obj, scene, Calib3d.RANSAC,3);

        int imageWidth = img2.cols();
        int imageHeight = img2.rows();

        // To avoid missing some of the possible stitching scenarios, we offset the homography to the middle of a mat which has three time the size of one of the pictures.
        // Extracted from this: https://stackoverflow.com/questions/21618044/stitching-2-images-opencv
        Mat Offset = new Mat(3, 3, H.type());
        Offset.put(0,0, new double[]{1});
        Offset.put(0,1, new double[]{0});
        Offset.put(0,2, new double[]{imageWidth});
        Offset.put(1,0, new double[]{0});
        Offset.put(1,1, new double[]{1});
        Offset.put(1,2, new double[]{imageHeight});
        Offset.put(2,0, new double[]{0});
        Offset.put(2,1, new double[]{0});
        Offset.put(2,2, new double[]{1});

        // Multiply the homography mat with the offset.
        Core.gemm(Offset, H, 1, new Mat(), 0, H);

        Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
        Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

        obj_corners.put(0,0, new double[]{0,0});
        obj_corners.put(0,0, new double[]{imageWidth,0});
        obj_corners.put(0,0,new double[]{imageWidth,imageHeight});
        obj_corners.put(0,0,new double[]{0,imageHeight});

        Core.perspectiveTransform(obj_corners, scene_corners, H);

        // The resulting mat will be three times the size (width and height) of one of the source images. (We assume, that both images have the same size.
        Size s = new Size(imageWidth *3,imageHeight*3);
        Mat img_matches = new Mat(new Size(img1.cols()+img2.cols(),img1.rows()), CvType.CV_32FC2);

        // Perform the perspective warp of img1 with the given homography and place it on the large result mat.
        Imgproc.warpPerspective(img1, img_matches, H, s);

        // Create another mat which is used to hold the second image and place it in the middle of the large sized result mat.
        int m_xPos = (int)(img_matches.size().width/2 - img2.size().width/2);
        int m_yPos = (int)(img_matches.size().height/2 - img2.size().height/2);
        Mat m = new Mat(img_matches,new Rect(m_xPos, m_yPos, img2.cols(), img2.rows()));
        // Copy img2 to the mat in the middle of the large result mat
        img2.copyTo(m);

        long elapsedTime = System.nanoTime() - startTime;
        elapsedTime = elapsedTime / 1000000; // Milliseconds (1:1000000)
        Log.i(this.getClass().getSimpleName(), "Stitching 2 images took " + elapsedTime + "ms");

        // The resulting mat is way to big. It holds a lot of empty "transparent" space.
        // We will not crop the image, so that only the "region of interest" remains.
        Mat gray_out=new Mat();
        Imgproc.cvtColor(img_matches,gray_out,Imgproc.COLOR_BGR2GRAY);
        Mat mThreshold=new Mat();
        Imgproc.threshold(gray_out,mThreshold,1,255,Imgproc.THRESH_BINARY);
        Mat Points=new Mat();
        Core.findNonZero(mThreshold,Points);
        Rect imageBoundingBox3=Imgproc.boundingRect(Points);

        Mat regionOfInterest = img_matches.submat(imageBoundingBox3);

        return regionOfInterest;
    }
    private void displayImage(Mat image)
    {
        Bitmap bitmap = Bitmap.createBitmap(image.cols(),
                image.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(image, bitmap);
        displayImage(bitmap);
    }
    private void displayImage(Bitmap bitmap) {
        imageView.setImageBitmap(bitmap);
    }
}