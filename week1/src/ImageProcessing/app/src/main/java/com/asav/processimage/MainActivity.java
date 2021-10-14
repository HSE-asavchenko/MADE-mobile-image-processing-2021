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
                    System.loadLibrary("OpenCvProcessImageLib");
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
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
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
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_openGallery:
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"Select Picture"),
                        SELECT_PICTURE);
                return true;

            case R.id.action_grayscale:
                if(isImageLoaded()) {
                    grayscale();
                }
                return true;
            case R.id.action_binary:
                if(isImageLoaded()) {
                    binary();
                }
                return true;
            case R.id.action_contrast:
                if(isImageLoaded()) {
                    contrast();
                }
                return true;
            case R.id.action_gamma:
                if(isImageLoaded()) {
                    gammaCorrection();
                }
                return true;
            case R.id.action_equalizehisto:
                if(isImageLoaded()) {
                    equalizeHisto();
                }
                return true;
            case R.id.action_blur:
                if(isImageLoaded()) {
                    blur();
                }
                return true;
            case R.id.action_fft:
                if(isImageLoaded()) {
                    fft();
                }
                return true;
            case R.id.action_fft_filter:
                if(isImageLoaded()) {
                    fftFilter();
                }
                return true;
            case R.id.action_addnoise:
                if(isImageLoaded()) {
                    addNoise();
                }
                return true;
            case R.id.action_median:
                if(isImageLoaded()) {
                    median();
                }
                return true;
            case R.id.action_bilateral:
                if(isImageLoaded()) {
                    bilateral();
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

    private void grayscale(){
        Mat grayImage=new Mat();
        Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);
        displayImage(grayImage);
    }
    private void binary(){
        Mat binImage = new Mat();
        if(false) {
            Mat grayImage = new Mat();
            Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
            Imgproc.GaussianBlur(grayImage,grayImage,new Size(5,5),0,0);

            //Imgproc.adaptiveThreshold(grayImage, binImage, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 12);
            //Imgproc.threshold(grayImage,binImage,128,255,Imgproc.THRESH_BINARY);
            Imgproc.threshold(grayImage,binImage,0,255,Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
        }
        else{
            OpenCVNativeCaller.niBlackThreshold(sampledImage.getNativeObjAddr(),binImage.getNativeObjAddr());
        }
        displayImage(binImage);
    }
    private final boolean useColor=true;
    private void contrast(){
        Mat grayImage=new Mat();
        Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);
        Mat out=new Mat();
        if(useColor){
            Mat HSV=new Mat();
            Imgproc.cvtColor(sampledImage, HSV, Imgproc.COLOR_RGB2HSV);
            ArrayList<Mat> hsv_list = new ArrayList(3);
            Core.split(HSV,hsv_list);

            for(int channel=1;channel<=2;++channel) {
                Core.MinMaxLocResult minMaxLocRes = Core.minMaxLoc(hsv_list.get(channel));
                double minVal = minMaxLocRes.minVal;//+20;
                double maxVal = minMaxLocRes.maxVal;//-50;
                Mat corrected = new Mat();
                hsv_list.get(channel).convertTo(corrected, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
                hsv_list.set(channel, corrected);
            }
            Core.merge(hsv_list,HSV);
            Imgproc.cvtColor(HSV, out, Imgproc.COLOR_HSV2RGB);
        }
        else {
            Core.MinMaxLocResult minMaxLocRes = Core.minMaxLoc(grayImage);
            double minVal = minMaxLocRes.minVal;//+20;
            double maxVal = minMaxLocRes.maxVal;//-50;
            grayImage.convertTo(out, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        }
        displayImage(out);
    }
    private void gammaCorrection(){
        double gammaValue = 1.3;
        Mat lookUpTable = new Mat(1, 256, CV_8U);
        byte[] lookUpTableData = new byte[(int) (lookUpTable.total() * lookUpTable.channels())];
        for (int i = 0; i < lookUpTable.cols(); i++) {
            lookUpTableData[i] = saturate(Math.pow(i / 255.0, gammaValue) * 255.0);
        }
        lookUpTable.put(0, 0, lookUpTableData);

        Mat out=new Mat();
        if(useColor){
            Mat HSV=new Mat();
            Imgproc.cvtColor(sampledImage, HSV, Imgproc.COLOR_RGB2HSV);
            ArrayList<Mat> hsv_list = new ArrayList(3);
            Core.split(HSV,hsv_list);

            for(int channel=1;channel<=2;++channel) {
                Mat corrected = new Mat();
                Core.LUT(hsv_list.get(channel), lookUpTable, corrected);
                hsv_list.set(channel, corrected);
            }
            Core.merge(hsv_list,HSV);
            Imgproc.cvtColor(HSV, out, Imgproc.COLOR_HSV2RGB);
        }
        else {
            Mat grayImage = new Mat();
            Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);

            Core.LUT(grayImage, lookUpTable, out);
        }
        displayImage(out);
    }
    private byte saturate(double val) {
        int iVal = (int) Math.round(val);
        iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
        return (byte) iVal;
    }
    private void equalizeHisto(){
        Mat out=new Mat();
        if(useColor){
            Mat HSV=new Mat();
            Imgproc.cvtColor(sampledImage, HSV, Imgproc.COLOR_RGB2HSV);
            ArrayList<Mat> hsv_list = new ArrayList(3);
            Core.split(HSV,hsv_list);
            for(int channel=1;channel<=2;++channel) {
                Mat equalizedValue = new Mat();
                Imgproc.equalizeHist(hsv_list.get(channel), equalizedValue);
                hsv_list.set(channel, equalizedValue);
            }
            Core.merge(hsv_list,HSV);
            Imgproc.cvtColor(HSV, out, Imgproc.COLOR_HSV2RGB);
        }
        else {
            Imgproc.cvtColor(sampledImage, out, Imgproc.COLOR_RGB2GRAY);
            Imgproc.equalizeHist(out, out);
        }
        displayImage(out);
    }
    private void blur(){
        Mat out=new Mat();
        //Imgproc.cvtColor(sampledImage,out, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(sampledImage,out,new Size(7,7),0,0);
        displayImage(out);
    }

    //https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
    private void fft(){
        Mat grayImage = new Mat();
        Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
        grayImage.convertTo(grayImage, CvType.CV_64FC1);

        int m = Core.getOptimalDFTSize(grayImage.rows());
        int n = Core.getOptimalDFTSize(grayImage.cols()); // on the border

        Mat padded = new Mat(new Size(n, m), CvType.CV_64FC1); // expand input

        Core.copyMakeBorder(grayImage, padded, 0, m - grayImage.rows(), 0,
                n - grayImage.cols(), Core.BORDER_CONSTANT);

        List<Mat> planes = new ArrayList<Mat>();
        planes.add(padded);
        planes.add(Mat.zeros(padded.rows(), padded.cols(), CvType.CV_64FC1));
        Mat complexI = new Mat();
        Core.merge(planes, complexI); // Add to the expanded another plane with zeros
        Mat complexI2=new Mat();
        Core.dft(complexI, complexI2); // this way the result may fit in the source matrix

        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        Core.split(complexI2, planes); // planes[0] = Re(DFT(I), planes[1] =Im(DFT(I))
        Mat spectrum = new Mat();
        if(true) {
            Core.magnitude(planes.get(0), planes.get(1), spectrum);
            Core.add(spectrum, new Scalar(1), spectrum);
            Core.log(spectrum, spectrum);
        }
        else{
            Core.phase(planes.get(1), planes.get(0), spectrum);
        }

        Mat crop = new Mat(spectrum, new Rect(0, 0, spectrum.cols() & -2,
                spectrum.rows() & -2));

        Mat out = crop.clone();

        // rearrange the quadrants of Fourier image so that the origin is at the
        // image center
        int cx = out.cols() / 2;
        int cy = out.rows() / 2;

        Rect q0Rect = new Rect(0, 0, cx, cy);
        Rect q1Rect = new Rect(cx, 0, cx, cy);
        Rect q2Rect = new Rect(0, cy, cx, cy);
        Rect q3Rect = new Rect(cx, cy, cx, cy);

        Mat q0 = new Mat(out, q0Rect); // Top-Left - Create a ROI per quadrant
        Mat q1 = new Mat(out, q1Rect); // Top-Right
        Mat q2 = new Mat(out, q2Rect); // Bottom-Left
        Mat q3 = new Mat(out, q3Rect); // Bottom-Right

        Mat tmp = new Mat(); // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);

        Core.normalize(out, out, 0, 255, Core.NORM_MINMAX);
        out.convertTo(out, CvType.CV_8UC1);
        displayImage(out);
    }
    //https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
    private void fftFilter(){
        Mat grayImage = new Mat();
        Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
        grayImage.convertTo(grayImage, CvType.CV_64FC1);

        int m = Core.getOptimalDFTSize(grayImage.rows());
        int n = Core.getOptimalDFTSize(grayImage.cols()); // on the border

        Mat padded = new Mat(new Size(n, m), CvType.CV_64FC1); // expand input

        Core.copyMakeBorder(grayImage, padded, 0, m - grayImage.rows(), 0,
                n - grayImage.cols(), Core.BORDER_CONSTANT);

        List<Mat> planes = new ArrayList<Mat>();
        planes.add(padded);
        planes.add(Mat.zeros(padded.rows(), padded.cols(), CvType.CV_64FC1));
        Mat complexI = new Mat();
        Core.merge(planes, complexI); // Add to the expanded another plane with zeros
        Mat complexI2=new Mat();
        Core.dft(complexI, complexI2); // this way the result may fit in the source matrix

        //Mat mask=Mat.zeros(padded.rows(), padded.cols(), CvType.CV_64FC2);
        if(false){
            int cropSizeX=8;
            int cropSizeY=8;
            Mat mask=Mat.zeros(complexI2.size(),CV_8U);
            Mat crop = new Mat(mask, new Rect(cropSizeX, cropSizeY, complexI2.cols() - 2 * cropSizeX, complexI2.rows() - 2 * cropSizeY));
            crop.setTo(new Scalar(1));
            Mat tmp=new Mat();
            complexI2.copyTo(tmp, mask);
            complexI2=tmp;
        }
        else {
            int cropSizeX=complexI2.cols()/32;
            int cropSizeY=complexI2.rows()/32;
            Mat crop = new Mat(complexI2, new Rect(cropSizeX, cropSizeY, complexI2.cols() - 2 * cropSizeX, complexI2.rows() - 2 * cropSizeY));
            crop.setTo(new Scalar(0, 0));
        }
        Mat complexII=new Mat();
        Core.idft(complexI2, complexII,DFT_SCALE);
        Core.split(complexII, planes);
        Mat out = planes.get(0);

        Core.normalize(out, out, 0, 255, Core.NORM_MINMAX);
        out.convertTo(out, CvType.CV_8UC1);
        displayImage(out);
    }

    private Mat getNoisyImage(boolean add_noise){
        Mat noisyImage;
        if(add_noise) {
            Mat noise = new Mat(sampledImage.size(), sampledImage.type());
            MatOfDouble mean = new MatOfDouble ();
            MatOfDouble dev = new MatOfDouble ();
            Core.meanStdDev(sampledImage,mean,dev);
            Core.randn(noise,0, 1*dev.get(0,0)[0]);
            noisyImage = new Mat();
            Core.add(sampledImage, noise, noisyImage);
        }
        else{
            noisyImage=sampledImage;
        }
        return noisyImage;
    }
    private void addNoise(){
        Mat noisyImage=getNoisyImage(true);
        displayImage(noisyImage);
    }
    private void median(){
        Mat noisyImage=getNoisyImage(true);
        Mat blurredImage=new Mat();
        Imgproc.medianBlur(noisyImage,blurredImage, 7);
        displayImage(blurredImage);
    }
    private void bilateral(){
        Mat noisyImage=getNoisyImage(false);
        Mat outImage=new Mat();
        Mat rgb=new Mat();
        Imgproc.cvtColor(noisyImage, rgb, Imgproc.COLOR_RGBA2RGB);
        Imgproc.bilateralFilter(rgb,outImage,9,75,75);
        displayImage(outImage);
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
 }