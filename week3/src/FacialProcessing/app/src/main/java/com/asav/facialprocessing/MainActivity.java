package com.asav.facialprocessing;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.*;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.*;

import com.asav.facialprocessing.mtcnn.Box;
import com.asav.facialprocessing.mtcnn.MTCNNModel;

import org.opencv.android.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.*;
import java.util.*;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private ImageView imageView;
    private Mat sampledImage=null;
    private static int minFaceSize=40;
    private CascadeClassifier faceCascadeClassifier =null, eyesCascadeClassifier =null;
    private MTCNNModel mtcnnFaceDetector=null;
    private AgeGenderEthnicityTfLiteClassifier facialAttributeClassifier=null;
    private EmotionTfLiteClassifier emotionClassifierTfLite =null;
    private EmotionPyTorchClassifier emotionClassifierPyTorch = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        androidx.appcompat.widget.Toolbar toolbar = (androidx.appcompat.widget.Toolbar) findViewById(R.id.main_toolbar);
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

    private CascadeClassifier loadCascade(int resourceId){
        CascadeClassifier res=null;
        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(resourceId);
            File tmpDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(tmpDir, "cascade.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            res = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (res.empty()) {
                Toast.makeText(getApplicationContext(),
                        "Failed to load cascade classifier "+resourceId,
                        Toast.LENGTH_SHORT).show();
                res = null;
            } else
                Toast.makeText(getApplicationContext(),
                        "Cascade loaded",
                        Toast.LENGTH_SHORT).show();
            cascadeFile.delete();
            tmpDir.delete();
        } catch (IOException e) {
            Log.e(TAG, "Failed to load models. Exception thrown: " + e + " " + Log.getStackTraceString(e));
        }
        return res;
    }
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    if(false) {
                        faceCascadeClassifier = loadCascade(R.raw.lbpcascade_frontalface_improved);
                    }
                    else {
                        faceCascadeClassifier = loadCascade(R.raw.haarcascade_frontalface_alt);
                        //eyesCascadeClassifier = loadCascade(R.raw.haarcascade_eye_tree_eyeglasses);
                    }
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
            mtcnnFaceDetector =MTCNNModel.Companion.create(getAssets());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing MTCNNModel!"+e);
        }
        try {
            facialAttributeClassifier=new AgeGenderEthnicityTfLiteClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing AgeGenderEthnicityTfLiteClassifier!", e);
        }

        try {
            emotionClassifierTfLite =new EmotionTfLiteClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing EmotionTfLiteClassifier!", e);
        }
        try {
            emotionClassifierPyTorch =new EmotionPyTorchClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing EmotionPyTorchClassifier!", e);
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

            case R.id.action_detectface_opencv:
                if(isImageLoaded()) {
                    detectFaceOpenCV();
                }
                return true;
            case R.id.action_detectface_mtcnn:
                if(isImageLoaded()) {
                    mtcnnDetectionAndAttributesRecognition(null);
                }
                return true;
            case R.id.action_agegender:
                if(isImageLoaded()) {
                    mtcnnDetectionAndAttributesRecognition(facialAttributeClassifier);
                }
                return true;
            case R.id.action_emotion_tf:
                if(isImageLoaded()) {
                    mtcnnDetectionAndAttributesRecognition(emotionClassifierTfLite);
                }
                return true;
            case R.id.action_emotion_torch:
                if(isImageLoaded()) {
                    mtcnnDetectionAndEmotionPyTorchRecognition();
                }
                return true;
            case R.id.action_comparefaces:
                if(isImageLoaded()) {
                    openImageFile(SELECT_TEMPLATE_PICTURE_MATCH);
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
                matchFaces(sampledImage,imageToMatch);
            }
        }
    }
    private Mat convertToMat(Uri selectedImageUri)
    {
        Mat resImage=null;
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            Bitmap bmp= BitmapFactory.decodeStream(ims);
            Mat rgbImage=new Mat();
            Utils.bitmapToMat(bmp, rgbImage);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,1);
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
    private void displayImage(Mat image)
    {
        Bitmap bitmap = Bitmap.createBitmap(image.cols(),
                image.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(image, bitmap);
        imageView.setImageBitmap(bitmap);
    }


    private void detectFaceOpenCV(){
        MatOfRect faces = new MatOfRect();
        Mat resImage=sampledImage;
        /*double minSize=600.0;
        double scale=Math.min(sampledImage.cols(),sampledImage.rows())/minSize;
        if(scale>1.0) {
            resImage=new Mat();
            Imgproc.resize(sampledImage,resImage,new Size((int)(sampledImage.cols()/scale),(int)(sampledImage.rows()/scale)));
        }*/
        Mat gray=new Mat();
        Imgproc.cvtColor(resImage, gray, Imgproc.COLOR_BGR2GRAY);

        if (faceCascadeClassifier != null) {
            long startTime = SystemClock.uptimeMillis();
            faceCascadeClassifier.detectMultiScale(gray, faces, 1.1, 3, 0, new
                    Size(minFaceSize, minFaceSize), new Size());
            Log.i(TAG, "Timecost to run detectMultiScale: " + Long.toString(SystemClock.uptimeMillis() - startTime));
        }
        Rect[] facesArray = faces.toArray();
        for (Rect face : facesArray) {
            Mat faceROI = gray.submat(face);
            boolean drawFace=true;

            // -- In each face, detect eyes
            if(eyesCascadeClassifier!=null) {
                MatOfRect eyes = new MatOfRect();
                eyesCascadeClassifier.detectMultiScale(faceROI, eyes);
                //eyesCascadeClassifier.detectMultiScale(faceROI, eyes,1.1,2,0,new Size(5,5));

                List<Rect> listOfEyes = eyes.toList();
                if(listOfEyes.isEmpty())
                    drawFace=false;
                for (Rect eye : listOfEyes) {
                    Point eyeCenter = new Point(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                    int radius = (int) Math.round((eye.width + eye.height) * 0.25);
                    Imgproc.circle(resImage, eyeCenter, radius, new Scalar(0, 0, 255), 4);
                }
            }

            if(drawFace)
                Imgproc.rectangle(resImage, face.tl(), face.br(), new Scalar(255,0,0), 3);
        }
        displayImage(resImage);
    }
    private void mtcnnDetectionAndAttributesRecognition(TfLiteClassifier classifier){
        Bitmap bmp = Bitmap.createBitmap(sampledImage.cols(), sampledImage.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(sampledImage, bmp);

        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(tempBmp);
        Paint p = new Paint();
        p.setStyle(Paint.Style.STROKE);
        p.setAntiAlias(true);
        p.setFilterBitmap(true);
        p.setDither(true);
        p.setColor(Color.BLUE);
        p.setStrokeWidth(5);

        Paint p_text = new Paint();
        p_text.setColor(Color.WHITE);
        p_text.setStyle(Paint.Style.FILL);
        p_text.setColor(Color.GREEN);
        p_text.setTextSize(24);

        c.drawBitmap(bmp, 0, 0, null);

        for (Box box : bboxes) {

            p.setColor(Color.RED);
            android.graphics.Rect bbox = new android.graphics.Rect(Math.max(0,bmp.getWidth()*box.left() / resizedBitmap.getWidth()),
                    Math.max(0,bmp.getHeight()* box.top() / resizedBitmap.getHeight()),
                    bmp.getWidth()* box.right() / resizedBitmap.getWidth(),
                    bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
            );

            c.drawRect(bbox, p);

            if(classifier!=null && bbox.width()>0 && bbox.height()>0) {
                Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
                Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, classifier.getImageSizeX(), classifier.getImageSizeY(), false);
                ClassifierResult res = classifier.classifyFrame(resultBitmap);
                c.drawText(res.toString(), bbox.left, Math.max(0, bbox.top - 20), p_text);
                Log.i(TAG, res.toString());
            }
        }
        imageView.setImageBitmap(tempBmp);
    }
    private void mtcnnDetectionAndEmotionPyTorchRecognition(){
        Bitmap bmp = Bitmap.createBitmap(sampledImage.cols(), sampledImage.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(sampledImage, bmp);

        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(tempBmp);
        Paint p = new Paint();
        p.setStyle(Paint.Style.STROKE);
        p.setAntiAlias(true);
        p.setFilterBitmap(true);
        p.setDither(true);
        p.setColor(Color.BLUE);
        p.setStrokeWidth(5);

        Paint p_text = new Paint();
        p_text.setColor(Color.WHITE);
        p_text.setStyle(Paint.Style.FILL);
        p_text.setColor(Color.GREEN);
        p_text.setTextSize(24);

        c.drawBitmap(bmp, 0, 0, null);

        for (Box box : bboxes) {

            p.setColor(Color.RED);
            android.graphics.Rect bbox = new android.graphics.Rect(Math.max(0,bmp.getWidth()*box.left() / resizedBitmap.getWidth()),
                    Math.max(0,bmp.getHeight()* box.top() / resizedBitmap.getHeight()),
                    bmp.getWidth()* box.right() / resizedBitmap.getWidth(),
                    bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
            );

            c.drawRect(bbox, p);

            if(emotionClassifierPyTorch!=null && bbox.width()>0 && bbox.height()>0) {
                Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
                String res=emotionClassifierPyTorch.recognize(faceBitmap);
                c.drawText(res, bbox.left, Math.max(0, bbox.top - 20), p_text);
                Log.i(TAG, res);
            }
        }
        imageView.setImageBitmap(tempBmp);
    }

    private void matchFaces(Mat img1, Mat img2){
        Mat resImage = new Mat();
        if(img2.rows()!=img1.rows()){
            Imgproc.resize(img2,img2,img1.size());
        }
        List<Mat> src = Arrays.asList(img1, img2);
        Core.hconcat(src, resImage);
        List<FaceFeatures> features1=getFacesFeatures(img1);
        List<FaceFeatures> features2=getFacesFeatures(img2);
        for(FaceFeatures face1 : features1){
            double minDist=10000;
            FaceFeatures bestFace=null;
            for(FaceFeatures face2 : features2){
                double dist = 0;
                for (int i = 0; i < face1.features.length; ++i) {
                    dist += (face1.features[i] - face2.features[i]) * (face1.features[i] - face2.features[i]);
                }
                dist = Math.sqrt(dist);
                if(dist<minDist){
                    minDist=dist;
                    bestFace=face2;
                }
            }
            if(bestFace!=null && minDist<1){
                float x=bestFace.centerX;
                Imgproc.line(resImage,new Point(face1.centerX*img1.cols(),face1.centerY*img1.rows()),
                        new Point(img1.cols()+bestFace.centerX*img2.cols(),bestFace.centerY*img2.rows()),
                        new Scalar(255,0,0),5);
                Log.i(TAG,"distance "+minDist);
            }
        }
        displayImage(resImage);
    }
    private List<FaceFeatures> getFacesFeatures(Mat img){
        Bitmap bmp = Bitmap.createBitmap(img.cols(), img.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(img, bmp);

        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            bmp=resizedBitmap;
        }
        long startTime = SystemClock.uptimeMillis();
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
        Log.i(TAG, "Timecost to run mtcnn: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        List<FaceFeatures> facesInfo=new ArrayList<>();
        for (Box box : bboxes) {
            android.graphics.Rect bbox = new android.graphics.Rect(Math.max(0,bmp.getWidth()*box.left() / resizedBitmap.getWidth()),
                    Math.max(0,bmp.getHeight()* box.top() / resizedBitmap.getHeight()),
                    bmp.getWidth()* box.right() / resizedBitmap.getWidth(),
                    bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
            );
            Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
            Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, facialAttributeClassifier.getImageSizeX(), facialAttributeClassifier.getImageSizeY(), false);
            FaceData res=(FaceData)facialAttributeClassifier.classifyFrame(resultBitmap);
            facesInfo.add(new FaceFeatures(res.features,0.5f*(box.left()+box.right()) / resizedBitmap.getWidth(),0.5f*(box.top()+box.bottom()) / resizedBitmap.getHeight()));
        }
        return facesInfo;
    }
    private class FaceFeatures{
        public FaceFeatures(float[] feat, float x, float y){
            features=feat;
            centerX=x;
            centerY=y;
        }
        public float[] features;
        public float centerX,centerY;
    }

}