package com.asav.android;


import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.exifinterface.media.ExifInterface;

import android.view.Menu;
import android.view.MenuItem;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * Created by avsavchenko.
 */

public class MainActivity extends AppCompatActivity {

    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

    private Bitmap sampledBitmap=null;
    private ObjectDetector objectDetectorTfLiteSSD;
    private ObjectDetector objectDetectorTfLiteYolo;
    private ObjectDetector objectDetectorTfMobile;
    private ObjectDetector objectDetectorTorch;
    private TfLiteImageSegmentation imageSegmentationModel;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        Toolbar toolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        imageView=(ImageView)findViewById(R.id.photoView);

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

    private void init(){
        try {
            objectDetectorTfLiteSSD = new TfLiteObjectDetection(getAssets());
        } catch (IOException e) {
            Log.e(TAG, "Failed to load TfLiteObjectDetection.", e);
        }
        try {
            objectDetectorTfMobile = new TfObjectDetectionModel(getAssets());
        } catch (IOException e) {
            Log.e(TAG, "Failed to load TfObjectDetectionModel.", e);
        }
        try {
            objectDetectorTfLiteYolo = new TfLiteYoloObjectDetection(getAssets());
        } catch (IOException e) {
            Log.e(TAG, "Failed to load TfLiteYoloObjectDetection.", e);
        }
        try {
            objectDetectorTorch = new TorchObjectDetection(this);
        } catch (IOException e) {
            Log.e(TAG, "Failed to load TorchObjectDetection.", e);
        }

        try {
            imageSegmentationModel=new TfLiteImageSegmentation(this);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load ImageSegmentationModelExecutor.", e);
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

            case R.id.action_detectLite:
                if(isImageLoaded()) {
                    objectDetection(objectDetectorTfLiteSSD);
                }
                return true;

            case R.id.action_detectMobile:
                if(isImageLoaded()) {
                    objectDetection(objectDetectorTfMobile);
                }
                return true;

            case R.id.action_detectLiteYolo:
                if(isImageLoaded()) {
                    objectDetection(objectDetectorTfLiteYolo);
                }
                return true;

            case R.id.action_detectTorch:
                if(isImageLoaded()) {
                    objectDetection(objectDetectorTorch);
                }
                return true;

            case R.id.action_segmentation:
                if(isImageLoaded()) {
                    semanticSegmentation();
                }
                return true;

            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                return super.onOptionsItemSelected(item);
        }
    }
    private boolean isImageLoaded(){
        if(sampledBitmap==null)
            Toast.makeText(getApplicationContext(),
                    "It is necessary to open image firstly",
                    Toast.LENGTH_SHORT).show();
        return sampledBitmap!=null;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == SELECT_PICTURE && resultCode == RESULT_OK) {
            Uri selectedImageUri = data.getData(); //The uri with the location of the file
            Log.d(TAG,"uri"+selectedImageUri);
            //imageView.setImageURI(selectedImageUri);
            /*String path=getPath1(selectedImageUri);
            Log.d(TAG,"path "+path);*/

            processImage(selectedImageUri);
        }
    }
    private List<DetectorData> detectObjects(ObjectDetector objectDetector, Bitmap bmp) {
        long startTime = SystemClock.uptimeMillis();
        List<DetectorData> detectionResults=objectDetector.detectObjects(bmp);
        long timeCost = SystemClock.uptimeMillis() - startTime;
        Log.i(TAG, "Timecost to run object detection model inference: " + Long.toString(timeCost));
        return detectionResults;
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
            int status=ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
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

    private void processImage(Uri selectedImageUri)
    {
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            Bitmap bmp= BitmapFactory.decodeStream(ims);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,1);
            int degreesForRotation=0;
            switch (orientation)
            {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degreesForRotation=90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degreesForRotation=270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degreesForRotation=180;
                    break;
            }
            if(degreesForRotation!=0) {
                Matrix matrix = new Matrix();
                matrix.setRotate(degreesForRotation);
                bmp=Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(),
                        bmp.getHeight(), matrix, true);
            }
            sampledBitmap=bmp;
            imageView.setImageBitmap(sampledBitmap);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: ",e);
        }
    }

    private void objectDetection(ObjectDetector objectDetector)
    {
        try {
            Bitmap bmp=objectDetector.resizeBitmap(sampledBitmap);
            List<DetectorData> detectionResults=detectObjects(objectDetector,bmp);
            Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
            Canvas c = new Canvas(tempBmp);
            Paint p = new Paint();
            p.setStyle(Paint.Style.STROKE);
            p.setAntiAlias(true);
            p.setFilterBitmap(true);
            p.setDither(true);
            p.setColor(Color.BLUE);
            p.setStrokeWidth(2);

            Paint p_text = new Paint();
            p_text.setColor(Color.WHITE);
            p_text.setStyle(Paint.Style.FILL);
            p_text.setColor(Color.BLUE);
            p_text.setTextSize(20);

            c.drawBitmap(bmp, 0, 0, null);
            for(DetectorData d : detectionResults){
                p.setColor(Color.RED);
                android.graphics.Rect bbox = new android.graphics.Rect((int)(bmp.getWidth()*d.location.left),
                        (int)(bmp.getHeight()* d.location.top),
                        (int)(bmp.getWidth()* d.location.right),
                        (int)(bmp.getHeight() * d.location.bottom)
                );
                c.drawRect(bbox, p);
                c.drawText(d.toString(), bbox.left, Math.max(20, bbox.top - 20), p_text);
                Log.i(TAG, d.toString());
            }

            imageView.setImageBitmap(tempBmp);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: ",e);
        }
    }

    private void semanticSegmentation(){
        Bitmap res=imageSegmentationModel.execute(sampledBitmap);
        imageView.setImageBitmap(res);
    }
}
