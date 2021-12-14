package com.asav.styletransfer;

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
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.*;


import java.io.*;
import java.util.*;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private ImageView imageView;
    private Bitmap sampledBitmap=null;
    private TfLiteStyleTransfer styleTransfer=null;
    private MagentaStyleTransfer magentaStyleTransfer=null;

    private Spinner stylesSpinner;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView=(ImageView)findViewById(R.id.inputImageView);
        stylesSpinner =findViewById(R.id.styles_spinner);
        String styles[]=new String[MagentaStyleTransfer.NUM_STYLES];
        for(int i=0;i<MagentaStyleTransfer.NUM_STYLES;++i){
            styles[i]=String.valueOf(i+1);
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,R.layout.spinner_item, styles);
        stylesSpinner.setAdapter(adapter);
        stylesSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            public void onItemSelected(AdapterView<?> parent,
                                       View itemSelected, int selectedItemPosition, long selectedId) {
                if (sampledBitmap != null) {
                    stylizeImage();
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
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

    private void init(){
        try {
            styleTransfer=new TfLiteStyleTransfer(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing TfLiteStyleTransfer!", e);
        }
        magentaStyleTransfer=new MagentaStyleTransfer(getAssets());
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
    private static final int SELECT_STYLE_IMAGE = 2;
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

            case R.id.action_stylize:
                if(isImageLoaded()) {
                    stylizeImage();
                }
                return true;

            case R.id.action_styleimage:
                if(isImageLoaded()) {
                    openImageFile(SELECT_STYLE_IMAGE);
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
        if(resultCode==RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Log.d(TAG, "uri" + selectedImageUri);
                sampledBitmap= processImage(selectedImageUri);
                imageView.setImageBitmap(sampledBitmap);
            }
            else if (requestCode==SELECT_STYLE_IMAGE){
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Log.d(TAG, "uri" + selectedImageUri);
                Bitmap styleBitmap= processImage(selectedImageUri);
                //imageView.setImageBitmap(sampledBitmap);
                applyStyleImageToContent(styleBitmap);
            }
        }
    }
    private Bitmap processImage(Uri selectedImageUri)
    {
        Bitmap bmp=null;
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            bmp=BitmapFactory.decodeStream(ims);
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

            imageView.setImageBitmap(bmp);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
        }
        return bmp;
    }
    private void stylizeImage() {
        Bitmap bitmap = magentaStyleTransfer.stylizeImage(sampledBitmap,stylesSpinner.getSelectedItemPosition());
        imageView.setImageBitmap(bitmap);
    }
    private void applyStyleImageToContent(Bitmap styleBitmap){
        Bitmap res=styleTransfer.execute(sampledBitmap,styleBitmap);
        imageView.setImageBitmap(res);
    }
}