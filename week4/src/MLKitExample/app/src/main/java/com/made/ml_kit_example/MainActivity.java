package com.made.ml_kit_example;

import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.*;
import java.util.*;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.barcode.Barcode;
import com.google.mlkit.vision.barcode.BarcodeScanner;
import com.google.mlkit.vision.barcode.BarcodeScanning;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.label.*;
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions;
import com.google.mlkit.vision.objects.*;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;
import com.google.mlkit.vision.text.*;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;
import com.googlecode.tesseract.android.TessBaseAPI;

public class MainActivity extends AppCompatActivity {

  private static String TAG="MainActivity";
  private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

  private Random rnd=new Random();

  private ImageView imageView=null;
  private TextView textView;

  private TextRecognizer recognizer=null;
  private ImageLabeler labeler = null;
  private ObjectDetector objectDetector = null;
  private BarcodeScanner barCodeScanner=null;
  private InputImage image = null;

  private TessBaseAPI tessBaseApi;
  private static final String lang = "rus";
  private String DATA_PATH="";
  private static final String TESSDATA = "tessdata";

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    //labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS);
    ImageLabelerOptions labelOptions = new ImageLabelerOptions.Builder()
         .setConfidenceThreshold(0.5f)
         .build();
    labeler = ImageLabeling.getClient(labelOptions);

    ObjectDetectorOptions detectorOptions =
            new ObjectDetectorOptions.Builder()
                    .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                    .enableMultipleObjects()
                    .enableClassification()  // Optional
                    .build();
    objectDetector = ObjectDetection.getClient(detectorOptions);

    recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

    barCodeScanner = BarcodeScanning.getClient();

    imageView = findViewById(R.id.image);
    textView = findViewById(R.id.text);
    textView.setMovementMethod(new ScrollingMovementMethod());

  Toolbar toolbar = (Toolbar) findViewById(R.id.my_toolbar);
    //setSupportActionBar(toolbar);
    if (!allPermissionsGranted()) {
      ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
    }
    else
      initTesseract();
  }

  private void copyTessDataFiles(String path) throws IOException{
    String fileList[] = getAssets().list(path);

    for (String fileName : fileList) {
      if(!fileName.endsWith(".traineddata"))
        continue;
      // open file within the assets folder
      // if it is not already there copy it to the sdcard
      String pathToDataFile = DATA_PATH + path + "/" + fileName;
      if (!(new File(pathToDataFile)).exists()) {

        InputStream in = getAssets().open(path + "/" + fileName);

        OutputStream out = new FileOutputStream(pathToDataFile);

        // Transfer bytes from in to out
        byte[] buf = new byte[1024];
        int len;

        while ((len = in.read(buf)) > 0) {
          out.write(buf, 0, len);
        }
        in.close();
        out.close();

        Log.d(TAG, "Copied " + fileName + "to tessdata");
      }
    }
  }
  private void initTesseract() {
    try{
      DATA_PATH = getFilesDir().toString() + "/TesseractSample/";
      String path=DATA_PATH + TESSDATA;
      File dir = new File(path);
      if (!dir.exists()) {
        if (!dir.mkdirs()) {
          Log.e(TAG, "ERROR: Creation of directory " + path + " failed, check does Android Manifest have permission to write to external storage.");
          return;
        }
      } else {
        Log.i(TAG, "Created directory " + path);
      }
      copyTessDataFiles(TESSDATA);
      tessBaseApi = new TessBaseAPI();
      tessBaseApi.init(DATA_PATH, lang);
    } catch (IOException e) {
      Log.e(TAG, "Unable to copy files to tessdata " + e+" "+Log.getStackTraceString(e));
      tessBaseApi=null;
    }
  }

  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    getMenuInflater().inflate(R.menu.toolbar_menu, menu);
    return true;
  }
  private static final int SELECT_PICTURE = 1;
  @Override
  public boolean onOptionsItemSelected(MenuItem item) {
    textView.setText("");
    switch (item.getItemId()) {
      case R.id.action_openGallery:
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent,"Select Picture"),
                SELECT_PICTURE);
        return true;
      case R.id.action_labelImage:
        if(image!=null)
          labelImage(image);
        else
          Toast.makeText(MainActivity.this, "Please, open image first", Toast.LENGTH_SHORT).show();
        return true;
      case R.id.action_detectObjects:
        if(image!=null)
          detectObjects(image);
        else
          Toast.makeText(MainActivity.this, "Please, open image first", Toast.LENGTH_SHORT).show();
        return true;
      case R.id.action_detectText:
        if(image!=null)
          recognizeText(image);
        else
          Toast.makeText(MainActivity.this, "Please, open image first", Toast.LENGTH_SHORT).show();
        return true;
      case R.id.action_detectTextTesseract:
        if(image!=null)
          recognizeTextTesseract();
        else
          Toast.makeText(MainActivity.this, "Please, open image first", Toast.LENGTH_SHORT).show();
        return true;
      case R.id.action_barCode:
        if(image!=null)
          processBarCode(image);
        else
          Toast.makeText(MainActivity.this, "Please, open image first", Toast.LENGTH_SHORT).show();
        return true;
      default:
        // If we got here, the user's action was not recognized.
        // Invoke the superclass to handle it.
        return super.onOptionsItemSelected(item);
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
          initTesseract();
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

  private void processImage(Uri selectedImageUri)
  {
    image = null;
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

      imageView.setImageBitmap(bmp);
      image = InputImage.fromBitmap(bmp, 0);
    } catch (Exception e) {
      Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
    }
  }

  private void recognizeTextTesseract(){
    BitmapDrawable drawable=(BitmapDrawable)imageView.getDrawable();
    if(drawable!=null) {
      Bitmap bmp = drawable.getBitmap();
      tessBaseApi.setImage(bmp);
      String extractedText = "";
      try {
        extractedText = tessBaseApi.getUTF8Text();
      } catch (Exception e) {
        Log.e(TAG, "Error in recognizing text.");
      }
      textView.setText(extractedText);
    }
  }
  private void recognizeText(InputImage image){
    Task<Text> result =
            recognizer.process(image)
                    .addOnSuccessListener(new OnSuccessListener<Text>() {
                      @Override
                      public void onSuccess(Text visionText) {
                        String resultText = visionText.getText();
                        /*for (Text.TextBlock block : visionText.getTextBlocks()) {
                          String blockText = block.getText();
                          Point[] blockCornerPoints = block.getCornerPoints();
                          Rect blockFrame = block.getBoundingBox();
                          for (Text.Line line : block.getLines()) {
                            String lineText = line.getText();
                            Point[] lineCornerPoints = line.getCornerPoints();
                            Rect lineFrame = line.getBoundingBox();
                            for (Text.Element element : line.getElements()) {
                              String elementText = element.getText();
                              Point[] elementCornerPoints = element.getCornerPoints();
                              Rect elementFrame = element.getBoundingBox();
                            }
                          }
                        }
                        */
                        textView.setText("OCR:"+resultText+'\n');
                      }
                    })
                    .addOnFailureListener(
                            new OnFailureListener() {
                              @Override
                              public void onFailure(@NonNull Exception e) {
                                Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
                              }
                            });
  }
  private void labelImage(InputImage image){
    labeler.process(image)
            .addOnSuccessListener(new OnSuccessListener<List<ImageLabel>>() {
              @Override
              public void onSuccess(List<ImageLabel> labels) {
                StringBuilder str=new StringBuilder();
                for (ImageLabel label : labels) {
                  String text = label.getText();
                  float confidence = label.getConfidence();
                  int index = label.getIndex();
                  str.append(String.format("%s (%.2f)\n",text,confidence));
                }
                textView.setText("Recognition:"+str.toString());
              }
            })
            .addOnFailureListener(new OnFailureListener() {
              @Override
              public void onFailure(@NonNull Exception e) {
                Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));              }
            });
  }

  private void detectObjects(InputImage image){
    objectDetector.process(image)
            .addOnSuccessListener(new OnSuccessListener<List<DetectedObject>>() {
              @Override
              public void onSuccess(List<DetectedObject> detectedObjects) {
                StringBuilder str=new StringBuilder();
                for (DetectedObject detectedObject : detectedObjects) {
                  Rect boundingBox = detectedObject.getBoundingBox();
                  Integer trackingId = detectedObject.getTrackingId();
                  for (DetectedObject.Label label : detectedObject.getLabels()) {
                    String text = label.getText();
                    float confidence = label.getConfidence();
                    str.append(String.format("%s (%.2f)",text,confidence));
                  }
                  str.append("\n");
                }
                textView.setText("Detection:"+str.toString());
              }
            })
            .addOnFailureListener(new OnFailureListener() {
              @Override
              public void onFailure(@NonNull Exception e) {
                Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));              }
            });
  }

  private void processBarCode(InputImage image){
    barCodeScanner.process(image)
            .addOnSuccessListener(new OnSuccessListener<List<Barcode>>() {
              @Override
              public void onSuccess(List<Barcode> barcodes) {
                StringBuilder str=new StringBuilder();
                for (Barcode barcode : barcodes) {
                  Rect bounds = barcode.getBoundingBox();
                  Point[] corners = barcode.getCornerPoints();
                  String rawValue = barcode.getRawValue();
                  str.append(rawValue+" ");
                  int valueType = barcode.getValueType();
                  // See API reference for complete list of supported types
                  switch (valueType) {
                    case Barcode.TYPE_URL:
                      String title = barcode.getUrl().getTitle();
                      String url = barcode.getUrl().getUrl();
                      str.append(title+" "+url);
                      break;
                  }
                  str.append("\n");
                }
                textView.setText("BarCode:"+str.toString());
              }
            })
            .addOnFailureListener(new OnFailureListener() {
              @Override
              public void onFailure(@NonNull Exception e) {
                Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
              }
            });
  }
}
