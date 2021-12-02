package com.asav.android;


import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Context;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;

import android.view.View;
import android.util.Log;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.asav.android.db.EXIFData;
import com.asav.android.db.ImageAnalysisResults;
import com.asav.android.db.TopCategoriesData;
import com.asav.android.db.RectFloat;

import java.io.File;
import java.util.*;

/**
 * Created by avsavchenko.
 */

public class MainActivity extends FragmentActivity {

    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;

    private HighLevelVisualPreferences preferencesFragment;
    private Photos photosFragment;

    private ProgressBar progressBar;
    private TextView progressBarinsideText;

    private Thread photoProcessingThread=null;
    private Map<String,Long> photosTaken;
    private ArrayList<String> photosFilenames;
    private int currentPhotoIndex=0;
    private PhotoProcessor photoProcessor = null;

    private String[] categoryList;

    private List<Map<String,Map<String, Set<String>>>> categoriesHistograms=new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else
            init();
    }
    private void init(){
        //checkServerSettings();
        categoryList = getResources().getStringArray(R.array.category_list);

        for(int i=0;i<categoryList.length-1;++i){
            categoriesHistograms.add(new HashMap<>());
        }

        photoProcessor = PhotoProcessor.getPhotoProcessor(this);
        photosTaken = photoProcessor.getCameraImages();
        photosFilenames=new ArrayList<String>(photosTaken.keySet());
        currentPhotoIndex=0;

        progressBar=(ProgressBar) findViewById(R.id.progress);
        progressBar.setMax(photosFilenames.size());
        progressBarinsideText=(TextView)findViewById(R.id.progressBarinsideText);
        progressBarinsideText.setText("");


        photoProcessingThread = new Thread(() -> {
         processAllPhotos();
        }, "photo-processing-thread");
        progressBar.setVisibility(View.VISIBLE);

        preferencesFragment = new HighLevelVisualPreferences();
        Bundle prefArgs = new Bundle();
        prefArgs.putInt("color", Color.GREEN);
        prefArgs.putString("title", "High-Level topCategories");
        preferencesFragment.setArguments(prefArgs);

        photosFragment=new Photos();
        Bundle args = new Bundle();
        args.putStringArray("photosTaken", new String[]{"0"});
        args.putStringArrayList("0",new ArrayList<String>(photoProcessor.getCameraImages().keySet()));
        photosFragment.setArguments(args);
        PreferencesClick(null);

        photoProcessingThread.setPriority(Thread.MIN_PRIORITY);
        photoProcessingThread.start();
    }
    public synchronized List<Map<String,Map<String, Set<String>>>> getCategoriesHistograms(){
        return categoriesHistograms;
    }

    private void processAllPhotos(){
        //ImageAnalysisResults previousPhotoProcessedResult=null;
        for(;currentPhotoIndex<photosTaken.size();++currentPhotoIndex){
            String filename=photosFilenames.get(currentPhotoIndex);
            try {
                File file = new File(filename);

                if (file.exists()) {
                    long startTime = SystemClock.uptimeMillis();
                    ImageAnalysisResults res = photoProcessor.getImageAnalysisResults(filename);

                    long endTime = SystemClock.uptimeMillis();
                    Log.d(TAG, "!!Processed: "+ filename+" in background thread:" + Long.toString(endTime - startTime));
                    processRecognitionResults(res);
                    final int progress=currentPhotoIndex+1;
                    runOnUiThread(() -> {
                        if(progressBar!=null) {
                            progressBar.setProgress(progress);
                            progressBarinsideText.setText(""+100*progress/photosTaken.size()+"%");
                        }
                    });
                }
            } catch (Exception e) {
                e.printStackTrace();
                Log.e(TAG, "While  processing image" + filename + " exception thrown: " + e);
            }
        }
    }



    private void updateCategory(List<Map<String,Map<String, Set<String>>>> histos, int highLevelCategory, String category, String filename){
        if(highLevelCategory>=0) {
            Map<String, Map<String, Set<String>>> histo = histos.get(highLevelCategory);
            if (!histo.containsKey(category)) {
                histo.put(category, new TreeMap<>());
                histo.get(category).put("0", new TreeSet<>());
            }
            histo.get(category).get("0").add(filename);
        }
    }

    private List<Map<String,Map<String, Set<String>>>> deepCopyCategories(List<Map<String,Map<String, Set<String>>>> categories){
        ArrayList<Map<String,Map<String, Set<String>>>> result=new ArrayList<>(categories.size());
        for(Map<String,Map<String, Set<String>>> m:categories){
            Map<String,Map<String, Set<String>>> m1=new HashMap<>(m.size());
            result.add(m1);
            for(Map.Entry<String,Map<String, Set<String>>> me:m.entrySet()){
                Map<String, Set<String>> m2=new TreeMap<>(Collections.reverseOrder());
                m1.put(me.getKey(),m2);
                for(Map.Entry<String, Set<String>> map_files:me.getValue().entrySet()){
                    m2.put(map_files.getKey(),new TreeSet<>(map_files.getValue()));
                }
            }
        }
        return result;
    }
    private synchronized void processRecognitionResults(ImageAnalysisResults results){
        String filename=results.filename;

        String location=results.locations.description;
        List<Map<String,Map<String, Set<String>>>> newCategoriesHistograms = deepCopyCategories(categoriesHistograms);

        List<String> scenes = results.scene.getMostReliableCategories();
        for (String scene : scenes) {
            updateCategory(newCategoriesHistograms, photoProcessor.getHighLevelCategory(scene), scene, filename);
        }
        if(location!=null)
            updateCategory(newCategoriesHistograms, newCategoriesHistograms.size() - 1, location, filename);

        categoriesHistograms=newCategoriesHistograms;

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                preferencesFragment.updateChart();
            }
        });
    }

    public void PreferencesClick(View view) {
        FragmentManager fm = getFragmentManager();
        FragmentTransaction fragmentTransaction = fm.beginTransaction();
        fragmentTransaction.replace(R.id.fragment_switch, preferencesFragment);
        fragmentTransaction.commit();
    }
    public void PhotosClick(View view) {
        FragmentManager fm = getFragmentManager();
        if(fm.getBackStackEntryCount()==0) {
            FragmentTransaction fragmentTransaction = fm.beginTransaction();
            fragmentTransaction.replace(R.id.fragment_switch, photosFragment);
            fragmentTransaction.addToBackStack(null);
            fragmentTransaction.commit();
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
}
