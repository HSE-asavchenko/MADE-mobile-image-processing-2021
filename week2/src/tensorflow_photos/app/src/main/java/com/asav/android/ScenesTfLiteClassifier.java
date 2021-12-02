package com.asav.android;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import com.asav.android.db.ClassifierResult;
import com.asav.android.db.SceneData;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Created by avsavchenko.
 */
public class ScenesTfLiteClassifier extends TfLiteClassifier{

    /** Tag for the {@link Log}. */
    private static final String TAG = "ScenesTfLiteClassifier";

    private static final boolean useMobileNet=true;
    private static final boolean usePrunnedMobileNet=false;
    private static final String MODEL_FILE =
            useMobileNet?(usePrunnedMobileNet?"mobilenet_v2_1.0_pruning_grad_25percent_model_ft.tflite":
                    "places_event_mobilenet2_alpha=1.0_augm_ft_sgd_model.tflite"):"places_event_enet0_augm_ft_sgd_model.tflite";
    private static final String SCENES_LABELS_FILE =
            "scenes_places.txt";
    private static final String FILTERED_INDICES_FILE =
            "scenes_unstable_places.txt";
    private static final String EVENTS_LABELS_FILE =
            "events.txt";

    public TreeMap<String,Integer> sceneLabels2Index =new TreeMap<>();
    private ArrayList<String> sceneLabels = new ArrayList<String>();
    private Map<String,Integer> labels2HighLevelCategories = new HashMap<>();
    private Set<Integer> filteredIndices=new HashSet<>();

    public TreeMap<String,Integer> eventLabels2Index =new TreeMap<>();
    private ArrayList<String> eventLabels = new ArrayList<String>();

    public ScenesTfLiteClassifier(final Context context) throws IOException {
        super(context,MODEL_FILE);

        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(context.getAssets().open(EVENTS_LABELS_FILE)));
            String line;
            int line_ind=0;
            while ((line = br.readLine()) != null) {
                ++line_ind;
                //line=line.toLowerCase();
                line=line.split("#")[0].trim();
                String[] categoryInfo=line.split("=");
                String category=categoryInfo[0];
                eventLabels.add(category);

                int highLevelCategory=Integer.parseInt(categoryInfo[1]);
                labels2HighLevelCategories.put(category,highLevelCategory);
            }
            br.close();

            TreeSet<String> labelsSorted=new TreeSet<>();
            for (int i = 0; i < eventLabels.size(); ++i) {
                String event = eventLabels.get(i);
                if(!labelsSorted.contains(event))
                    labelsSorted.add(event);
            }

            int index=0;
            for(String label : labelsSorted) {
                eventLabels2Index.put(label, index);
                ++index;
            }
        } catch (IOException e) {
            throw new RuntimeException("Problem reading event label file!" , e);
        }

        try {
            br = new BufferedReader(new InputStreamReader(context.getAssets().open(FILTERED_INDICES_FILE)));
            String line;
            while ((line = br.readLine()) != null) {
                filteredIndices.add(Integer.parseInt(line)-1);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading filtered label file!" , e);
        }

        try {
            br = new BufferedReader(new InputStreamReader(context.getAssets().open(SCENES_LABELS_FILE)));
            String line;
            int line_ind=0;
            while ((line = br.readLine()) != null) {
                ++line_ind;
                /*if(filteredIndices.contains(line_ind-1))
                  continue;*/
                //line=line.toLowerCase();
                line=line.split("#")[0].trim();
                String[] categoryInfo=line.split("=");
                String category=categoryInfo[0];
                sceneLabels.add(category);

                int highLevelCategory=Integer.parseInt(categoryInfo[1]);
                labels2HighLevelCategories.put(category,highLevelCategory);
            }
            br.close();

            TreeSet<String> labelsSorted=new TreeSet<>();
            for (int i = 0; i < sceneLabels.size(); ++i) {
                if(filteredIndices.contains(i))
                    continue;
                String scene= sceneLabels.get(i);
                if(!labelsSorted.contains(scene))
                    labelsSorted.add(scene);
            }

            int index=0;
            for(String label : labelsSorted) {
                sceneLabels2Index.put(label, index);
                ++index;
            }
        } catch (IOException e) {
          throw new RuntimeException("Problem reading scene label file!" , e);
        }
    }

    private TreeMap<String,Float> getCategory2Score(float[] predictions, ArrayList<String> labels, boolean filter){
        TreeMap<String,Float> category2Score=new TreeMap<>();
        for (int i = 0; i < predictions.length; ++i) {
            if(filter && filteredIndices.contains(i))
                continue;
            String scene= labels.get(i);
            float score=predictions[i];
            if(category2Score.containsKey(scene)){
                score+=category2Score.get(scene);
            }
            category2Score.put(scene,score);
        }
        return category2Score;
    }
    protected ClassifierResult getResults(float[][][] outputs) {
        TreeMap<String,Float> scene2Score=getCategory2Score(outputs[0][0],sceneLabels,true);
        TreeMap<String,Float> event2Score=null;
        if(usePrunnedMobileNet)
            event2Score=new TreeMap<>();
        else
            event2Score=getCategory2Score(outputs[1][0],eventLabels,false);
        SceneData res=new SceneData(sceneLabels2Index,scene2Score,eventLabels2Index,event2Score);
        return res;
    }
    public int getHighLevelCategory(String category){
        int res=-1;
        if(labels2HighLevelCategories.containsKey(category))
          res= labels2HighLevelCategories.get(category);
        return res;
    }
    protected void addPixelValue(int val) {
        //'RGB'->'BGR' is not needed for all our scene recognition networks
        if(useMobileNet) {
            //float std=127.5f; //mobilenet v1
            float std = 128.0f; //mobilenet v2
            imgData.putFloat(((val >> 16) & 0xFF) / std - 1.0f);
            imgData.putFloat(((val >> 8) & 0xFF) / std - 1.0f);
            imgData.putFloat((val & 0xFF) / std - 1.0f);
        }
        else {
            imgData.putFloat((((val >> 16) & 0xFF) / 255.0f - 0.485f) / 0.229f);
            imgData.putFloat((((val >> 8) & 0xFF) / 255.0f - 0.456f) / 0.224f);
            imgData.putFloat((((val) & 0xFF) / 255.0f - 0.406f) / 0.225f);
        }
    }
    protected TensorOperator getPreprocessNormalizeOp(){
        if(useMobileNet) {
            //float std=127.5f; //mobilenet v1
            float std = 128.0f; //mobilenet v2
            return new NormalizeOp(std, std);
        }
        else{
            float mean[] = {255.0f*0.485f, 255.0f*0.456f, 255.0f*0.406f};
            float std[] = {255.0f*0.229f, 255.0f*0.224f, 255.0f*0.225f};
            return new NormalizeOp(mean,std);
            //return new NormalizeOp(127.0f, 128.0f);
        }
    }

}
