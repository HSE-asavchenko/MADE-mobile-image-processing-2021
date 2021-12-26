package com.asav.facialprocessing;

import java.io.Serializable;


/**
 * Created by avsavchenko.
 */
public class SmilingData implements ClassifierResult,Serializable {
    public float[] emotionScores=null;

    public SmilingData(){

    }
    public SmilingData(float[] emotionScores){
        this.emotionScores = new float[emotionScores.length];
        System.arraycopy(emotionScores, 0, this.emotionScores, 0, emotionScores.length);
    }

    private static String[] emotions={"","Smiling", "Not smiling"};
    public static String getEmotion(float[] emotionScores){
        int bestInd=-1;
        if (emotionScores!=null){
            float maxScore=0;
            for(int i=0;i<emotionScores.length;++i){
                if(maxScore<emotionScores[i]){
                    maxScore=emotionScores[i];
                    bestInd=i;
                }
            }
        }
        return emotions[bestInd+1];
    }
    public String toString(){
        return getEmotion(emotionScores);
    }
}
