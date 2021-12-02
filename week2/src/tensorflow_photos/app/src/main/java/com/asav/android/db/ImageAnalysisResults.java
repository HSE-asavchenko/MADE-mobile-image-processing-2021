package com.asav.android.db;

import java.io.Serializable;

/**
 * Created by avsavchenko.
 */
public class ImageAnalysisResults implements Serializable {
    public String filename=null;
    public SceneData scene=null;
    public EXIFData locations=null;

    public ImageAnalysisResults() {}

    public ImageAnalysisResults(String filename, SceneData scene, EXIFData locations){
        this.filename=filename;
        this.scene=scene;
        this.locations=locations;
    }
    public ImageAnalysisResults(SceneData scene){
        this.scene=scene;
    }
}
