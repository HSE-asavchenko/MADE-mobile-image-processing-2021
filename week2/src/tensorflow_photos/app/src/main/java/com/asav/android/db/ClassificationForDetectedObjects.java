package com.asav.android.db;

import java.io.Serializable;
import java.util.List;

/**
 * Created by avsavchenko.
 */
public class ClassificationForDetectedObjects implements Serializable {
    public List<TopCategoriesData> concreteTypes=null;

    public ClassificationForDetectedObjects(){}

    public ClassificationForDetectedObjects(List<TopCategoriesData> concreteTypes){
        this.concreteTypes=concreteTypes;
    }
}
