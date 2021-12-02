package com.asav.android;

import android.app.Fragment;
import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.*;
import android.widget.*;


import androidx.annotation.Nullable;

import com.github.mikephil.charting.charts.*;
import com.github.mikephil.charting.components.*;
import com.github.mikephil.charting.data.*;
import com.github.mikephil.charting.formatter.IAxisValueFormatter;
import com.github.mikephil.charting.formatter.IValueFormatter;
import com.github.mikephil.charting.highlight.Highlight;
import com.github.mikephil.charting.listener.OnChartValueSelectedListener;
import com.github.mikephil.charting.utils.ViewPortHandler;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by avsavchenko.
 */
public class VisualPreferences extends Fragment implements OnChartValueSelectedListener{
    /** Tag for the {@link Log}. */
    private static final String TAG = "VisualPreferences";

    protected MainActivity mainActivity;
    protected HorizontalBarChart chart=null;

    //protected TextView infoText;

    protected int color, categoryPosition;
    private Button backButton;

    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        return inflater.inflate(R.layout.fragment_preferences, container, false);
    }

    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        color=getArguments().getInt("color",Color.BLACK);
        categoryPosition=getArguments().getInt("position",0);
        String title =getArguments().getString("title","");

        mainActivity=(MainActivity)getActivity();

        TextView titleText=(TextView)view.findViewById(R.id.title_text);
        titleText.setText(title);

        chart = (HorizontalBarChart) view.findViewById(R.id.rating_chart);
        chart.setPinchZoom(false);
        chart.setDrawValueAboveBar(true);
        Description descr=new Description();
        descr.setText("");
        chart.setDescription(descr);
        XAxis xAxis = chart.getXAxis();
        xAxis.setDrawGridLines(false);
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setEnabled(true);
        xAxis.setDrawAxisLine(false);

        YAxis yRight = chart.getAxisRight();
        yRight.setDrawAxisLine(true);
        yRight.setDrawGridLines(false);
        yRight.setEnabled(false);

        chart.getAxisLeft().setAxisMinimum(0);

        backButton=(Button)view.findViewById(R.id.back_hl_prefs_button);
        backButton.setVisibility((getFragmentManager().getBackStackEntryCount()>0)?View.VISIBLE:View.GONE);
        backButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                FragmentManager fm = getFragmentManager();
                if(fm.getBackStackEntryCount() > 0) {
                    Log.d(TAG, "popping backstack");
                    fm.popBackStackImmediate();
                }
            }
        });

        chart.setOnChartValueSelectedListener(this);
        updateChart();
    }
    protected List<Map<String,Map<String, Set<String>>>> getCategoriesHistograms(){
        return mainActivity.getCategoriesHistograms();
    }

    @Override
    public void onValueSelected(Entry entry, Highlight highlight) {
        BarEntry barEntry=(BarEntry)entry;
        IAxisValueFormatter formatter=chart.getXAxis().getValueFormatter();
        if(formatter!=null) {
            String category=formatter.getFormattedValue(entry.getX(), null);
            //Toast.makeText(getActivity(), category + " stack=" + highlight.getStackIndex(), Toast.LENGTH_SHORT).show();
            if(mainActivity==null)
                return;
            FragmentManager fm = getFragmentManager();

            List<Map<String,Map<String,Set<String>>>> categoriesHistograms=getCategoriesHistograms();
            Map<String,Set<String>> fileLists=null;
            if(categoryPosition<categoriesHistograms.size()){
                Map<String,Map<String,Set<String>>> cat_files=categoriesHistograms.get(categoryPosition);
                if(cat_files.containsKey(category)){
                    fileLists=cat_files.get(category);
                }
            }
            if(fileLists!=null && !fileLists.isEmpty()) {
                    Photos photosFragment = new Photos();
                    Bundle args = new Bundle();
                    String[] titles=new String[fileLists.size()];
                    int i=0;
                    for(Map.Entry<String,Set<String>> category2fileList:fileLists.entrySet()){
                        titles[i]=category2fileList.getKey();
                        if(Character.isDigit(titles[i].charAt(0)))
                            titles[i]=String.valueOf(i+1);
                        args.putStringArrayList(titles[i], new ArrayList<String>(category2fileList.getValue()));
                        ++i;
                    }
                    args.putStringArray("photosTaken", titles);
                    photosFragment.setArguments(args);
                    FragmentTransaction fragmentTransaction = fm.beginTransaction();
                    fragmentTransaction.replace(R.id.fragment_switch, photosFragment);
                    fragmentTransaction.addToBackStack(null);
                    fragmentTransaction.commit();
                }

        }
    }

    @Override
    public void onNothingSelected() {
        //Toast.makeText(getActivity(),"Nothing selected",Toast.LENGTH_SHORT).show();
    }

    public void updateChart(){
        if(mainActivity!=null) {
            List<Map<String,Map<String, Set<String>>>> categoriesHistograms=getCategoriesHistograms();
            Map<String,Map<String,Set<String>>> histo=null;
            if (categoryPosition<categoriesHistograms.size()) {
                histo=categoriesHistograms.get(categoryPosition);
            }
            if(histo!=null && !histo.isEmpty())
                updateCategoryChart(histo);
            else
                backButton.performClick();
        }
    }

    protected int getFilesCount(Map<String, Set<String>> id2Files){
        int count = 0;
        for (Set<String> filenames : id2Files.values())
            count += filenames.size();
        return count;
    }
    private void updateCategoryChart(Map<String,Map<String,Set<String>>> histo){
        //infoText.setText("");

        ArrayList<Map.Entry<String,Map<String,Set<String>>>> sortedHisto = new ArrayList<>(histo.entrySet());
        Collections.sort(sortedHisto, new Comparator<Map.Entry<String, Map<String, Set<String>>>>() {
            @Override
            public int compare(Map.Entry<String, Map<String, Set<String>>> kvEntry, Map.Entry<String, Map<String, Set<String>>> t1) {
                return getFilesCount(t1.getValue())-getFilesCount(kvEntry.getValue());
            }
        });

        //Map<String,Set<String>> sortedHisto=sortByValueSize(histo);
        final ArrayList<String> xLabel = new ArrayList<>();
        final List<BarEntry> entries = new ArrayList<BarEntry>();
        int index=0;
        int maxCount=15;
        List<String> keys=new ArrayList<>();
        for (Map.Entry<String,Map<String,Set<String>>> entry : sortedHisto) {
            keys.add(entry.getKey());
            if(keys.size()>maxCount)
                break;
        }
        Collections.reverse(keys);
        for(String key : keys){
            xLabel.add(key);
            int value=(int)Math.round(getFilesCount(histo.get(key)));
            entries.add(new BarEntry(index, value));
            ++index;

            if(index>maxCount)
                break;
        }
        if(!entries.isEmpty())
            chart.getAxisLeft().setAxisMaximum(entries.get(entries.size()-1).getY()+2);

        XAxis xAxis = chart.getXAxis();
        xAxis.setLabelCount(xLabel.size());
        xAxis.setValueFormatter(new IAxisValueFormatter() {
            @Override
            public String getFormattedValue(float value, AxisBase axis) {
                //value=-value;
                if (value>=0 && value<xLabel.size())
                    return xLabel.get((int)value);
                else
                    return "";

            }
        });

        BarDataSet barDataSet = new BarDataSet(entries, "");
        barDataSet.setColor(color);

        BarData data = new BarData(barDataSet);
        data.setBarWidth(0.7f*xLabel.size()/maxCount);
        data.setValueFormatter(new IValueFormatter(){

            @Override
            public String getFormattedValue(float v, Entry entry, int i, ViewPortHandler viewPortHandler) {
                return "" + ((int) v);
            }
        });
        chart.setData(data);
        chart.getLegend().setEnabled(false);
        chart.invalidate();

    }
}