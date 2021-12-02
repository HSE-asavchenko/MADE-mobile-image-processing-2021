package com.asav.neural_classifiers;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import org.pytorch.Device;
import org.pytorch.IValue;
import org.pytorch.Tensor;
import org.pytorch.Module;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.LiteModuleLoader;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

/**
 * Created by avsavchenko.
 */
public class TorchModel  implements DeepModel{
    /** Tag for the {@link Log}. */
    private static final String TAG = "TorchModel";

    private Module module=null;
    private int width=224;
    private int height=224;
    private int channels=3;

    private Random rnd=new Random();

    public TorchModel(final Context context, String model_path,int w, int h, int c) throws IOException {
        module=LiteModuleLoader.load(model_path,null, Device.CPU);
        width=w;
        height=h;
        channels=c;
    }
    private static MappedByteBuffer loadModelFile(Context context, String modelFile) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        MappedByteBuffer retFile = inputStream.getChannel().map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
        fileDescriptor.close();
        return retFile;
    }

    public Pair<Long,float[]> classifyImage(Bitmap bitmap) {
        bitmap=Bitmap.createScaledBitmap(bitmap, width, height, false);
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        long startTime = SystemClock.uptimeMillis();
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        long timecostMs=SystemClock.uptimeMillis() - startTime;
        Log.i(TAG, "Timecost to run model inference: " + timecostMs);
        final float[] scores = outputTensor.getDataAsFloatArray();
        return new Pair<Long,float[]>(timecostMs,scores);
    }
}
