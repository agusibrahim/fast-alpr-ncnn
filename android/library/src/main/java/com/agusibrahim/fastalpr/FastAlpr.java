package com.agusibrahim.fastalpr;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class FastAlpr {
    static {
        System.loadLibrary("fast_alpr_ncnn");
    }

    // Initialize models using Android Assets
    public native boolean init(AssetManager assetManager);

    // Native inference call
    private native AlprResult[] nativeDetect(Bitmap bitmap, boolean doOcr);

    // Feature 1: Plate Bounding Box Detection Only
    public AlprResult[] detectPlates(Bitmap bitmap) {
        if (bitmap == null) return new AlprResult[0];
        return nativeDetect(bitmap, false);
    }

    // Feature 2: Plate Bounding Box + OCR Text Recognition
    public AlprResult[] recognizePlates(Bitmap bitmap) {
        if (bitmap == null) return new AlprResult[0];
        return nativeDetect(bitmap, true);
    }
}
