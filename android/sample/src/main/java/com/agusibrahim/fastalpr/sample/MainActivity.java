package com.agusibrahim.fastalpr.sample;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.agusibrahim.fastalpr.FastAlpr;
import com.agusibrahim.fastalpr.AlprResult;

import java.io.InputStream;

public class MainActivity extends Activity {
    private static final int REQUEST_PICK_IMAGE = 1001;

    private FastAlpr alpr;
    private ImageView imageView;
    private CheckBox chkOcr;
    private TextView txtLog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(getResources().getIdentifier("activity_main", "layout", getPackageName()));

        imageView = findViewById(getResources().getIdentifier("image_view", "id", getPackageName()));
        chkOcr = findViewById(getResources().getIdentifier("chk_ocr", "id", getPackageName()));
        txtLog = findViewById(getResources().getIdentifier("txt_log", "id", getPackageName()));

        Button btnSample1 = findViewById(getResources().getIdentifier("btn_sample1", "id", getPackageName()));
        Button btnSample2 = findViewById(getResources().getIdentifier("btn_sample2", "id", getPackageName()));
        Button btnGallery = findViewById(getResources().getIdentifier("btn_gallery", "id", getPackageName()));

        alpr = new FastAlpr();

        // Initialize ALPR in a separate thread so UI does not freeze
        txtLog.setText("Loading models... Please wait...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                final boolean success = alpr.init(getAssets());
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (success) {
                            txtLog.setText("Ready.\nModels loaded successfully.");
                        } else {
                            txtLog.setText("Error: Failed to load models.");
                            Toast.makeText(MainActivity.this, "Failed to load models!", Toast.LENGTH_LONG).show();
                        }
                    }
                });
            }
        }).start();

        btnSample1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                loadAndProcessAsset("65f9903cda758.jpg");
            }
        });

        btnSample2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                loadAndProcessAsset("20220507_165417.jpg");
            }
        });

        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, REQUEST_PICK_IMAGE);
            }
        });
    }

    private void loadAndProcessAsset(String fileName) {
        try {
            InputStream is = getAssets().open(fileName);
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            is.close();
            if (bitmap != null) {
                processImage(bitmap);
            } else {
                txtLog.setText("Error: Failed to load sample image " + fileName);
            }
        } catch (Exception e) {
            txtLog.setText("Error loading sample image: " + e.getMessage());
        }
    }

    private void processImage(Bitmap bitmap) {
        txtLog.setText("Processing image...");
        final boolean enableOcr = chkOcr.isChecked();
        final Bitmap inputBitmap = bitmap;

        new Thread(new Runnable() {
            @Override
            public void run() {
                long t1 = System.currentTimeMillis();
                final AlprResult[] results;
                if (enableOcr) {
                    results = alpr.recognizePlates(inputBitmap);
                } else {
                    results = alpr.detectPlates(inputBitmap);
                }
                long t2 = System.currentTimeMillis();
                final long elapsed = t2 - t1;

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        displayResults(inputBitmap, results, elapsed, enableOcr);
                    }
                });
            }
        }).start();
    }

    private void displayResults(Bitmap original, AlprResult[] results, long elapsedMs, boolean ocrEnabled) {
        if (results == null || results.length == 0) {
            imageView.setImageBitmap(original);
            txtLog.setText("No plates found.\nInference time: " + elapsedMs + " ms");
            return;
        }

        // Draw green boxes on a mutable copy of the bitmap
        Bitmap mutableBitmap = original.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        
        Paint boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Math.max(4f, original.getWidth() / 200f));

        Paint textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(Math.max(20f, original.getWidth() / 30f));
        textPaint.setFakeBoldText(true);

        StringBuilder log = new StringBuilder();
        log.append("=== Detections (").append(elapsedMs).append(" ms) ===\n");

        for (int i = 0; i < results.length; i++) {
            AlprResult res = results[i];
            canvas.drawRect(res.x, res.y, res.x + res.w, res.y + res.h, boxPaint);
            
            log.append(String.format("%d. Box: [%d, %d, %d, %d] | Conf: %.1f%%\n", 
                    i + 1, res.x, res.y, res.w, res.h, res.confidence * 100f));
            
            if (ocrEnabled && res.plate != null) {
                canvas.drawText(res.plate, res.x, res.y - 12, textPaint);
                log.append("   OCR Result: \"").append(res.plate)
                   .append("\" | Conf: ").append(String.format("%.2f", res.ocrConfidence)).append("\n");
            }
        }

        imageView.setImageBitmap(mutableBitmap);
        txtLog.setText(log.toString());
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            try {
                Uri selectedImage = data.getData();
                InputStream is = getContentResolver().openInputStream(selectedImage);
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                is.close();
                if (bitmap != null) {
                    processImage(bitmap);
                } else {
                    txtLog.setText("Error: Selected image is null");
                }
            } catch (Exception e) {
                txtLog.setText("Error loading picked image: " + e.getMessage());
            }
        }
    }
}
