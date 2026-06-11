package com.agusibrahim.fastalpr.sample;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.TextView;
import android.widget.Toast;

import com.agusibrahim.fastalpr.FastAlpr;
import com.agusibrahim.fastalpr.AlprResult;

import java.util.List;

@SuppressWarnings("deprecation")
public class LiveCameraActivity extends Activity implements TextureView.SurfaceTextureListener {
    private static final int PERMISSION_REQUEST_CAMERA = 2001;

    private TextureView textureView;
    private OverlayView overlayView;
    private TextView txtInfo;
    private CheckBox chkOcr;
    private Button btnBack;

    private Camera camera;
    private FastAlpr alpr;
    
    private HandlerThread processingThread;
    private Handler processingHandler;
    
    private volatile boolean isProcessingFrame = false;
    private volatile boolean isLoopActive = false;
    private boolean isCameraReady = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Keep screen on while using live camera
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        setContentView(getResources().getIdentifier("activity_live_camera", "layout", getPackageName()));

        textureView = findViewById(getResources().getIdentifier("camera_preview", "id", getPackageName()));
        overlayView = findViewById(getResources().getIdentifier("overlay_view", "id", getPackageName()));
        txtInfo = findViewById(getResources().getIdentifier("txt_info", "id", getPackageName()));
        chkOcr = findViewById(getResources().getIdentifier("chk_ocr", "id", getPackageName()));
        btnBack = findViewById(getResources().getIdentifier("btn_back", "id", getPackageName()));

        textureView.setSurfaceTextureListener(this);

        alpr = new FastAlpr();
        
        btnBack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });

        // Check and request camera permission
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, PERMISSION_REQUEST_CAMERA);
        } else {
            initAlprAndStartLoop();
        }
    }

    private void initAlprAndStartLoop() {
        // Run ALPR initialization in a helper thread (typically instantaneous since it's already cached)
        new Thread(new Runnable() {
            @Override
            public void run() {
                alpr.init(getAssets());
            }
        }).start();

        // Start processing background thread
        processingThread = new HandlerThread("AlprProcessingThread");
        processingThread.start();
        processingHandler = new Handler(processingThread.getLooper());
        
        isLoopActive = true;
        processingHandler.post(frameGrabberRunnable);
    }

    private final Runnable frameGrabberRunnable = new Runnable() {
        @Override
        public void run() {
            if (isLoopActive) {
                if (!isProcessingFrame && isCameraReady && textureView.isAvailable()) {
                    isProcessingFrame = true;
                    // Grab current frame resized to 640x480 to keep high speed processing
                    final Bitmap frameBitmap = textureView.getBitmap(640, 480);
                    if (frameBitmap != null) {
                        final boolean runOcr = chkOcr.isChecked();
                        processingHandler.post(new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    long startTime = System.currentTimeMillis();
                                    final AlprResult[] results;
                                    if (runOcr) {
                                        results = alpr.recognizePlates(frameBitmap);
                                    } else {
                                        results = alpr.detectPlates(frameBitmap);
                                    }
                                    long endTime = System.currentTimeMillis();
                                    final long elapsed = endTime - startTime;

                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            overlayView.setResults(results, frameBitmap.getWidth(), frameBitmap.getHeight());
                                            txtInfo.setText(String.format("Inference: %d ms | Mode: %s", 
                                                    elapsed, runOcr ? "YOLO+OCR" : "YOLO Only"));
                                            frameBitmap.recycle();
                                            isProcessingFrame = false;
                                        }
                                    });
                                } catch (Exception e) {
                                    frameBitmap.recycle();
                                    isProcessingFrame = false;
                                }
                            }
                        });
                    } else {
                        isProcessingFrame = false;
                    }
                }
                // Schedule next grab in 40ms (~25 FPS target check)
                if (isLoopActive) {
                    new Handler(getMainLooper()).postDelayed(frameGrabberRunnable, 40);
                }
            }
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            if (processingThread == null) {
                initAlprAndStartLoop();
            }
            if (textureView.isAvailable()) {
                openCamera(textureView.getSurfaceTexture());
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        releaseCamera();
        stopProcessingLoop();
    }

    private void stopProcessingLoop() {
        isLoopActive = false;
        if (processingThread != null) {
            processingThread.quitSafely();
            try {
                processingThread.join();
            } catch (InterruptedException e) {
                // Ignore
            }
            processingThread = null;
            processingHandler = null;
        }
    }

    private void openCamera(SurfaceTexture surfaceTexture) {
        if (camera != null) {
            releaseCamera();
        }

        try {
            camera = Camera.open(0); // Rear camera
            camera.setPreviewTexture(surfaceTexture);
            
            Camera.Parameters params = camera.getParameters();
            List<String> focusModes = params.getSupportedFocusModes();
            if (focusModes != null) {
                if (focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                } else if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                    params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                }
            }
            camera.setParameters(params);
            
            // Set preview display angle to portrait (90 deg)
            camera.setDisplayOrientation(90);
            
            camera.startPreview();
            isCameraReady = true;
        } catch (Exception e) {
            Toast.makeText(this, "Failed to open camera: " + e.getMessage(), Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void releaseCamera() {
        isCameraReady = false;
        if (camera != null) {
            try {
                camera.stopPreview();
                camera.release();
            } catch (Exception e) {
                // Ignore
            }
            camera = null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initAlprAndStartLoop();
                if (textureView.isAvailable()) {
                    openCamera(textureView.getSurfaceTexture());
                }
            } else {
                Toast.makeText(this, "Camera permission is required for live detection!", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    // --- TextureView.SurfaceTextureListener Callbacks ---

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera(surface);
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        // Handle changes in texture layout if needed
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        releaseCamera();
        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        // Called when a preview frame updates
    }
}
