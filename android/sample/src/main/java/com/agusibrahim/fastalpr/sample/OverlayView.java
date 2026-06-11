package com.agusibrahim.fastalpr.sample;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import com.agusibrahim.fastalpr.AlprResult;

public class OverlayView extends View {
    private AlprResult[] results;
    private int previewWidth;
    private int previewHeight;
    private Paint boxPaint;
    private Paint textPaint;
    private Paint textBgPaint;

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6.0f);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(40.0f);
        textPaint.setFakeBoldText(true);

        textBgPaint = new Paint();
        textBgPaint.setColor(Color.RED);
        textBgPaint.setStyle(Paint.Style.FILL);
    }

    public void setResults(AlprResult[] results, int previewWidth, int previewHeight) {
        this.results = results;
        this.previewWidth = previewWidth;
        this.previewHeight = previewHeight;
        invalidate(); // Trigger repaint on UI thread
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (results == null || results.length == 0 || previewWidth == 0 || previewHeight == 0) {
            return;
        }

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        // Calculate scaling factor to map preview bitmap coordinates to screen view dimensions
        float scaleX = (float) viewWidth / previewWidth;
        float scaleY = (float) viewHeight / previewHeight;

        for (AlprResult res : results) {
            float left = res.x * scaleX;
            float top = res.y * scaleY;
            float right = (res.x + res.w) * scaleX;
            float bottom = (res.y + res.h) * scaleY;

            // Draw bounding box
            canvas.drawRect(left, top, right, bottom, boxPaint);

            // Draw label background and text if plate text exists
            if (res.plate != null && !res.plate.trim().isEmpty()) {
                String text = String.format("%s (%.1f%%)", res.plate, res.ocrConfidence * 100f);
                float textWidth = textPaint.measureText(text);
                float textHeight = textPaint.getTextSize();

                // Draw solid red background behind text
                canvas.drawRect(left, top - textHeight - 15, left + textWidth + 10, top, textBgPaint);
                // Draw text inside the background box
                canvas.drawText(text, left + 5, top - 8, textPaint);
            } else {
                String text = String.format("Conf: %.1f%%", res.confidence * 100f);
                float textWidth = textPaint.measureText(text);
                float textHeight = textPaint.getTextSize();
                canvas.drawRect(left, top - textHeight - 15, left + textWidth + 10, top, textBgPaint);
                canvas.drawText(text, left + 5, top - 8, textPaint);
            }
        }
    }
}
