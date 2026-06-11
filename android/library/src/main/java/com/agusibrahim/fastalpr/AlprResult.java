package com.agusibrahim.fastalpr;

public class AlprResult {
    public int x;
    public int y;
    public int w;
    public int h;
    public float confidence;
    public String plate;          // null if doOcr was false
    public float ocrConfidence;   // 0.0f if doOcr was false

    public AlprResult(int x, int y, int w, int h, float confidence, String plate, float ocrConfidence) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.confidence = confidence;
        this.plate = plate;
        this.ocrConfidence = ocrConfidence;
    }
}
