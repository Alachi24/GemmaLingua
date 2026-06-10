package com.gemtun.pipeline;

public class StageResult {

    private final String stageName;
    private final boolean success;
    private final String message;
    private final long durationMs;

    public StageResult(String stageName, boolean success, String message, long durationMs) {
        this.stageName = stageName;
        this.success = success;
        this.message = message;
        this.durationMs = durationMs;
    }

    public String getStageName() { return stageName; }
    public boolean isSuccess() { return success; }
    public String getMessage() { return message; }
    public long getDurationMs() { return durationMs; }
}
