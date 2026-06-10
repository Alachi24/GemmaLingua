package com.gemtun.training;

public class PythonResult {

    private final boolean success;
    private final String message;
    private final String output;
    private final long durationMs;

    public PythonResult(boolean success, String message, String output, long durationMs) {
        this.success = success;
        this.message = message;
        this.output = output;
        this.durationMs = durationMs;
    }

    public boolean isSuccess() { return success; }
    public String getMessage() { return message; }
    public String getOutput() { return output; }
    public long getDurationMs() { return durationMs; }
}
