package com.gemtun.config;

import com.fasterxml.jackson.annotation.JsonProperty;

public class TrainingConfig {

    private String outputDir;
    private int numTrainEpochs = 3;
    private int perDeviceTrainBatchSize = 2;
    private int gradientAccumulationSteps = 4;
    private double learningRate = 2e-4;
    private int loggingSteps = 50;
    private String saveStrategy = "epoch";
    private int loraR = 16;
    private int loraAlpha = 32;
    private double loraDropout = 0.05;
    private boolean bf16 = true;

    @JsonProperty("output_dir")
    public String getOutputDir() { return outputDir; }
    @JsonProperty("output_dir")
    public void setOutputDir(String v) { outputDir = v; }

    @JsonProperty("num_train_epochs")
    public int getNumTrainEpochs() { return numTrainEpochs; }
    @JsonProperty("num_train_epochs")
    public void setNumTrainEpochs(int v) { numTrainEpochs = v; }

    @JsonProperty("per_device_train_batch_size")
    public int getPerDeviceTrainBatchSize() { return perDeviceTrainBatchSize; }
    @JsonProperty("per_device_train_batch_size")
    public void setPerDeviceTrainBatchSize(int v) { perDeviceTrainBatchSize = v; }

    @JsonProperty("gradient_accumulation_steps")
    public int getGradientAccumulationSteps() { return gradientAccumulationSteps; }
    @JsonProperty("gradient_accumulation_steps")
    public void setGradientAccumulationSteps(int v) { gradientAccumulationSteps = v; }

    @JsonProperty("learning_rate")
    public double getLearningRate() { return learningRate; }
    @JsonProperty("learning_rate")
    public void setLearningRate(double v) { learningRate = v; }

    @JsonProperty("logging_steps")
    public int getLoggingSteps() { return loggingSteps; }
    @JsonProperty("logging_steps")
    public void setLoggingSteps(int v) { loggingSteps = v; }

    @JsonProperty("save_strategy")
    public String getSaveStrategy() { return saveStrategy; }
    @JsonProperty("save_strategy")
    public void setSaveStrategy(String v) { saveStrategy = v; }

    @JsonProperty("lora_r")
    public int getLoraR() { return loraR; }
    @JsonProperty("lora_r")
    public void setLoraR(int v) { loraR = v; }

    @JsonProperty("lora_alpha")
    public int getLoraAlpha() { return loraAlpha; }
    @JsonProperty("lora_alpha")
    public void setLoraAlpha(int v) { loraAlpha = v; }

    @JsonProperty("lora_dropout")
    public double getLoraDropout() { return loraDropout; }
    @JsonProperty("lora_dropout")
    public void setLoraDropout(double v) { loraDropout = v; }

    @JsonProperty("bf16")
    public boolean isBf16() { return bf16; }
    @JsonProperty("bf16")
    public void setBf16(boolean v) { bf16 = v; }
}
