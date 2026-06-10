package com.gemtun.config;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Defines the source/target language pair for translation training.
 * Drives both the CSV column names and the human-readable names used in the
 * prompt template, so the pipeline can be retargeted at any language pair
 * purely through config.
 */
public class LanguageConfig {

    private String sourceCode = "en";
    private String sourceName = "English";
    private String targetCode = "ig";
    private String targetName = "Igbo";

    @JsonProperty("source_code")
    public String getSourceCode() { return sourceCode; }
    @JsonProperty("source_code")
    public void setSourceCode(String v) { sourceCode = v; }

    @JsonProperty("source_name")
    public String getSourceName() { return sourceName; }
    @JsonProperty("source_name")
    public void setSourceName(String v) { sourceName = v; }

    @JsonProperty("target_code")
    public String getTargetCode() { return targetCode; }
    @JsonProperty("target_code")
    public void setTargetCode(String v) { targetCode = v; }

    @JsonProperty("target_name")
    public String getTargetName() { return targetName; }
    @JsonProperty("target_name")
    public void setTargetName(String v) { targetName = v; }
}
