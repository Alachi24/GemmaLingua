package com.gemtun.pipeline;

import com.gemtun.config.PipelineConfig;

public class PipelineContext {

    private final PipelineConfig config;

    public PipelineContext(PipelineConfig config) {
        this.config = config;
    }

    public PipelineConfig getConfig() {
        return config;
    }
}
