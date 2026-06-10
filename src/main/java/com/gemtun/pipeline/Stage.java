package com.gemtun.pipeline;

public interface Stage {
    String name();
    StageResult execute(PipelineContext context);
}
