package com.gemtun.pipeline;

import com.gemtun.pipeline.stages.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;

public class PipelineOrchestrator {

    private static final Logger log = LoggerFactory.getLogger(PipelineOrchestrator.class);

    private final Map<String, Stage> stages;

    public PipelineOrchestrator() {
        stages = new LinkedHashMap<>();
        register(new CacheBaseModelStage());
        register(new ValidateDataStage());
        register(new TrainLanguageModelStage());
        register(new TrainTranslationStage());
        register(new TrainMaskingStage());
        register(new ExportModelStage());
    }

    public void register(Stage stage) {
        stages.put(stage.name(), stage);
    }

    public List<StageResult> run(PipelineContext ctx) {
        var pipelineDef = ctx.getConfig().getPipeline();
        List<String> ordered = pipelineDef.getStages();
        List<StageResult> results = new ArrayList<>();

        log.info("=== Pipeline: {} ===", pipelineDef.getName());
        log.info("Stages to run: {}", String.join(" -> ", ordered));

        for (String stageName : ordered) {
            Stage stage = stages.get(stageName);
            if (stage == null) {
                log.warn("Unknown stage '{}', skipping", stageName);
                continue;
            }

            log.info("--- Stage: {} ---", stageName);
            StageResult result = stage.execute(ctx);
            results.add(result);

            if (result.isSuccess()) {
                log.info("✓ {} completed in {}ms", stageName, result.getDurationMs());
            } else {
                log.error("✗ {} FAILED: {}", stageName, result.getMessage());
            }
        }

        printSummary(results);
        return results;
    }

    private void printSummary(List<StageResult> results) {
        log.info("=== Pipeline Summary ===");
        boolean allPassed = true;
        for (var r : results) {
            String icon = r.isSuccess() ? "✓" : "✗";
            log.info("{} {} ({}ms) - {}", icon, r.getStageName(), r.getDurationMs(), r.getMessage());
            if (!r.isSuccess()) allPassed = false;
        }
        if (allPassed) {
            log.info("Pipeline finished successfully!");
        } else {
            log.warn("Pipeline finished with failures. Check logs above.");
        }
    }
}
