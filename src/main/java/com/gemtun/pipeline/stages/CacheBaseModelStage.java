package com.gemtun.pipeline.stages;

import com.gemtun.pipeline.*;
import com.gemtun.training.PythonResult;
import com.gemtun.training.PythonRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.List;

public class CacheBaseModelStage implements Stage {

    private static final Logger log = LoggerFactory.getLogger(CacheBaseModelStage.class);

    @Override
    public String name() {
        return "cache-base-model";
    }

    @Override
    public StageResult execute(PipelineContext ctx) {
        long start = System.currentTimeMillis();
        try {
            var modelCfg = ctx.getConfig().getModel();
            var runner = new PythonRunner("python3", 60);

            List<String> args = new ArrayList<>();
            args.add("--model-id"); args.add(modelCfg.getId());
            args.add("--cache-dir"); args.add(modelCfg.getCacheDir());
            if (modelCfg.getHfToken() != null && !modelCfg.getHfToken().isBlank()) {
                args.add("--hf-token"); args.add(modelCfg.getHfToken());
            }

            log.info("Caching base model {} to {}", modelCfg.getId(), modelCfg.getCacheDir());
            PythonResult result = runner.run("python/cache_model.py", args);

            return new StageResult(name(), result.isSuccess(), result.getMessage(), result.getDurationMs());

        } catch (Exception e) {
            log.error("Cache model stage error", e);
            return new StageResult(name(), false, e.getMessage(), System.currentTimeMillis() - start);
        }
    }
}
