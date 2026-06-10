package com.gemtun.pipeline.stages;

import com.gemtun.config.TrainingConfig;
import com.gemtun.data.DatasetConverter;
import com.gemtun.pipeline.*;
import com.gemtun.training.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.nio.file.Path;
import java.util.List;

public class TrainLanguageModelStage implements Stage {

    private static final Logger log = LoggerFactory.getLogger(TrainLanguageModelStage.class);

    @Override
    public String name() {
        return "train-language-model";
    }

    @Override
    public StageResult execute(PipelineContext ctx) {
        long start = System.currentTimeMillis();
        try {
            List<String> lmFiles = ctx.getConfig().getData().getActiveLmFiles();
            if (lmFiles.isEmpty()) {
                return new StageResult(name(), false, "No LM data files configured", System.currentTimeMillis() - start);
            }

            Path prepared = new DatasetConverter().prepareLmFiles(ctx.getConfig());
            TrainingConfig lmConfig = ctx.getConfig().getTraining().getLm();
            var modelCfg = ctx.getConfig().getModel();

            var job = new TrainingJob(
                TrainingJob.Type.LM,
                lmConfig,
                prepared.toString(),
                modelCfg.getId(),
                modelCfg.getHfToken(),
                modelCfg.getCacheDir()
            );

            PythonResult result = job.execute();

            if (result.isSuccess()) {
                log.info("Language model training completed in {}ms", result.getDurationMs());
            } else {
                log.error("Language model training failed: {}", result.getMessage());
            }

            return new StageResult(name(), result.isSuccess(), result.getMessage(), result.getDurationMs());

        } catch (Exception e) {
            log.error("LM training stage error", e);
            return new StageResult(name(), false, e.getMessage(), System.currentTimeMillis() - start);
        }
    }
}
