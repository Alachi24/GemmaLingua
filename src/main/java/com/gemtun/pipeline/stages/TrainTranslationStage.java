package com.gemtun.pipeline.stages;

import com.gemtun.config.LanguageConfig;
import com.gemtun.config.TrainingConfig;
import com.gemtun.model.ModelManager;
import com.gemtun.pipeline.*;
import com.gemtun.training.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;

public class TrainTranslationStage implements Stage {

    private static final Logger log = LoggerFactory.getLogger(TrainTranslationStage.class);

    @Override
    public String name() {
        return "train-translation";
    }

    @Override
    public StageResult execute(PipelineContext ctx) {
        long start = System.currentTimeMillis();
        try {
            List<String> transFiles = ctx.getConfig().getData().getActiveTranslationFiles();
            if (transFiles.isEmpty()) {
                return new StageResult(name(), false, "No translation data files configured", System.currentTimeMillis() - start);
            }

            TrainingConfig transConfig = ctx.getConfig().getTraining().getTranslation();
            var modelCfg = ctx.getConfig().getModel();

            LanguageConfig language = ctx.getConfig().getLanguage();
            if (language == null) {
                language = new LanguageConfig();   // sensible en->ig defaults
            }

            var job = new TrainingJob(
                TrainingJob.Type.TRANSLATION,
                transConfig,
                transFiles,
                modelCfg.getId(),
                modelCfg.getHfToken(),
                modelCfg.getCacheDir()
            ).withLanguage(language);

            // Chain on top of the monolingual adapter if the LM stage produced one.
            var lmConfig = ctx.getConfig().getTraining().getLm();
            if (lmConfig != null && lmConfig.getOutputDir() != null
                    && new ModelManager().adapterExists(lmConfig.getOutputDir())) {
                log.info("Found monolingual adapter at {}, translation will continue training it",
                    lmConfig.getOutputDir());
                job.withInitAdapter(lmConfig.getOutputDir());
            } else {
                log.info("No monolingual adapter found, translation will train a fresh adapter from base");
            }

            PythonResult result = job.execute();

            if (result.isSuccess()) {
                log.info("Translation training completed in {}ms", result.getDurationMs());
            } else {
                log.error("Translation training failed: {}", result.getMessage());
            }

            return new StageResult(name(), result.isSuccess(), result.getMessage(), result.getDurationMs());

        } catch (Exception e) {
            log.error("Translation training stage error", e);
            return new StageResult(name(), false, e.getMessage(), System.currentTimeMillis() - start);
        }
    }
}
