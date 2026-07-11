package com.gemtun.pipeline.stages;

import com.gemtun.config.LanguageConfig;
import com.gemtun.config.TrainingConfig;
import com.gemtun.model.ModelManager;
import com.gemtun.pipeline.*;
import com.gemtun.training.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;

public class TrainMaskingStage implements Stage {

    private static final Logger log = LoggerFactory.getLogger(TrainMaskingStage.class);

    @Override
    public String name() {
        return "train-masking";
    }

    @Override
    public StageResult execute(PipelineContext ctx) {
        long start = System.currentTimeMillis();
        try {
            // Masking examples are derived from the target column of the translation CSV(s).
            List<String> dataFiles = ctx.getConfig().getData().getActiveTranslationFiles();
            if (dataFiles.isEmpty()) {
                return new StageResult(name(), false, "No translation data files configured for masking", System.currentTimeMillis() - start);
            }

            TrainingConfig maskConfig = ctx.getConfig().getTraining().getMasking();
            if (maskConfig == null) {
                return new StageResult(name(), false, "No masking training config defined", System.currentTimeMillis() - start);
            }
            var modelCfg = ctx.getConfig().getModel();

            LanguageConfig language = ctx.getConfig().getLanguage();
            if (language == null) {
                language = new LanguageConfig();   // sensible en->ig defaults
            }

            var job = new TrainingJob(
                TrainingJob.Type.MASKING,
                maskConfig,
                dataFiles,
                modelCfg.getId(),
                modelCfg.getHfToken(),
                modelCfg.getCacheDir()
            ).withLanguage(language);

            // Chain on the translation adapter; fall back to the LM adapter, then base.
            var manager = new ModelManager();
            var transConfig = ctx.getConfig().getTraining().getTranslation();
            var lmConfig = ctx.getConfig().getTraining().getLm();
            if (transConfig != null && transConfig.getOutputDir() != null
                    && manager.adapterExists(transConfig.getOutputDir())) {
                log.info("Found translation adapter at {}, masking will continue training it",
                    transConfig.getOutputDir());
                job.withInitAdapter(transConfig.getOutputDir());
            } else if (lmConfig != null && lmConfig.getOutputDir() != null
                    && manager.adapterExists(lmConfig.getOutputDir())) {
                log.info("No translation adapter found; masking will continue the monolingual adapter at {}",
                    lmConfig.getOutputDir());
                job.withInitAdapter(lmConfig.getOutputDir());
            } else {
                log.info("No prior adapter found, masking will train a fresh adapter from base");
            }

            PythonResult result = job.execute();

            if (result.isSuccess()) {
                log.info("Masking training completed in {}ms", result.getDurationMs());
            } else {
                log.error("Masking training failed: {}", result.getMessage());
            }

            return new StageResult(name(), result.isSuccess(), result.getMessage(), result.getDurationMs());

        } catch (Exception e) {
            log.error("Masking training stage error", e);
            return new StageResult(name(), false, e.getMessage(), System.currentTimeMillis() - start);
        }
    }
}
