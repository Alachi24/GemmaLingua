package com.gemtun.training;

import com.gemtun.config.LanguageConfig;
import com.gemtun.config.TrainingConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class TrainingJob {

    private static final Logger log = LoggerFactory.getLogger(TrainingJob.class);

    public enum Type {
        LM("python/train_lm.py"),
        TRANSLATION("python/train_translation.py"),
        MASKING("python/train_masking.py");

        private final String scriptPath;
        Type(String scriptPath) { this.scriptPath = scriptPath; }
        public String getScriptPath() { return scriptPath; }
    }

    private final Type type;
    private final TrainingConfig config;
    private final List<String> dataPaths;
    private final String modelId;
    private final String hfToken;
    private final String cacheDir;
    private final PythonRunner runner;

    // Optional, translation-only: set via fluent setters before execute().
    private LanguageConfig language;
    private String initAdapter;

    public TrainingJob(Type type, TrainingConfig config, List<String> dataPaths, String modelId, String hfToken, String cacheDir) {
        this.type = type;
        this.config = config;
        this.dataPaths = dataPaths;
        this.modelId = modelId;
        this.hfToken = hfToken;
        this.cacheDir = cacheDir;
        this.runner = new PythonRunner();
    }

    /** Source/target language pair used to build the translation prompt and pick CSV columns. */
    public TrainingJob withLanguage(LanguageConfig language) {
        this.language = language;
        return this;
    }

    /** Path to an existing adapter to load and keep training (chains LM -> translation). */
    public TrainingJob withInitAdapter(String initAdapter) {
        this.initAdapter = initAdapter;
        return this;
    }

    public PythonResult execute() {
        log.info("Starting {} training stage", type.name());

        String resolvedModelId = resolveModelId();
        log.info("Using model: {}", resolvedModelId);

        List<String> args = new ArrayList<>();
        args.add("--model-id"); args.add(resolvedModelId);
        for (String dp : dataPaths) {
            args.add("--data"); args.add(dp);
        }
        args.add("--output"); args.add(config.getOutputDir());
        args.add("--epochs"); args.add(String.valueOf(config.getNumTrainEpochs()));
        args.add("--batch-size"); args.add(String.valueOf(config.getPerDeviceTrainBatchSize()));
        args.add("--grad-accum"); args.add(String.valueOf(config.getGradientAccumulationSteps()));
        args.add("--lr"); args.add(String.valueOf(config.getLearningRate()));
        args.add("--logging-steps"); args.add(String.valueOf(config.getLoggingSteps()));
        args.add("--save-strategy"); args.add(config.getSaveStrategy());
        args.add("--lora-r"); args.add(String.valueOf(config.getLoraR()));
        args.add("--lora-alpha"); args.add(String.valueOf(config.getLoraAlpha()));
        args.add("--lora-dropout"); args.add(String.valueOf(config.getLoraDropout()));
        if (config.isBf16()) {
            args.add("--bf16");
        }
        if (hfToken != null && !hfToken.isBlank()) {
            args.add("--hf-token"); args.add(hfToken);
        }
        if (language != null) {
            args.add("--source-col"); args.add(language.getSourceCode());
            args.add("--target-col"); args.add(language.getTargetCode());
            args.add("--source-name"); args.add(language.getSourceName());
            args.add("--target-name"); args.add(language.getTargetName());
        }
        if (type == Type.MASKING) {
            args.add("--mask-rate"); args.add(String.valueOf(config.getMaskRate()));
            args.add("--mean-span"); args.add(String.valueOf(config.getMeanSpan()));
            args.add("--max-spans"); args.add(String.valueOf(config.getMaxSpans()));
        }
        if (initAdapter != null && !initAdapter.isBlank()) {
            log.info("Continuing training from adapter at {}", initAdapter);
            args.add("--init-adapter"); args.add(initAdapter);
        }

        return runner.run(type.getScriptPath(), args);
    }

    private String resolveModelId() {
        if (cacheDir == null || cacheDir.isBlank()) {
            return modelId;
        }
        String localPath = cacheDir.replaceFirst("/+$", "") + "/" + modelId.replace("/", "--");
        if (Files.exists(Path.of(localPath, "config.json"))) {
            log.info("Using cached model at {}", localPath);
            return localPath;
        }
        return modelId;
    }
}
