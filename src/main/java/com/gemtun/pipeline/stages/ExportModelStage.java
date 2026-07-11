package com.gemtun.pipeline.stages;

import com.gemtun.pipeline.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.nio.file.*;

public class ExportModelStage implements Stage {

    private static final Logger log = LoggerFactory.getLogger(ExportModelStage.class);

    @Override
    public String name() {
        return "export-model";
    }

    @Override
    public StageResult execute(PipelineContext ctx) {
        long start = System.currentTimeMillis();
        try {
            var training = ctx.getConfig().getTraining();
            var outputBase = ctx.getConfig().getModel().getOutputBase();
            Path exportDir = Path.of(outputBase, "final");

            Files.createDirectories(exportDir);

            if (training.getLm() != null) {
                Path lmSrc = Path.of(training.getLm().getOutputDir());
                if (Files.exists(lmSrc)) {
                    copyDirectory(lmSrc, exportDir.resolve("lm"));
                    log.info("Copied LM adapter to {}", exportDir.resolve("lm"));
                }
            }

            if (training.getTranslation() != null) {
                Path transSrc = Path.of(training.getTranslation().getOutputDir());
                if (Files.exists(transSrc)) {
                    copyDirectory(transSrc, exportDir.resolve("translation"));
                    log.info("Copied translation adapter to {}", exportDir.resolve("translation"));
                }
            }

            if (training.getMasking() != null) {
                Path maskSrc = Path.of(training.getMasking().getOutputDir());
                if (Files.exists(maskSrc)) {
                    copyDirectory(maskSrc, exportDir.resolve("masking"));
                    log.info("Copied masking adapter to {}", exportDir.resolve("masking"));
                }
            }

            log.info("Model export complete at {}", exportDir);
            return new StageResult(name(), true, "Models exported to " + exportDir, System.currentTimeMillis() - start);

        } catch (Exception e) {
            log.error("Export stage error", e);
            return new StageResult(name(), false, e.getMessage(), System.currentTimeMillis() - start);
        }
    }

    private void copyDirectory(Path src, Path dest) throws IOException {
        Files.walk(src).forEach(source -> {
            try {
                Path target = dest.resolve(src.relativize(source));
                if (source.toFile().isDirectory()) {
                    Files.createDirectories(target);
                } else {
                    Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
