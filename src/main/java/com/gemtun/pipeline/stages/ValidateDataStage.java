package com.gemtun.pipeline.stages;

import com.gemtun.data.DataValidator;
import com.gemtun.pipeline.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;

public class ValidateDataStage implements Stage {

    private static final Logger log = LoggerFactory.getLogger(ValidateDataStage.class);

    @Override
    public String name() {
        return "validate-data";
    }

    @Override
    public StageResult execute(PipelineContext ctx) {
        long start = System.currentTimeMillis();
        var data = ctx.getConfig().getData();

        List<String> errors = new java.util.ArrayList<>();

        List<String> lmFiles = data.getActiveLmFiles();
        if (lmFiles.isEmpty()) {
            log.warn("No LM data files configured, skipping LM validation");
        } else {
            errors.addAll(DataValidator.validateAll(lmFiles, true));
        }

        List<String> transFiles = data.getActiveTranslationFiles();
        if (transFiles.isEmpty()) {
            log.warn("No translation data files configured, skipping translation validation");
        } else {
            errors.addAll(DataValidator.validateAll(transFiles, false));
        }

        if (!errors.isEmpty()) {
            String msg = "Validation failed:\n" + String.join("\n", errors);
            log.error(msg);
            return new StageResult(name(), false, msg, System.currentTimeMillis() - start);
        }

        return new StageResult(name(), true, "All data files valid", System.currentTimeMillis() - start);
    }
}
