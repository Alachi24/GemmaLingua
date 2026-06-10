package com.gemtun.config;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class ConfigLoaderTest {

    @Test
    void loadsDefaultConfig() throws Exception {
        Path cfg = Path.of("config/pipeline.yaml");
        if (!Files.exists(cfg)) return;

        var loader = new ConfigLoader(cfg.toString());
        var config = loader.getConfig();

        assertNotNull(config.getPipeline());
        assertNotNull(config.getModel());
        assertEquals("google/gemma-4-E4B-it", config.getModel().getId());
    }

    @Test
    void validatesLmDataFile() {
        var errors = com.gemtun.data.DataValidator.validateLmFile("/nonexistent/file.txt");
        assertFalse(errors.isEmpty());
        assertTrue(errors.get(0).contains("not found"));
    }

    @Test
    void validatesTranslationFile() {
        var errors = com.gemtun.data.DataValidator.validateTranslationFile("/nonexistent/file.csv");
        assertFalse(errors.isEmpty());
        assertTrue(errors.get(0).contains("not found"));
    }
}
