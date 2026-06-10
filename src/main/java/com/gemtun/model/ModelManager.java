package com.gemtun.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.nio.file.*;

public class ModelManager {

    private static final Logger log = LoggerFactory.getLogger(ModelManager.class);

    public boolean adapterExists(String path) {
        Path adapterDir = Path.of(path);
        if (!Files.exists(adapterDir)) return false;
        return Files.exists(adapterDir.resolve("adapter_config.json"))
            || Files.exists(adapterDir.resolve("adapter_model.safetensors"));
    }

    public long getAdapterSize(String path) {
        Path adapterDir = Path.of(path);
        if (!Files.exists(adapterDir)) return 0;
        try {
            return Files.walk(adapterDir)
                .filter(Files::isRegularFile)
                .mapToLong(p -> p.toFile().length())
                .sum();
        } catch (Exception e) {
            log.warn("Failed to compute adapter size for {}", path, e);
            return 0;
        }
    }

    public boolean deleteAdapter(String path) {
        Path adapterDir = Path.of(path);
        if (!Files.exists(adapterDir)) return true;
        try {
            Files.walk(adapterDir)
                .sorted(java.util.Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);
            log.info("Deleted adapter at {}", path);
            return true;
        } catch (Exception e) {
            log.error("Failed to delete adapter at {}", path, e);
            return false;
        }
    }
}
