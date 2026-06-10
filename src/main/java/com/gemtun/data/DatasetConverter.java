package com.gemtun.data;

import com.gemtun.config.PipelineConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.List;

public class DatasetConverter {

    private static final Logger log = LoggerFactory.getLogger(DatasetConverter.class);

    public Path prepareLmFiles(PipelineConfig config) throws IOException {
        List<String> sources = config.getData().getActiveLmFiles();
        Path dest = Path.of(config.getData().getProcessedDir(), "lm_data_merged.txt");
        Files.createDirectories(dest.getParent());

        try (var writer = Files.newBufferedWriter(dest, StandardCharsets.UTF_8)) {
            for (String src : sources) {
                Path srcPath = resolve(src);
                long kept = 0;
                // Stream line-by-line so large corpora (100s of MB) never load fully into memory.
                try (var reader = Files.newBufferedReader(srcPath, StandardCharsets.UTF_8)) {
                    String raw;
                    while ((raw = reader.readLine()) != null) {
                        String line = raw.strip();
                        if (!line.isEmpty()) {
                            writer.write(line);
                            writer.newLine();
                            kept++;
                        }
                    }
                }
                log.info("Merged {} non-empty lines from {}", kept, srcPath);
            }
        }
        log.info("Merged LM data written to {}", dest);
        return dest;
    }

    public Path prepareTranslationFiles(PipelineConfig config) throws IOException {
        List<String> sources = config.getData().getActiveTranslationFiles();
        Path dest = Path.of(config.getData().getProcessedDir(), "translation_data_merged.csv");
        Files.createDirectories(dest.getParent());

        boolean headerWritten = false;
        try (var writer = Files.newBufferedWriter(dest, StandardCharsets.UTF_8)) {
            for (String src : sources) {
                Path srcPath = resolve(src);
                List<String> lines = Files.readAllLines(srcPath, StandardCharsets.UTF_8);
                for (int i = 0; i < lines.size(); i++) {
                    String line = lines.get(i).strip();
                    if (line.isEmpty()) continue;
                    if (i == 0 && !headerWritten) {
                        writer.write(line);
                        writer.newLine();
                        headerWritten = true;
                    } else if (i > 0) {
                        writer.write(line);
                        writer.newLine();
                    }
                }
                log.info("Merged {} rows from {}", lines.size() - 1, srcPath);
            }
        }
        log.info("Merged translation data written to {}", dest);
        return dest;
    }

    private Path resolve(String rawPath) {
        Path p = Path.of(rawPath);
        if (!p.isAbsolute()) {
            p = Path.of(System.getProperty("user.dir"), rawPath);
        }
        return p;
    }
}
