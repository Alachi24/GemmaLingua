package com.gemtun.config;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.List;

public class PipelineConfig {

    private PipelineDef pipeline;
    private LanguageConfig language;
    private ModelDef model;
    private DataDef data;
    private TrainingDef training;

    public PipelineDef getPipeline() { return pipeline; }
    public void setPipeline(PipelineDef v) { pipeline = v; }

    public LanguageConfig getLanguage() { return language; }
    public void setLanguage(LanguageConfig v) { language = v; }

    public ModelDef getModel() { return model; }
    public void setModel(ModelDef v) { model = v; }

    public DataDef getData() { return data; }
    public void setData(DataDef v) { data = v; }

    public TrainingDef getTraining() { return training; }
    public void setTraining(TrainingDef v) { training = v; }

    public static class PipelineDef {
        private String name;
        private List<String> stages;

        public String getName() { return name; }
        public void setName(String v) { name = v; }

        public List<String> getStages() { return stages; }
        public void setStages(List<String> v) { stages = v; }
    }

    public static class ModelDef {
        private String id;
        @JsonProperty("hf_token")
        private String hfToken;
        @JsonProperty("output_base")
        private String outputBase;
        @JsonProperty("cache_dir")
        private String cacheDir;

        public String getId() { return id; }
        public void setId(String v) { id = v; }

        public String getHfToken() { return hfToken; }
        public void setHfToken(String v) { hfToken = v; }

        public String getOutputBase() { return outputBase; }
        public void setOutputBase(String v) { outputBase = v; }

        public String getCacheDir() { return cacheDir; }
        public void setCacheDir(String v) { cacheDir = v; }
    }

    public static class DataDef {
        @JsonProperty("raw_dir")
        private String rawDir;
        @JsonProperty("processed_dir")
        private String processedDir;
        @JsonProperty("lm_dir")
        private String lmDir;
        @JsonProperty("lm_file")
        private String lmFile;
        @JsonProperty("lm_files")
        private List<String> lmFiles;
        @JsonProperty("translation_dir")
        private String translationDir;
        @JsonProperty("translation_file")
        private String translationFile;
        @JsonProperty("translation_files")
        private List<String> translationFiles;

        public String getRawDir() { return rawDir; }
        public void setRawDir(String v) { rawDir = v; }

        public String getProcessedDir() { return processedDir; }
        public void setProcessedDir(String v) { processedDir = v; }

        public String getLmDir() { return lmDir; }
        public void setLmDir(String v) { lmDir = v; }

        public String getLmFile() { return lmFile; }
        public void setLmFile(String v) { lmFile = v; }

        public List<String> getLmFiles() { return lmFiles; }
        public void setLmFiles(List<String> v) { lmFiles = v; }

        public String getTranslationDir() { return translationDir; }
        public void setTranslationDir(String v) { translationDir = v; }

        public String getTranslationFile() { return translationFile; }
        public void setTranslationFile(String v) { translationFile = v; }

        public List<String> getTranslationFiles() { return translationFiles; }
        public void setTranslationFiles(List<String> v) { translationFiles = v; }

        public List<String> getActiveLmFiles() {
            var files = new LinkedHashSet<String>();
            if (lmFiles != null) files.addAll(lmFiles);
            if (lmFile != null && !lmFile.isBlank()) files.add(lmFile);
            files.addAll(expandDir(lmDir));
            return List.copyOf(files);
        }

        public List<String> getActiveTranslationFiles() {
            var files = new LinkedHashSet<String>();
            if (translationFiles != null) files.addAll(translationFiles);
            if (translationFile != null && !translationFile.isBlank()) files.add(translationFile);
            files.addAll(expandDir(translationDir));
            return List.copyOf(files);
        }

        /** Lists all .txt/.csv files directly inside {@code dir}, sorted; empty if unset/missing. */
        private static List<String> expandDir(String dir) {
            if (dir == null || dir.isBlank()) return List.of();
            Path d = Path.of(dir);
            if (!Files.isDirectory(d)) return List.of();
            try (var stream = Files.list(d)) {
                return stream.filter(Files::isRegularFile)
                    .filter(p -> {
                        String n = p.getFileName().toString().toLowerCase();
                        return n.endsWith(".txt") || n.endsWith(".csv");
                    })
                    .map(Path::toString)
                    .sorted()
                    .toList();
            } catch (IOException e) {
                return List.of();
            }
        }
    }

    public static class TrainingDef {
        @JsonProperty("lm")
        private TrainingConfig lm;
        @JsonProperty("translation")
        private TrainingConfig translation;
        @JsonProperty("masking")
        private TrainingConfig masking;

        public TrainingConfig getLm() { return lm; }
        public void setLm(TrainingConfig v) { lm = v; }

        public TrainingConfig getTranslation() { return translation; }
        public void setTranslation(TrainingConfig v) { translation = v; }

        public TrainingConfig getMasking() { return masking; }
        public void setMasking(TrainingConfig v) { masking = v; }
    }
}
