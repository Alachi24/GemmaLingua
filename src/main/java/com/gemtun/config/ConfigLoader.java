package com.gemtun.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import java.io.File;
import java.io.IOException;

public class ConfigLoader {

    private final PipelineConfig config;

    public ConfigLoader(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        mapper.findAndRegisterModules();
        this.config = mapper.readValue(new File(path), PipelineConfig.class);
        resolveEnvVars();
    }

    public PipelineConfig getConfig() {
        return config;
    }

    private void resolveEnvVars() {
        var model = config.getModel();
        if (model.getHfToken() != null && model.getHfToken().contains("${HF_TOKEN}")) {
            String env = System.getenv("HF_TOKEN");
            if (env != null && !env.isBlank()) {
                model.setHfToken(model.getHfToken().replace("${HF_TOKEN}", env));
            } else {
                model.setHfToken(null);
            }
        }
    }
}
