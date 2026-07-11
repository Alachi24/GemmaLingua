package com.gemtun;

import com.gemtun.config.ConfigLoader;
import com.gemtun.pipeline.PipelineContext;
import com.gemtun.pipeline.PipelineOrchestrator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import java.util.concurrent.Callable;

@CommandLine.Command(
    name = "gemtun",
    mixinStandardHelpOptions = true,
    version = "1.0.0",
    description = "Igbo Gemma 4 E4B Fine-Tuning Pipeline"
)
public class GemTunApplication implements Callable<Integer> {

    private static final Logger log = LoggerFactory.getLogger(GemTunApplication.class);

    @CommandLine.Option(
        names = {"-c", "--config"},
        defaultValue = "config/pipeline.yaml",
        description = "Path to pipeline YAML config (default: config/pipeline.yaml)"
    )
    private String configPath;

    @CommandLine.Option(
        names = {"--list-stages"},
        description = "List available pipeline stages and exit"
    )
    private boolean listStages;

    @Override
    public Integer call() throws Exception {
        if (listStages) {
            System.out.println("Available pipeline stages:");
            System.out.println("  cache-base-model     - Download/cache base model from Hugging Face");
            System.out.println("  validate-data        - Validate input data files");
            System.out.println("  train-language-model - Fine-tune on Igbo-only text");
            System.out.println("  train-translation    - Fine-tune on EN-IG pairs");
            System.out.println("  export-model         - Copy trained adapters to final dir");
            return 0;
        }

        log.info("Loading configuration from {}", configPath);
        var loader = new ConfigLoader(configPath);
        var config = loader.getConfig();
        log.info("Pipeline: {}", config.getPipeline().getName());
        log.info("Model: {}", config.getModel().getId());

        var ctx = new PipelineContext(config);
        var orchestrator = new PipelineOrchestrator();
        var results = orchestrator.run(ctx);

        boolean allPassed = results.stream().allMatch(r -> r.isSuccess());
        return allPassed ? 0 : 1;
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new GemTunApplication()).execute(args);
        System.exit(exitCode);
    }
}
