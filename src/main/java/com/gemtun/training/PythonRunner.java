package com.gemtun.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.*;
import java.nio.file.*;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class PythonRunner {

    private static final Logger log = LoggerFactory.getLogger(PythonRunner.class);

    private final String pythonCmd;
    private final long timeoutMinutes;

    public PythonRunner(String pythonCmd, long timeoutMinutes) {
        this.pythonCmd = pythonCmd;
        this.timeoutMinutes = timeoutMinutes;
    }

    public PythonRunner() {
        this(resolvePythonCmd(), 240);
    }

    private static String resolvePythonCmd() {
        Path venv = Path.of("venv/bin/python3");
        if (Files.exists(venv)) {
            log.info("Using venv Python at {}", venv);
            return venv.toString();
        }
        log.info("No venv found, using system python3");
        return "python3";
    }

    public PythonResult run(String scriptPath, List<String> args) {
        long start = System.currentTimeMillis();
        try {
            var command = new java.util.ArrayList<String>();
            command.add(pythonCmd);
            command.add(scriptPath);
            command.addAll(args);

            log.info("Running: {}", String.join(" ", command));

            var pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            pb.directory(new File(System.getProperty("user.dir")));

            Process process = pb.start();

            var output = new StringBuilder();
            try (var reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    log.info("[python] {}", line);
                    output.append(line).append("\n");
                }
            }

            boolean finished = process.waitFor(timeoutMinutes, TimeUnit.MINUTES);
            if (!finished) {
                process.destroyForcibly();
                long elapsed = System.currentTimeMillis() - start;
                return new PythonResult(false, "Timed out after " + timeoutMinutes + " minutes", output.toString(), elapsed);
            }

            int exitCode = process.exitValue();
            long elapsed = System.currentTimeMillis() - start;
            return new PythonResult(exitCode == 0, "Exit code: " + exitCode, output.toString(), elapsed);

        } catch (Exception e) {
            long elapsed = System.currentTimeMillis() - start;
            return new PythonResult(false, "Exception: " + e.getMessage(), "", elapsed);
        }
    }
}
