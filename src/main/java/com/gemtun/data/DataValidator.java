package com.gemtun.data;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DataValidator {

    public static List<String> validateLmFile(String path) {
        List<String> errors = new ArrayList<>();
        File f = new File(path);
        if (!f.exists()) {
            errors.add("LM data file not found: " + path);
            return errors;
        }
        String name = f.getName().toLowerCase();
        if (!name.endsWith(".txt") && !name.endsWith(".csv")) {
            errors.add("LM file must be .txt or .csv, got: " + name);
        }
        if (f.length() == 0) {
            errors.add("LM file is empty: " + path);
        }
        return errors;
    }

    public static List<String> validateTranslationFile(String path) {
        List<String> errors = new ArrayList<>();
        File f = new File(path);
        if (!f.exists()) {
            errors.add("Translation data file not found: " + path);
            return errors;
        }
        if (!f.getName().toLowerCase().endsWith(".csv")) {
            errors.add("Translation file must be .csv, got: " + f.getName());
        }
        if (f.length() == 0) {
            errors.add("Translation file is empty: " + path);
        }
        return errors;
    }

    public static List<String> validateAll(List<String> paths, boolean isLm) {
        List<String> errors = new ArrayList<>();
        for (String path : paths) {
            List<String> errs = isLm ? validateLmFile(path) : validateTranslationFile(path);
            errors.addAll(errs);
        }
        return errors;
    }
}
