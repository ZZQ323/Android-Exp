package com.example.myapplication1;

/**
 * 猫、狗两个独立模型的入口：先选物种，再用各自的 .tflite/标签文件做识别。
 * 拆成两个模型是为了避免把猫狗塞进同一个 140 类模型时，因为狗的类别数（120）
 * 远多于猫（20），模型学会"蒙成狗"这种走捷径的偏向。
 */
public enum Species {
    CAT("cat_classifier.tflite", "cat_labels.txt", "猫"),
    DOG("dog_classifier.tflite", "dog_labels.txt", "狗");

    public static final String EXTRA_KEY = "extra_species";

    public final String modelFileName;
    public final String labelsFileName;
    public final String displayNameZh;

    Species(String modelFileName, String labelsFileName, String displayNameZh) {
        this.modelFileName = modelFileName;
        this.labelsFileName = labelsFileName;
        this.displayNameZh = displayNameZh;
    }
}
