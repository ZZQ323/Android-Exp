package com.example.myapplication1;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 加载 assets/ 下指定的 .tflite 模型 + 标签文件并跑单张图片推理。同一个类给猫、狗两个
 * 独立模型共用——先选物种再调对应的 cat_classifier.tflite/dog_classifier.tflite，
 * 而不是用一个塞了猫狗两百多类的大模型（那样类别数悬殊会让模型学会"蒙成大类"的捷径）。
 *
 * 特意没用 org.tensorflow.lite.support 那套辅助库（TensorImage/ImageProcessor 之类）：
 * 它的 "-support" 和 "-support-api" 两个 aar 共用同一个 manifest namespace，新版 AGP
 * 对 namespace 唯一性的检查会直接把编译打挂，改名叫 LiteRT 之后这个问题依然存在。
 * 这里直接用最基础的 Interpreter，手动做缩放和归一化，图片小、模型小，性能完全够用。
 *
 * 输入尺寸/数据类型直接从模型里读，不写死，这样不管 Colab 那边导出的是
 * float32 模型还是全整数量化模型，这里都能跑；但是归一化方式（均值/方差）
 * 必须和训练时的预处理一致，见下面 bitmapToInputBuffer 里的注释。
 */
public class PetClassifier {

    public static class Prediction {
        public final String label;
        public final float confidence;

        Prediction(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }

    /**
     * cat_labels.txt / dog_labels.txt 每行是 "英文品种名,中文品种名"，比如
     * "Pembroke,彭布罗克威尔士柯基犬"；没有中文翻译时只有一列也兼容（纯英文品种名）。
     * 物种（猫/狗）不需要再放进标签文件里——用哪个文件就代表哪个物种，见 speciesZh。
     */
    private static class PetLabel {
        final String english;
        final String chinese;

        PetLabel(String english, String chinese) {
            this.english = english;
            this.chinese = chinese;
        }

        String displayName(String speciesZh) {
            String name = chinese.isEmpty() ? english : english + " " + chinese;
            return speciesZh.isEmpty() ? name : name + "（" + speciesZh + "）";
        }
    }

    private final Interpreter interpreter;
    private final List<PetLabel> labels;
    private final String speciesZh;
    private final int inputHeight;
    private final int inputWidth;
    private final DataType inputDataType;
    private final int numClasses;
    private final DataType outputDataType;

    /**
     * @param modelFileName  assets/ 下的 .tflite 文件名，比如 "cat_classifier.tflite"
     * @param labelsFileName assets/ 下对应的标签文件名，比如 "cat_labels.txt"
     * @param speciesZh      结果里追加显示的物种中文名（"猫"/"狗"），传空字符串就不显示
     */
    public PetClassifier(Context context, String modelFileName, String labelsFileName, String speciesZh)
            throws IOException {
        this.speciesZh = speciesZh;
        MappedByteBuffer modelBuffer = loadModelFile(context, modelFileName);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(modelBuffer, options);

        labels = loadLabels(context, labelsFileName);

        int[] inputShape = interpreter.getInputTensor(0).shape(); // [1, height, width, 3]
        inputHeight = inputShape[1];
        inputWidth = inputShape[2];
        inputDataType = interpreter.getInputTensor(0).dataType();

        int[] outputShape = interpreter.getOutputTensor(0).shape(); // [1, numClasses]
        numClasses = outputShape[outputShape.length - 1];
        outputDataType = interpreter.getOutputTensor(0).dataType();

        if (numClasses != labels.size()) {
            throw new IllegalStateException("labels.txt 里有 " + labels.size()
                    + " 个标签，但模型输出是 " + numClasses
                    + " 类，两者必须一致——请对照 Colab 里训练用的 class_names 顺序检查 assets/labels.txt");
        }
    }

    public int getLabelCount() {
        return labels.size();
    }

    public int getInputSize() {
        return Math.max(inputWidth, inputHeight);
    }

    /** 按置信度从高到低排序的全部预测结果 */
    public List<Prediction> classify(Bitmap bitmap) {
        ByteBuffer inputBuffer = bitmapToInputBuffer(bitmap);

        int outputBytesPerElement = (outputDataType == DataType.FLOAT32) ? 4 : 1;
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputBytesPerElement * numClasses);
        outputBuffer.order(ByteOrder.nativeOrder());

        interpreter.run(inputBuffer, outputBuffer);
        outputBuffer.rewind();

        float[] scores = readScores(outputBuffer);

        List<Prediction> predictions = new ArrayList<>(scores.length);
        for (int i = 0; i < scores.length && i < labels.size(); i++) {
            predictions.add(new Prediction(labels.get(i).displayName(speciesZh), scores[i]));
        }
        Collections.sort(predictions, (a, b) -> Float.compare(b.confidence, a.confidence));
        return predictions;
    }

    private ByteBuffer bitmapToInputBuffer(Bitmap bitmap) {
        Bitmap resized = (bitmap.getWidth() == inputWidth && bitmap.getHeight() == inputHeight)
                ? bitmap
                : Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true);

        int bytesPerChannel = (inputDataType == DataType.FLOAT32) ? 4 : 1;
        ByteBuffer buffer = ByteBuffer.allocateDirect(bytesPerChannel * inputWidth * inputHeight * 3);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[inputWidth * inputHeight];
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight);

        for (int pixel : pixels) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            if (inputDataType == DataType.FLOAT32) {
                // 按 Keras MobileNetV2 的 preprocess_input 惯例，把像素从 [0,255] 映射到 [-1,1]。
                // 这是针对上一版 pet_classifier.tflite 核实过的设定；换成新模型后，如果图里
                // 没有内置 Rescaling/Normalize 算子，且训练时用的是 Rescaling(1./255)（映射到
                // [0,1]）而不是 preprocess_input，要把下面三行的 127.5f/127.5f 换成 0f/255f。
                buffer.putFloat((r - 127.5f) / 127.5f);
                buffer.putFloat((g - 127.5f) / 127.5f);
                buffer.putFloat((b - 127.5f) / 127.5f);
            } else {
                // 全整数量化模型（UINT8/INT8 输入）：保持原始 0-255 像素值，不做归一化。
                buffer.put((byte) r);
                buffer.put((byte) g);
                buffer.put((byte) b);
            }
        }
        buffer.rewind();
        return buffer;
    }

    private float[] readScores(ByteBuffer outputBuffer) {
        float[] scores = new float[numClasses];
        if (outputDataType == DataType.FLOAT32) {
            for (int i = 0; i < numClasses; i++) {
                scores[i] = outputBuffer.getFloat();
            }
        } else {
            Tensor.QuantizationParams params = interpreter.getOutputTensor(0).quantizationParams();
            for (int i = 0; i < numClasses; i++) {
                byte raw = outputBuffer.get();
                int rawValue = (outputDataType == DataType.UINT8) ? (raw & 0xFF) : raw;
                scores[i] = (rawValue - params.getZeroPoint()) * params.getScale();
            }
        }
        return scores;
    }

    private MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFileName);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
        }
    }

    private List<PetLabel> loadLabels(Context context, String labelsFileName) throws IOException {
        List<PetLabel> result = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(labelsFileName), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(",", 2);
                String english = parts[0].trim();
                String chinese = parts.length == 2 ? parts[1].trim() : "";
                result.add(new PetLabel(english, chinese));
            }
        }
        return result;
    }

    public void close() {
        interpreter.close();
    }
}
