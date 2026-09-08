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
 * 加载 assets/pet_classifier.tflite + assets/labels.txt 并跑单张图片推理。
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

    private static final String MODEL_PATH = "pet_classifier.tflite";
    private static final String LABELS_PATH = "labels.txt";

    public static class Prediction {
        public final String label;
        public final float confidence;

        Prediction(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }

    private final Interpreter interpreter;
    private final List<String> labels;
    private final int inputHeight;
    private final int inputWidth;
    private final DataType inputDataType;
    private final int numClasses;
    private final DataType outputDataType;

    public PetClassifier(Context context) throws IOException {
        MappedByteBuffer modelBuffer = loadModelFile(context);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(modelBuffer, options);

        labels = loadLabels(context);

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
            predictions.add(new Prediction(labels.get(i), scores[i]));
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
                // 已经用 pet_classifier.tflite 实际的算子图核实过：输入张量后面直接就是
                // CONV_2D（MobileNetV2 的 stem），图里没有内置任何 Rescaling/Normalize 算子，
                // 所以缩放要在 app 这一侧做。按 Keras MobileNetV2 的 preprocess_input 惯例，
                // 像素从 [0,255] 映射到 [-1,1]。如果之后换了别的 base model / 训练时用的是
                // Rescaling(1./255)（映射到 [0,1]），把下面三行的 127.5f/127.5f 换成 0f/255f。
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

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_PATH);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
        }
    }

    private List<String> loadLabels(Context context) throws IOException {
        List<String> result = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(LABELS_PATH), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    result.add(line.trim());
                }
            }
        }
        return result;
    }

    public void close() {
        interpreter.close();
    }
}
