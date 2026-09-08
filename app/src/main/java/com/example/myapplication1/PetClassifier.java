package com.example.myapplication1;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 加载 assets/pet_classifier.tflite + assets/labels.txt 并跑单张图片推理。
 * 输入尺寸/数据类型直接从模型里读，不写死，这样不管 Colab 那边导出的是
 * float32 模型还是全整数量化模型，这里都能跑；但是归一化方式（均值/方差）
 * 必须和训练时的预处理一致，见下面 buildImageProcessor 里的注释。
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
    private final int[] outputShape;
    private final DataType outputDataType;

    public PetClassifier(Context context) throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        interpreter = new Interpreter(modelBuffer, options);

        labels = FileUtil.loadLabels(context, LABELS_PATH);

        int[] inputShape = interpreter.getInputTensor(0).shape(); // [1, height, width, 3]
        inputHeight = inputShape[1];
        inputWidth = inputShape[2];
        inputDataType = interpreter.getInputTensor(0).dataType();

        outputShape = interpreter.getOutputTensor(0).shape(); // [1, numClasses]
        outputDataType = interpreter.getOutputTensor(0).dataType();

        int numClasses = outputShape[outputShape.length - 1];
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
        TensorImage tensorImage = new TensorImage(inputDataType);
        tensorImage.load(bitmap);
        tensorImage = buildImageProcessor().process(tensorImage);

        TensorBuffer rawOutput = TensorBuffer.createFixedSize(outputShape, outputDataType);
        interpreter.run(tensorImage.getBuffer(), rawOutput.getBuffer().rewind());

        float[] scores = toFloatScores(rawOutput);

        List<Prediction> predictions = new ArrayList<>(scores.length);
        for (int i = 0; i < scores.length && i < labels.size(); i++) {
            predictions.add(new Prediction(labels.get(i), scores[i]));
        }
        Collections.sort(predictions, (a, b) -> Float.compare(b.confidence, a.confidence));
        return predictions;
    }

    private ImageProcessor buildImageProcessor() {
        ImageProcessor.Builder builder = new ImageProcessor.Builder()
                .add(new ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR));
        if (inputDataType == DataType.FLOAT32) {
            // 默认按 Keras MobileNetV2 的 preprocess_input 处理：像素从 [0,255] 映射到 [-1,1]。
            // 如果 Colab 里训练时用的是 Rescaling(1./255)（映射到 [0,1]），
            // 把下面这行换成 new NormalizeOp(0f, 255f)，否则识别结果会明显不对/很随机。
            builder.add(new NormalizeOp(127.5f, 127.5f));
        }
        // 如果 inputDataType 是 UINT8（全整数量化模型），保持原始 0-255 像素值，不做归一化。
        return builder.build();
    }

    private float[] toFloatScores(TensorBuffer rawOutput) {
        if (outputDataType == DataType.UINT8 || outputDataType == DataType.INT8) {
            Tensor.QuantizationParams params = interpreter.getOutputTensor(0).quantizationParams();
            TensorProcessor dequantizer = new TensorProcessor.Builder()
                    .add(new DequantizeOp(params.getZeroPoint(), params.getScale()))
                    .build();
            return dequantizer.process(rawOutput).getFloatArray();
        }
        return rawOutput.getFloatArray();
    }

    public void close() {
        interpreter.close();
    }
}
