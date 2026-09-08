package com.example.myapplication1;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Locale;

/**
 * “输出结果页面”：拿到 CaptureActivity 传来的图片 Uri，后台线程里跑 TFLite 推理，
 * 结果显示 Top-1 + Top-2/3，方便对比模型到底有没有把品种分对。
 */
public class ResultActivity extends AppCompatActivity {

    public static final String EXTRA_IMAGE_URI = "extra_image_uri";

    private ImageView imagePreview;
    private ProgressBar progressRunning;
    private TextView textTop1;
    private TextView textTopK;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        imagePreview = findViewById(R.id.imagePreview);
        progressRunning = findViewById(R.id.progressRunning);
        textTop1 = findViewById(R.id.textTop1);
        textTopK = findViewById(R.id.textTopK);

        findViewById(R.id.buttonRetry).setOnClickListener(v -> finish());
        findViewById(R.id.buttonBackHome).setOnClickListener(v -> {
            Intent intent = new Intent(this, MainActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            startActivity(intent);
            finish();
        });

        Uri imageUri = getImageUriFromIntent();
        if (imageUri == null) {
            progressRunning.setVisibility(View.GONE);
            textTop1.setText(R.string.result_error_load_image);
            return;
        }
        runInference(imageUri);
    }

    @SuppressWarnings("deprecation")
    private Uri getImageUriFromIntent() {
        Intent intent = getIntent();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            return intent.getParcelableExtra(EXTRA_IMAGE_URI, Uri.class);
        }
        return intent.getParcelableExtra(EXTRA_IMAGE_URI);
    }

    private void runInference(Uri imageUri) {
        new Thread(() -> {
            Bitmap bitmap;
            try {
                bitmap = loadBitmap(imageUri);
            } catch (IOException e) {
                runOnUiThread(() -> {
                    progressRunning.setVisibility(View.GONE);
                    textTop1.setText(R.string.result_error_load_image);
                });
                return;
            }

            runOnUiThread(() -> imagePreview.setImageBitmap(bitmap));

            PetClassifier classifier = null;
            try {
                classifier = new PetClassifier(this);
                List<PetClassifier.Prediction> predictions = classifier.classify(bitmap);
                showPredictions(predictions);
            } catch (IOException e) {
                String message = e.getMessage();
                runOnUiThread(() -> {
                    progressRunning.setVisibility(View.GONE);
                    textTop1.setText(getString(R.string.result_error_load_model, message));
                });
            } catch (IllegalStateException e) {
                String message = e.getMessage();
                runOnUiThread(() -> {
                    progressRunning.setVisibility(View.GONE);
                    textTop1.setText(message);
                });
            } finally {
                if (classifier != null) {
                    classifier.close();
                }
            }
        }).start();
    }

    private void showPredictions(List<PetClassifier.Prediction> predictions) {
        runOnUiThread(() -> {
            progressRunning.setVisibility(View.GONE);
            if (predictions.isEmpty()) {
                textTop1.setText(R.string.result_error_load_image);
                return;
            }
            PetClassifier.Prediction top1 = predictions.get(0);
            textTop1.setText(getString(R.string.result_top1_format, top1.label, top1.confidence * 100f));

            StringBuilder sb = new StringBuilder();
            for (int i = 1; i < Math.min(3, predictions.size()); i++) {
                PetClassifier.Prediction p = predictions.get(i);
                if (sb.length() > 0) {
                    sb.append('\n');
                }
                sb.append(String.format(Locale.getDefault(), "%d. %s  %.1f%%", i + 1, p.label, p.confidence * 100f));
            }
            textTopK.setText(sb.toString());
        });
    }

    private Bitmap loadBitmap(Uri uri) throws IOException {
        Bitmap bitmap;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), uri);
            bitmap = ImageDecoder.decodeBitmap(source, (decoder, info, src) ->
                    decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
        } else {
            bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
        }
        if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        }
        return applyExifRotation(uri, bitmap);
    }

    /** 相机 / 相册返回的图片经常带 EXIF 旋转标记，不处理的话预览和识别都会是躺倒的。 */
    private Bitmap applyExifRotation(Uri uri, Bitmap bitmap) {
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            if (in == null) {
                return bitmap;
            }
            ExifInterface exif = new ExifInterface(in);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            int degrees;
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degrees = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degrees = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degrees = 270;
                    break;
                default:
                    degrees = 0;
            }
            if (degrees == 0) {
                return bitmap;
            }
            Matrix matrix = new Matrix();
            matrix.postRotate(degrees);
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        } catch (IOException e) {
            return bitmap;
        }
    }
}
