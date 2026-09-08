package com.example.myapplication1;

import android.content.Intent;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        findViewById(R.id.buttonStart).setOnClickListener(v ->
                startActivity(new Intent(this, CaptureActivity.class)));

        showModelStatus();
    }

    /**
     * 只检查 assets 里有没有模型和标签文件、标签有多少行，不真正加载 Interpreter，
     * 这样首页打开速度不受影响；真正的加载在 ResultActivity 里做识别时才发生。
     */
    private void showModelStatus() {
        TextView statusView = findViewById(R.id.textModelStatus);
        AssetManager assets = getAssets();
        try (InputStream modelStream = assets.open("pet_classifier.tflite")) {
            int labelCount = countLabels(assets);
            statusView.setText(getString(R.string.home_model_status_ok, "pet_classifier.tflite", labelCount));
        } catch (IOException e) {
            statusView.setText(R.string.home_model_status_missing);
        }
    }

    private int countLabels(AssetManager assets) throws IOException {
        int count = 0;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(assets.open("labels.txt"), StandardCharsets.UTF_8))) {
            while (reader.readLine() != null) {
                count++;
            }
        }
        return count;
    }
}
