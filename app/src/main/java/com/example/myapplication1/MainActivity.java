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

        findViewById(R.id.buttonCat).setOnClickListener(v -> startCapture(Species.CAT));
        findViewById(R.id.buttonDog).setOnClickListener(v -> startCapture(Species.DOG));

        showModelStatus();
    }

    private void startCapture(Species species) {
        Intent intent = new Intent(this, CaptureActivity.class);
        intent.putExtra(Species.EXTRA_KEY, species.name());
        startActivity(intent);
    }

    /**
     * 只检查 assets 里猫、狗两套模型和标签文件在不在、标签有多少行，不真正加载
     * Interpreter，这样首页打开速度不受影响；真正的加载在 ResultActivity 里做识别时才发生。
     */
    private void showModelStatus() {
        TextView statusView = findViewById(R.id.textModelStatus);
        String catLine = getString(R.string.home_model_status_line, "猫", describeModel(Species.CAT));
        String dogLine = getString(R.string.home_model_status_line, "狗", describeModel(Species.DOG));
        statusView.setText(catLine + "\n" + dogLine);
    }

    private String describeModel(Species species) {
        AssetManager assets = getAssets();
        try (InputStream modelStream = assets.open(species.modelFileName)) {
            int labelCount = countLabels(assets, species.labelsFileName);
            return getString(R.string.home_model_status_ok, labelCount);
        } catch (IOException e) {
            return getString(R.string.home_model_status_missing, species.modelFileName, species.labelsFileName);
        }
    }

    private int countLabels(AssetManager assets, String labelsFileName) throws IOException {
        int count = 0;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(assets.open(labelsFileName), StandardCharsets.UTF_8))) {
            while (reader.readLine() != null) {
                count++;
            }
        }
        return count;
    }
}
