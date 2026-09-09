package com.example.myapplication1;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import java.io.File;
import java.util.List;

/**
 * “拍照页面”：一个按钮走系统相机拍一张（存到 app 私有缓存目录，通过 FileProvider
 * 交给相机 app 写入），另一个按钮走系统相册选择器。选好之后直接跳转到 ResultActivity。
 */
public class CaptureActivity extends AppCompatActivity {

    private Uri cameraPhotoUri;
    private Species species;

    private final ActivityResultLauncher<Uri> takePictureLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
                if (success) {
                    openResult(cameraPhotoUri);
                }
            });

    private final ActivityResultLauncher<PickVisualMediaRequest> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                if (uri != null) {
                    openResult(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_capture);

        species = readSpeciesFromIntent();
        TextView title = findViewById(R.id.textCaptureTitle);
        title.setText(getString(R.string.capture_title_format, species.displayNameZh));

        boolean hasCamera = getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY);
        findViewById(R.id.buttonTakePhoto).setEnabled(hasCamera);
        findViewById(R.id.textNoCamera).setVisibility(hasCamera ? View.GONE : View.VISIBLE);

        findViewById(R.id.buttonTakePhoto).setOnClickListener(v -> launchCamera());
        findViewById(R.id.buttonPickGallery).setOnClickListener(v -> pickImageLauncher.launch(
                new PickVisualMediaRequest.Builder()
                        .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                        .build()));
    }

    private void launchCamera() {
        cameraPhotoUri = createCameraPhotoUri();

        // 部分机型上，只把 Uri 塞进 EXTRA_OUTPUT 不够，系统相机 app（另一个进程）
        // 还需要显式的读写权限才能把照片写回这个 Uri。
        Intent probeIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        List<ResolveInfo> resolvedApps = getPackageManager()
                .queryIntentActivities(probeIntent, PackageManager.MATCH_DEFAULT_ONLY);
        for (ResolveInfo resolveInfo : resolvedApps) {
            grantUriPermission(resolveInfo.activityInfo.packageName, cameraPhotoUri,
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
        }

        takePictureLauncher.launch(cameraPhotoUri);
    }

    private Uri createCameraPhotoUri() {
        File dir = new File(getCacheDir(), "captured_images");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        File file = new File(dir, "current_photo.jpg");
        return FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", file);
    }

    private void openResult(Uri imageUri) {
        Intent intent = new Intent(this, ResultActivity.class);
        intent.putExtra(ResultActivity.EXTRA_IMAGE_URI, imageUri);
        intent.putExtra(Species.EXTRA_KEY, species.name());
        startActivity(intent);
    }

    private Species readSpeciesFromIntent() {
        String name = getIntent().getStringExtra(Species.EXTRA_KEY);
        return name != null ? Species.valueOf(name) : Species.DOG;
    }
}
