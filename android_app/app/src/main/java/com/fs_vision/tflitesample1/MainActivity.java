package com.fs_vision.tflitesample1;
import org.tensorflow.lite.Interpreter;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

public class MainActivity extends AppCompatActivity {
    private Executor executor = Executors.newSingleThreadExecutor();
    private Interpreter tflite;
    private String modelFile="converted_model.tflite";
    private TFHelper tfHelper;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        tfHelper = TFHelper.getInstance();
        initTensorFlowAndLoadModel();
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 100);


//        int[] intValues = new int[bitmap.getWidth() * bitmap.getHeight()];
//        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//        Log.d("tag", String.valueOf(intValues[0]));

    }

    public void start() {
        Bitmap bitmap=BitmapFactory.decodeResource(getResources(),R.drawable.a);
         bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        int[] intValues = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        float[][][][] inp=new float[1][224][224][3];
        float[][][][] out=new float[1][224][224][3];
        int pixel=0;
        for (int i=0; i<224;i++) {
            for (int j=0; j<224;j++) {
                    final int val = intValues[pixel++];
                    inp[0][i][j][0]=((val >> 16) & 0xFF);
                    inp[0][i][j][1]=((val >> 8) & 0xFF);
                    inp[0][i][j][2]=((val) & 0xFF);

            }
        }
        tflite.run(inp,out);

        Bitmap bitmap_out = bitmap.copy( bitmap.getConfig(), true);
        int width=bitmap.getWidth();
        int height=bitmap.getHeight();
        Log.d("tag","height:"+height+" width:"+width);
        int[] array=new int[width*height];
        pixel=0;
        for (int i=0; i<height;i++) {
            for (int j=0; j<width;j++) {
                int r=(int)(out[0][i][j][0])<<16;
                int g=(int)(out[0][i][j][1])<<8;
                int b=(int)out[0][i][j][2];
                int pixel_val=r+g+b;
                array[pixel++]=pixel_val;
            }
        }
//
        bitmap_out.setPixels(array, 0, width, 0, 0, width, height);
        Log.d("tag",String.valueOf(out[0][0][0][0]));
//                    inp[0][i][j][0]=((val >> 16) & 0xFF);
//                    inp[0][i][j][1]=((val >> 8) & 0xFF);
//                    inp[0][i][j][2]=((val) & 0xFF);


        SaveImage(bitmap_out);
    }



    private  void SaveImage(Bitmap finalBitmap) {

        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File myDir = new File(root + "/saved_images");
        myDir.mkdirs();

        String fname = "Image-.jpg";
        File file = new File (myDir, fname);
        if (file.exists ()) file.delete ();
        try {
            FileOutputStream out = new FileOutputStream(file);
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }



    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    tflite=new Interpreter(tfHelper.loadModelFile(MainActivity.this,modelFile), new Interpreter.Options());
                    start();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
