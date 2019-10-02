package com.reactlibrary;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Canvas;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;
import java.util.Vector;

public class AutomlVisionTfliteModule extends ReactContextBaseJavaModule {
    class LoadedModel {
        private Interpreter tfLite;
        private Vector<String> labels = new Vector<>();

        private LoadedModel(ByteBuffer model) {
            this(model, 1);
        }
        private LoadedModel(ByteBuffer model, int numThreads) {
            final Interpreter.Options tfliteOptions = new Interpreter.Options();
            tfliteOptions.setNumThreads(numThreads);
            tfLite = new Interpreter(model, tfliteOptions);
        }

        public void readLabels(InputStream labelStream) throws IOException {
            try (
                InputStreamReader isr = new InputStreamReader(labelStream);
                BufferedReader br = new BufferedReader(isr);
            ) {
                labels = new Vector<>();
                String line;
                while ((line = br.readLine()) != null) {
                    labels.add(line);
                }
            } catch (Exception e) {
                throw new IOException("Failed to read labels", e);
            }
        }

        public WritableArray run(final String path, final int numResults, final float threshold) throws IOException {
            byte[][] labelProb = new byte[1][labels.size()];
            tfLite.run(readFileToBuffer(path), labelProb);
            return getTopAboveThreshold(labelProb[0], numResults, threshold);
        }

        public void close() {
            if (tfLite != null) tfLite.close();
        }

        private ByteBuffer readFileToBuffer(String path) throws IOException {
            Tensor tensor = tfLite.getInputTensor(0);
            int inputSize1 = tensor.shape()[1];
            int inputSize2 = tensor.shape()[2];
            int inputChannels = tensor.shape()[3];

            try (
                InputStream inputStream = new FileInputStream(path.replace("file://", ""));
            ) {
                Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

                Matrix matrix = new Matrix();
                matrix.postScale(inputSize1 / (float) bitmapRaw.getWidth(), inputSize2 / (float) bitmapRaw.getHeight());
                matrix.invert(new Matrix());

                int[] intValues = new int[inputSize1 * inputSize2];
                ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize1 * inputSize2 * inputChannels);
                imgData.order(ByteOrder.nativeOrder());

                Bitmap bitmap = Bitmap.createBitmap(inputSize1, inputSize2, Bitmap.Config.ARGB_8888);
                final Canvas canvas = new Canvas(bitmap);
                canvas.drawBitmap(bitmapRaw, matrix, null);
                bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

                int pixel = 0;
                for (int i = 0; i < inputSize1; ++i) {
                    for (int j = 0; j < inputSize2; ++j) {
                        final int pixelValue = intValues[pixel++];
                        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                        imgData.put((byte) (pixelValue & 0xFF));
                    }
                }
                return imgData;
            } catch (Exception e) {
                throw new IOException("Failed to read image", e);
            }
        }
        
        private WritableArray getTopAboveThreshold(byte[] scores, int num, float threshold) {
            ArrayList<WritableMap> list = new ArrayList<>(scores.length);
            for (int i = 0; i < scores.length; ++i) {
                float confidence = (float)((scores[i] & 0xff) / 255f);
                if (confidence > threshold) {
                    WritableMap res = Arguments.createMap();
                    // res.putInt("index", i);
                    res.putString("label", labels.size() > i ? labels.get(i) : "unknown");
                    res.putDouble("confidence", confidence);
                    list.add(res);
                }
            }
            list.sort(
                new Comparator<WritableMap>() {
                    @Override
                    public int compare(WritableMap lhs, WritableMap rhs) {
                        return Double.compare(rhs.getDouble("confidence"), lhs.getDouble("confidence"));
                    }
                }
            );

            WritableArray results = Arguments.createArray();
            int recognitionsSize = Math.min(list.size(), num);
            for (int i = 0; i < recognitionsSize; ++i) {
                results.pushMap(list.get(i));
            }
            return results;
        }
    }

    private final ReactApplicationContext reactContext;
    private final Random rand = new Random();
    private final HashMap<String, LoadedModel> loadedModels = new HashMap<>();

    public AutomlVisionTfliteModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
    }

    @Override
    public String getName() {
        return "AutomlVisionTflite";
    }

    @ReactMethod
    public void loadModel(final String modelPath, final String labelsPath,
                            final Callback callback) {
        LoadedModel model = null;
        try (
            AssetManager assetManager = reactContext.getAssets();
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
        ) {
            MappedByteBuffer buffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.getStartOffset(),
                fileDescriptor.getDeclaredLength());

            model = new LoadedModel(buffer);
            try (
                InputStream is = assetManager.open(labelsPath);
            ) {
                model.readLabels(is);
            }

            String id = String.format("%019d", rand.nextLong());
            loadedModels.put(id, model);

            callback.invoke(null, id);
        } catch (Exception e) {
            if (model != null) model.close();
            String msg = e.getMessage();
            callback.invoke(msg != null ? msg : "Failed to load model", null);
        }
    }

    @ReactMethod
    public void runModelOnImage(final String modelId, final String path, final int numResults, final float threshold,
                                    final Callback callback) {
        try {
            if (modelId == null || !loadedModels.containsKey(modelId)) {
                throw new Exception("No such modelId");
            }
            callback.invoke(null, loadedModels.get(modelId).run(path, numResults, threshold));
        } catch (Exception e) {
            String msg = e.getMessage();
            callback.invoke(msg != null ? msg : "Failed to run model", null);
        }
    }

    @ReactMethod
    public void close(final String modelId) {
        if (modelId != null && loadedModels.containsKey(modelId)) {
            loadedModels.remove(modelId).close();
        }
    }

}
