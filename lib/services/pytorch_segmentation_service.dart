import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';
import 'package:path_provider/path_provider.dart';

import 'dart:async';

class PigSegmentationService {
  static const String _modelPathAsset = 'assets/model/rf detr segmentation.ptl';

  Module? _module;
  bool _isInitialized = false;
  Completer<void>? _initCompleter;

  bool get isInitialized => _isInitialized;

  /// Initializes the PyTorch Lite engine and loads the model from assets.
  Future<void> initialize() async {
    if (_isInitialized) return;
    if (_initCompleter != null) return await _initCompleter!.future;

    _initCompleter = Completer<void>();
    try {
      // flutter_pytorch_lite requires an absolute path to the file.
      // We must copy it from assets to a temporary system directory first.
      final assetData = await rootBundle.load(_modelPathAsset);
      final rawBytes = assetData.buffer.asUint8List();
      final tempDir = (await getTemporaryDirectory()).path;
      final filePath = '$tempDir/rf_detr_segmentation.ptl';
      final file = File(filePath);
      await file.writeAsBytes(rawBytes);

      // Load the model
      _module = await FlutterPytorchLite.load(filePath);
      _isInitialized = true;
      _initCompleter!.complete();
      debugPrint("PyTorch Lite Model loaded successfully from $filePath");
    } catch (e) {
      debugPrint("Failed to load PyTorch Lite model: $e");
      _initCompleter!.completeError(e);
      rethrow;
    } finally {
      _initCompleter = null;
    }
  }

  /// Runs segmentation on the provided image bytes.
  Future<SegmentationResult?> runSegmentation(Uint8List imageBytes) async {
    if (!_isInitialized || _module == null) {
      debugPrint("Model not initialized. Call initialize() first.");
      return null;
    }

    try {
      // 1. Preprocess: decode, resize to 480x480, and normalize float32.
      // This is heavy, so we run it in an isolate.
      final Float32List inputFloats =
          await compute(_preprocessImage, imageBytes);

      // 2. Prepare the Input Tensor
      final inputShape = Int64List.fromList([1, 3, 480, 480]);
      final inputTensor = Tensor.fromBlobFloat32(inputFloats, inputShape);
      final inputIValue = IValue.from(inputTensor);

      // 3. Run Inference
      final stopwatch = Stopwatch()..start();
      final outputIValue = await _module!.forward([inputIValue]);
      stopwatch.stop();
      debugPrint(
          "PyTorch Lite inference took: ${stopwatch.elapsedMilliseconds}ms");

      // 4. Extract outputs from the resulting Tuple (Classes, Boxes, Masks).
      final outputTuple = outputIValue.toTuple();
      final classesTensor = outputTuple[0].toTensor();
      // final boxesTensor = outputTuple[1].toTensor(); // (omitted as we don't need UI boxes for now)
      final masksTensor = outputTuple[2].toTensor();

      // Ensure they exist and have correct float arrays attached
      final classesFloats =
          Float32List.fromList(classesTensor.dataAsFloat32List);
      final masksFloats = Float32List.fromList(masksTensor.dataAsFloat32List);

      // 5. Postprocess the masks into an RGBA map (Isolate computation)
      final result = await compute(
          _postProcessSegmentation,
          _PostProcessData(
            classesTensor: classesFloats,
            masksTensor: masksFloats,
          ));

      return result;
    } catch (e) {
      debugPrint("Error running inference: $e");
      return null;
    }
  }

  /// Clean up memory
  Future<void> dispose() async {
    if (_module != null) {
      await _module!.destroy();
      _module = null;
    }
    _isInitialized = false;
  }
}

// ------------------------------------------------------------------------
// DATA CLASSES
// ------------------------------------------------------------------------

class _PostProcessData {
  final Float32List classesTensor;
  final Float32List masksTensor;

  _PostProcessData({
    required this.classesTensor,
    required this.masksTensor,
  });
}

class SegmentationResult {
  // A flat array of RGBA pixels (length = 120 * 120 * 4)
  final Uint8List rgbaPixels;
  final int maskWidth = 120;
  final int maskHeight = 120;

  SegmentationResult(this.rgbaPixels);
}

// ------------------------------------------------------------------------
// ISOLATE FUNCTIONS (Run on background thread to prevent UI freezing)
// ------------------------------------------------------------------------

Float32List _preprocessImage(Uint8List imageBytes) {
  // 1. Decode Image
  final originalImage = img.decodeImage(imageBytes);
  if (originalImage == null) {
    throw ArgumentError('Failed to decode image - unsupported or corrupt data');
  }

  // 2. Resize to required 480x480
  final resizedImage = img.copyResize(originalImage, width: 480, height: 480);

  // 3. Convert to Float32 Tensor [1, 3, 480, 480] (NCHW layout)
  final floatList = Float32List(1 * 3 * 480 * 480);

  // PyTorch ImageNet Normalization Stats
  const double meanR = 0.485;
  const double meanG = 0.456;
  const double meanB = 0.406;
  const double stdR = 0.229;
  const double stdG = 0.224;
  const double stdB = 0.225;

  int pixelIndex = 0;
  final channelStride = 480 * 480;

  for (int y = 0; y < 480; y++) {
    for (int x = 0; x < 480; x++) {
      final pixel = resizedImage.getPixel(x, y);

      // Normalize to 0-1
      final r = pixel.r / 255.0;
      final g = pixel.g / 255.0;
      final b = pixel.b / 255.0;

      // Apply Mean & Std
      floatList[pixelIndex] = (r - meanR) / stdR;
      floatList[channelStride + pixelIndex] = (g - meanG) / stdG;
      floatList[channelStride * 2 + pixelIndex] = (b - meanB) / stdB;

      pixelIndex++;
    }
  }

  return floatList;
}

SegmentationResult _postProcessSegmentation(_PostProcessData data) {
  final classesAsFloats = data.classesTensor;
  final masksAsFloats = data.masksTensor;

  // classes shape: [1, 200, 2]
  // masks shape: [1, 200, 120, 120]

  const int maskArea = 120 * 120;
  final rgbaPixels = Uint8List(maskArea * 4);

  // We loop over the 200 proposals
  for (int i = 0; i < 200; i++) {
    // Classes contain 2 logits representing (background, object).
    final logitClass0 = classesAsFloats[i * 2 + 0]; // Background?
    final logitClass1 = classesAsFloats[i * 2 + 1]; // Object?

    // If the object probability is higher, keep it.
    if (logitClass1 > logitClass0 && logitClass1 > 0.5) {
      final maskOffset = i * maskArea;

      for (int p = 0; p < maskArea; p++) {
        // DETR mask logits threshold
        if (masksAsFloats[maskOffset + p] > 0.0) {
          rgbaPixels[p * 4 + 0] = 255; // R
          rgbaPixels[p * 4 + 1] = 50; // G
          rgbaPixels[p * 4 + 2] = 50; // B
          rgbaPixels[p * 4 + 3] = 128; // A (50% opacity)
        }
      }
    }
  }

  return SegmentationResult(rgbaPixels);
}
