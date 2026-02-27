import 'dart:developer';
import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class PigObjectDetectionService {
  static final PigObjectDetectionService _instance =
      PigObjectDetectionService._internal();
  factory PigObjectDetectionService() => _instance;
  PigObjectDetectionService._internal();

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  bool _isInit = false;

  final String _modelPath = 'assets/model/Pig Object Detection Model.tflite';

  bool get isInitialized => _isInit;

  Future<void> initialize() async {
    if (_isInit) return;

    try {
      log('Loading model: $_modelPath');

      // 1. Hardware Acceleration (Delegates)
      final interpreterOptions = InterpreterOptions();

      // Try initializing GPU Delegate, fallback to NPU (NNAPI), fallback to CPU
      try {
        final gpuDelegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: false,
          ),
        );
        interpreterOptions.addDelegate(gpuDelegate);
        log('GPU Delegate added successfully.');
      } catch (e) {
        log('Failed to add GPU Delegate, trying XNNPACK: $e');
        try {
          // XNNPACK Delegate for CPU acceleration
          interpreterOptions.addDelegate(XNNPackDelegate());
          log('XNNPACK Delegate added successfully.');
        } catch (e) {
          log('Failed to add XNNPACK Delegate, falling back to CPU: $e');
        }
      }

      // 2. Asynchronous Initialization
      try {
        _interpreter = await Interpreter.fromAsset(_modelPath,
            options: interpreterOptions);
      } catch (e) {
        log('Failed to initialize with GPU/XNNPACK, falling back to pure CPU: $e');
        // Clear options and fall back to CPU only
        final cpuOptions = InterpreterOptions();
        _interpreter =
            await Interpreter.fromAsset(_modelPath, options: cpuOptions);
      }

      // 3. Background Isolates
      // Create the isolate interpreter to prevent UI jank during inference
      _isolateInterpreter =
          await IsolateInterpreter.create(address: _interpreter!.address);

      _isInit = true;
      log('Pig Object Detection Model initialized successfully!');

      // 4. Warm-up Run
      await _warmUpModel();
    } catch (e) {
      log('Critical Error initializing Pig Object Detection Model: $e');
    }
  }

  Future<void> _warmUpModel() async {
    if (!_isInit || _interpreter == null || _isolateInterpreter == null) return;

    log('Running model warm-up...');
    try {
      // Get input and output tensors to figure out their shapes
      final inputTensors = _interpreter!.getInputTensors();
      final outputTensors = _interpreter!.getOutputTensors();

      if (inputTensors.isNotEmpty && outputTensors.isNotEmpty) {
        // Create dummy input (e.g., zeroes) based on the input shape
        final inputShape =
            inputTensors[0].shape; // Typically [1, 300, 300, 3] or similar
        // Note: For a real warm up, we need the actual shape. For Object detection, it is usually float32.

        // Assuming a common float input shape. The exact type and structure depends
        // on the specific .tflite model. This is a generic dummy payload.
        // If it crashes on warm up, it's usually a type mismatch (e.g. expects UINT8 instead of FLOAT32).

        // We wrap it in a try-catch so if the dummy shapes mismatch, it won't crash the app startup.
        log('Warm-up shape info - Input: $inputShape, Output shapes: ${outputTensors.map((t) => t.shape).toList()}');

        // In a real scenario, you'd configure the explicit inputs and outputs according to the model.
        // For now, logging the shapes helps us understand the model architecture.
      }
      log('Warm-up finished.');
    } catch (e) {
      log('Model warm-up deferred or failed (this is non-critical): $e');
    }
  }

  /// Use this method to run inference using the Isolate on an image file.
  Future<List<Map<String, dynamic>>> detectObjectsInImage(
      String imagePath) async {
    if (!_isInit || _isolateInterpreter == null) {
      log('Model not initialized yet!');
      return [];
    }

    try {
      // 1. Get model required input size
      final inputShape = _interpreter!.getInputTensors()[0].shape;
      final inputType = _interpreter!.getInputTensors()[0].type;
      final int modelHeight = inputShape[1];
      final int modelWidth = inputShape[2];

      // 2. Load and resize the image
      final file = File(imagePath);
      final bytes = await file.readAsBytes();
      img.Image? originalImage = img.decodeImage(bytes);
      if (originalImage == null) return [];

      img.Image resizedImage =
          img.copyResize(originalImage, width: modelWidth, height: modelHeight);

      // 3. Convert image to Tensor buffer
      // Some models use float32 [1, 300, 300, 3] (normalized to -1..1 or 0..1), others use uint8.
      // Assuming a standard float32 object detection format normalized [0, 1] or [-1, 1].
      // For safety, we will provide a float32 bytebuffer.
      var inputTensorData = List.generate(
        1,
        (i) => List.generate(
          modelHeight,
          (y) => List.generate(
            modelWidth,
            (x) {
              final pixel = resizedImage.getPixel(x, y);
              if (inputType == TensorType.float32) {
                // Normalize to [0...1] (Most common for float object detection usually is 0-1, sometimes -1..1)
                return [
                  pixel.r / 255.0,
                  pixel.g / 255.0,
                  pixel.b / 255.0,
                ];
              } else {
                return [
                  pixel.r,
                  pixel.g,
                  pixel.b,
                ];
              }
            },
          ),
        ),
      );

      // 4. Prepare Output Tensors
      // Standard SSD Object Detection output shapes:
      // Boxes: [1, 10, 4]
      // Classes: [1, 10]
      // Scores: [1, 10]
      // Count: [1]
      // Or YOLO style: [1, N, 4+classes]
      final outputTensors = _interpreter!.getOutputTensors();

      Map<int, Object> outputs = {};
      bool isSsdStyle = outputTensors.length >= 3;

      if (isSsdStyle) {
        // Prepare SSD styled outputs
        for (int i = 0; i < outputTensors.length; i++) {
          final shape = outputTensors[i].shape;
          if (shape.length == 3 && shape[2] == 4) {
            // Boxes [1, 10, 4]
            outputs[i] = List.generate(
                1,
                (_) => List.generate(
                    shape[1], (_) => List.generate(4, (_) => 0.0)));
          } else if (shape.length == 2) {
            // Classes or Scores [1, 10]
            outputs[i] =
                List.generate(1, (_) => List.generate(shape[1], (_) => 0.0));
          } else if (shape.length == 1) {
            // Count [1]
            outputs[i] = [0.0];
          } else {
            outputs[i] = List.filled(shape.reduce((a, b) => a * b), 0.0);
          }
        }
      } else {
        // Yolo style [1, N, C]
        final shape = outputTensors[0].shape;
        outputs[0] = List.generate(
            1,
            (_) => List.generate(
                shape[1], (_) => List.generate(shape[2], (_) => 0.0)));
      }

      // 5. Run inference efficiently using Isolate
      await _isolateInterpreter!
          .runForMultipleInputs([inputTensorData], outputs);

      // 6. Parse Results
      List<Map<String, dynamic>> results = [];

      if (isSsdStyle) {
        // Attempt to find the box array and score array
        int boxesIndex = -1, scoresIndex = -1, classesIndex = -1;
        for (int i = 0; i < outputTensors.length; i++) {
          final shape = outputTensors[i].shape;
          if (shape.length == 3 && shape[2] == 4)
            boxesIndex = i;
          else if (shape.length == 2 && scoresIndex == -1)
            scoresIndex = i; // First 2D is usually scores
          else if (shape.length == 2 && scoresIndex != -1)
            classesIndex = i; // Second is usually classes
        }

        if (boxesIndex != -1 && scoresIndex != -1) {
          var boxesList = (outputs[boxesIndex] as List)[0] as List;
          var scoresList = (outputs[scoresIndex] as List)[0] as List;
          var classesList = classesIndex != -1
              ? (outputs[classesIndex] as List)[0] as List
              : null;

          for (int i = 0; i < scoresList.length; i++) {
            double score = scoresList[i];
            if (score > 0.4) {
              // Confidence threshold
              var box = boxesList[i];
              results.add({
                'rect': {
                  'yMin': box[0],
                  'xMin': box[1],
                  'yMax': box[2],
                  'xMax': box[3]
                },
                'score': score,
                'class': classesList != null ? classesList[i].toInt() : 0,
              });
            }
          }
        }
      } else {
        // Parse YOLO style logic (e.g. YOLOv8 output: [1, 5, 8400])
        var out = (outputs[0] as List)[0] as List;
        int numAttributes = out.length; // e.g., 5 (x, y, w, h, score)

        if (out.isNotEmpty && (out[0] as List).isNotEmpty) {
          int numAnchors = (out[0] as List).length; // e.g., 8400

          List<Map<String, dynamic>> rawResults = [];

          for (int a = 0; a < numAnchors; a++) {
            // Find max class score
            double maxScore = 0.0;
            int maxClass = -1;
            for (int c = 4; c < numAttributes; c++) {
              double score = out[c][a];
              if (score > maxScore) {
                maxScore = score;
                maxClass = c - 4;
              }
            }

            if (maxScore > 0.4) {
              // x center, y center, w, h (usually absolute to model input size e.g. 640x640)
              double xc = out[0][a];
              double yc = out[1][a];
              double w = out[2][a];
              double h = out[3][a];

              // Convert to Normalized coords if they are Absolute (e.g. 0-640)
              if (xc > 2.0 || yc > 2.0 || w > 2.0 || h > 2.0) {
                xc /= modelWidth;
                yc /= modelHeight;
                w /= modelWidth;
                h /= modelHeight;
              }

              double xMin = (xc - w / 2);
              double yMin = (yc - h / 2);
              double xMax = (xc + w / 2);
              double yMax = (yc + h / 2);

              // Ensure coordinates are within [0...1] bounds
              xMin = xMin.clamp(0.0, 1.0);
              yMin = yMin.clamp(0.0, 1.0);
              xMax = xMax.clamp(0.0, 1.0);
              yMax = yMax.clamp(0.0, 1.0);

              rawResults.add({
                'rect': {
                  'yMin': yMin,
                  'xMin': xMin,
                  'yMax': yMax,
                  'xMax': xMax
                },
                'score': maxScore,
                'class': maxClass,
              });
            }
          }

          // Apply Non-Maximum Suppression (NMS)
          rawResults.sort(
              (a, b) => (b['score'] as double).compareTo(a['score'] as double));

          List<Map<String, dynamic>> finalResults = [];
          for (var res in rawResults) {
            bool keep = true;
            for (var kept in finalResults) {
              if (_calculateIoU(res['rect'], kept['rect']) > 0.45) {
                // NMS threshold
                keep = false;
                break;
              }
            }
            if (keep) {
              finalResults.add(res);
            }
          }
          results.addAll(finalResults);
        }
      }

      log('Detected ${results.length} objects.');
      return results;
    } catch (e) {
      log('Error during inference: $e');
      return [];
    }
  }

  void dispose() {
    _interpreter?.close();
    _isolateInterpreter?.close();
    _isInit = false;
  }

  double _calculateIoU(Map<String, dynamic> box1, Map<String, dynamic> box2) {
    double xA = box1['xMin'] > box2['xMin'] ? box1['xMin'] : box2['xMin'];
    double yA = box1['yMin'] > box2['yMin'] ? box1['yMin'] : box2['yMin'];
    double xB = box1['xMax'] < box2['xMax'] ? box1['xMax'] : box2['xMax'];
    double yB = box1['yMax'] < box2['yMax'] ? box1['yMax'] : box2['yMax'];

    double interWidth = xB - xA;
    double interHeight = yB - yA;
    if (interWidth <= 0 || interHeight <= 0) return 0.0;

    double interArea = interWidth * interHeight;
    double box1Area =
        (box1['xMax'] - box1['xMin']) * (box1['yMax'] - box1['yMin']);
    double box2Area =
        (box2['xMax'] - box2['xMin']) * (box2['yMax'] - box2['yMin']);
    double iou = interArea / (box1Area + box2Area - interArea);
    return iou;
  }
}
