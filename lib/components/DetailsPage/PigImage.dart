import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:blackpig/utils/responsive.dart';
import 'package:blackpig/services/object_detection_service.dart';

class PigImage extends StatefulWidget {
  final String? imagePath;
  const PigImage({super.key, this.imagePath});

  @override
  State<PigImage> createState() => _PigImageState();
}

class _PigImageState extends State<PigImage> {
  double? _aspectRatio;
  bool _isLoading = true;
  List<Map<String, dynamic>> _detections = [];

  @override
  void initState() {
    super.initState();
    if (widget.imagePath != null) {
      _loadImageAspectRatio();
    } else {
      _isLoading = false;
    }
  }

  @override
  void didUpdateWidget(PigImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.imagePath != oldWidget.imagePath) {
      if (widget.imagePath != null) {
        _loadImageAspectRatio();
      } else {
        setState(() {
          _aspectRatio = null;
          _isLoading = false;
          _detections = [];
        });
      }
    }
  }

  Future<void> _loadImageAspectRatio() async {
    if (widget.imagePath == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      final file = File(widget.imagePath!);
      if (!await file.exists()) {
        setState(() {
          _aspectRatio = 16 / 9; // Default aspect ratio
          _isLoading = false;
        });
        return;
      }

      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();

      if (mounted) {
        final width = frame.image.width.toDouble();
        final height = frame.image.height.toDouble();
        setState(() {
          _aspectRatio = width / height;
        });
        frame.image.dispose();

        // Run object detection inference
        final detections = await PigObjectDetectionService()
            .detectObjectsInImage(widget.imagePath!);

        if (mounted) {
          setState(() {
            _detections = detections;
            _isLoading = false;
          });
        }
      }
    } catch (e) {
      // If we can't load the image, use a default aspect ratio
      if (mounted) {
        setState(() {
          _aspectRatio = 16 / 9; // Default aspect ratio
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: ResponsiveUtils.responsivePadding(context, horizontal: 28),
      child: _isLoading
          ? Container(
              width: ResponsiveUtils.width(context, 90),
              height: ResponsiveUtils.height(context, 32),
              decoration: BoxDecoration(
                color: const Color(0xFFD9D9D9),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(
                child: CircularProgressIndicator(),
              ),
            )
          : _aspectRatio != null
              ? AspectRatio(
                  aspectRatio: _aspectRatio!,
                  child: CustomPaint(
                    foregroundPainter: BoundingBoxPainter(_detections),
                    child: Container(
                      width: ResponsiveUtils.width(context, 90),
                      decoration: BoxDecoration(
                        color: const Color(0xFFD9D9D9),
                        borderRadius: BorderRadius.circular(12),
                        image: widget.imagePath != null
                            ? DecorationImage(
                                image: FileImage(File(widget.imagePath!)),
                                fit: BoxFit.cover,
                              )
                            : null,
                      ),
                    ),
                  ),
                )
              : Container(
                  width: ResponsiveUtils.width(context, 90),
                  height: ResponsiveUtils.height(context, 32),
                  decoration: BoxDecoration(
                    color: const Color(0xFFD9D9D9),
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
    );
  }
}

class BoundingBoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;

  BoundingBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color.fromARGB(255, 62, 255, 62)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final textStyle = const TextStyle(
      color: Color.fromARGB(255, 62, 255, 62),
      fontSize: 18,
      fontWeight: FontWeight.bold,
      backgroundColor: Colors.black54,
    );

    for (var detection in detections) {
      final rectMap = detection['rect'];
      final score = detection['score'] as double;
      // Object detection coordinate order is usually [ymin, xmin, ymax, xmax]
      // Because outputs were normalized 0-1 we must multiply by Canvas width and height
      final yMin = rectMap['yMin'];
      final xMin = rectMap['xMin'];
      final yMax = rectMap['yMax'];
      final xMax = rectMap['xMax'];

      // Add a sanity check if boxes are not normalized
      final multiplierX = xMax > 2 ? 1.0 : size.width;
      final multiplierY = yMax > 2 ? 1.0 : size.height;

      final rect = Rect.fromLTRB(
        xMin * multiplierX,
        yMin * multiplierY,
        xMax * multiplierX,
        yMax * multiplierY,
      );

      canvas.drawRect(rect, paint);

      final textSpan = TextSpan(
        text: ' Pig ${(score * 100).toStringAsFixed(1)}% ',
        style: textStyle,
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(
            xMin * multiplierX, (yMin * multiplierY) - textPainter.height - 2),
      );
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
