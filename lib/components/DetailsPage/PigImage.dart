import 'dart:io';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:blackpig/utils/responsive.dart';
import 'package:blackpig/services/pytorch_segmentation_service.dart';

class PigImage extends StatefulWidget {
  final String? imagePath;
  const PigImage({super.key, this.imagePath});

  @override
  State<PigImage> createState() => _PigImageState();
}

class _PigImageState extends State<PigImage> {
  double? _aspectRatio;
  bool _isLoading = true;
  ui.Image? _maskImage;
  int _segmentationGeneration = 0;

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
  void dispose() {
    _maskImage?.dispose();
    super.dispose();
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

      // Start AI Segmentation asynchronously
      _runSegmentation(bytes);

      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();

      if (mounted) {
        final width = frame.image.width.toDouble();
        final height = frame.image.height.toDouble();
        setState(() {
          _aspectRatio = width / height;
          _isLoading = false;
        });
        frame.image.dispose();
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

  Future<void> _runSegmentation(Uint8List bytes) async {
    final int generation = ++_segmentationGeneration;
    try {
      final service = PigSegmentationService();
      await service.initialize();

      if (generation != _segmentationGeneration) return;

      final result = await service.runSegmentation(bytes);
      if (generation != _segmentationGeneration) return;

      if (result != null && mounted) {
        ui.decodeImageFromPixels(
          result.rgbaPixels,
          result.maskWidth,
          result.maskHeight,
          ui.PixelFormat.rgba8888,
          (img) {
            if (generation != _segmentationGeneration) {
              img.dispose();
              return;
            }
            if (mounted) {
              setState(() {
                _maskImage?.dispose();
                _maskImage = img;
              });
            } else {
              img.dispose();
            }
          },
        );
      }
    } catch (e) {
      debugPrint("Segmentation failed: $e");
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
                    child: _maskImage != null
                        ? CustomPaint(
                            painter: SegmentationMaskPainter(_maskImage!),
                          )
                        : null,
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

class SegmentationMaskPainter extends CustomPainter {
  final ui.Image maskImage;

  SegmentationMaskPainter(this.maskImage);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..isAntiAlias = false
      ..filterQuality = FilterQuality.none;

    // Stretch the 120x120 mask perfectly over the image's bounding box
    canvas.drawImageRect(
      maskImage,
      Rect.fromLTWH(
          0, 0, maskImage.width.toDouble(), maskImage.height.toDouble()),
      Rect.fromLTWH(0, 0, size.width, size.height),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant SegmentationMaskPainter oldDelegate) {
    return oldDelegate.maskImage != maskImage;
  }
}
