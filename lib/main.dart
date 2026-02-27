import 'package:flutter/material.dart';
import 'package:blackpig/pages/home.dart';
import 'package:blackpig/utils/camera_metadata.dart';
import 'package:blackpig/services/object_detection_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize camera hardware metadata cache on first launch
  await CameraMetadataCache.initializeHardwareMetadata();

  // Initialize ML Model asynchronously to avoid blocking the app startup
  PigObjectDetectionService().initialize();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(fontFamily: 'DB HelvethaicaMon X'),
      home: HomePage(),
    );
  }
}
