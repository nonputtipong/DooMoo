import 'package:flutter/material.dart';
import 'package:flutter_svg/svg.dart';
import 'package:image_picker/image_picker.dart';
import 'package:blackpig/pages/details.dart';
import 'package:blackpig/utils/responsive.dart';
import 'package:blackpig/utils/camera_metadata.dart';

class Upload extends StatelessWidget {
  const Upload({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: ResponsiveUtils.width(context, 90),
      height: ResponsiveUtils.height(context, 30),
      decoration: BoxDecoration(
        color: const Color(0xFFFCFCFC),
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Color.fromARGB(25, 0, 0, 0),
            blurRadius: 4,
            spreadRadius: 1,
            offset: Offset(0, 2), // Shadow position
          )
        ],
      ),
      child: Stack(
        children: [
          Align(
            alignment: Alignment.topCenter,
            child: Column(
              children: [
                uploadLogo(context),
                text(context),
                button(context),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Padding button(BuildContext context) {
    return Padding(
        padding: const EdgeInsets.only(top: 8),
        child: GestureDetector(
          onTap: () async {
            final picker = ImagePicker();
            final XFile? xfile = await picker.pickImage(
              source: ImageSource.gallery,
              maxWidth: 2048,
              imageQuality: 90,
            );
            if (xfile != null && context.mounted) {
              // Extract camera metadata from the uploaded image
              final metadata = await CameraMetadataExtractor.extractFromImage(xfile.path);
              
              // Debug: Print metadata (you can remove this later)
              print('Camera Metadata: $metadata');
              
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => DetailsPage(
                    imagePath: xfile.path,
                    cameraMetadata: metadata,
                  ),
                ),
              );
            }
          },
          child: Container(
            width: ResponsiveUtils.width(context, 60),
            height: ResponsiveUtils.height(context, 7),
            decoration: BoxDecoration(
                color: Color(0xFFFFFFFF),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Color(0xFF2671F4), width: 2),
                boxShadow: [
                  BoxShadow(
                    color: Color.fromARGB(25, 0, 0, 0),
                    blurRadius: 4,
                    offset: Offset(0, 4), // Shadow position
                  )
                ]),
            child: Center(
              child: Text(
                'คลิก',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 24),
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF2671F4),
                ),
              ),
            ),
          ),
        ));
  }

  Padding text(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(top: 3),
      child: Text(
        'อัพโหลดรูปหมู',
        style: TextStyle(
          fontSize: ResponsiveUtils.fontSize(context, 34),
          fontWeight: FontWeight.bold,
          color: Color(0xFF5A5A5A),
        ),
      ),
    );
  }

  Padding uploadLogo(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 30),
      child: SvgPicture.asset(
        'assets/icons/Upload.svg',
        width: ResponsiveUtils.width(context, 20),
        height: ResponsiveUtils.width(context, 20),
      ),
    );
  }
}
