import 'package:flutter/material.dart';
import 'package:blackpig/utils/responsive.dart';

class PigInfo extends StatelessWidget {
  const PigInfo({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: ResponsiveUtils.responsivePadding(context, horizontal: 31),
      child: Container(
        width: ResponsiveUtils.width(context, 90),
        constraints: BoxConstraints(
          minHeight: ResponsiveUtils.height(context, 40),
        ),
        decoration: BoxDecoration(
          color: const Color.fromRGBO(252, 252, 252, 30),
          borderRadius: BorderRadius.circular(15),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 13,
              spreadRadius: 6,
              offset: Offset(0, 0),
            ),
          ],
        ),
        child: Padding(
          padding: ResponsiveUtils.responsivePadding(context, all: 32, top: 46),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'น้ำหนัก: ',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 35),
                  color: Color(0xFF5A5A5A),
                ),
              ),
              SizedBox(height: ResponsiveUtils.height(context, 1.5)),
              Text(
                'รอบอก: ',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 35),
                  color: Color(0xFF5A5A5A),
                ),
              ),
              SizedBox(height: ResponsiveUtils.height(context, 1.5)),
              Text(
                'ความยาวลำตัว: ',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 35),
                  color: Color(0xFF5A5A5A),
                ),
              ),
              SizedBox(height: ResponsiveUtils.height(context, 1.5)),
              Text(
                'ความกว้างลำตัว: ',
                style: TextStyle(
                  fontSize: ResponsiveUtils.fontSize(context, 35),
                  color: Color(0xFF5A5A5A),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}