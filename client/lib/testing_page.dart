import 'package:clay_containers/widgets/clay_container.dart';
import 'package:flutter/material.dart';
import 'package:client/widgets.dart';

const int itemCount = 20;

class TestingPage extends StatelessWidget {
  const TestingPage({super.key});

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    double height = MediaQuery.of(context).size.height;
    var padding = MediaQuery.of(context).viewPadding;
    double heightNoToolbar = height - padding.top - kToolbarHeight;
    double edge20 = (width * 0.01 + heightNoToolbar * 0.02) / 2;

    return Container(
      constraints: BoxConstraints(maxWidth: width, maxHeight: heightNoToolbar),
      decoration: const BoxDecoration(
        image: DecorationImage(
          fit: BoxFit.fill,
          opacity: 0.65,
          image: AssetImage(
            '/home/eslamallam/Python/AIFR_with_feature_extraction/client/images/add_image_background.jpg',
          ),
        ),
      ),
      width: width,
      alignment: Alignment.center,
      child: ClayContainer(
        width: width,
        height: heightNoToolbar,
        child: VideoFeed(
            heightNoToolbar: heightNoToolbar, edge20: edge20, isRunning: true),
      ),
    );
  }
}
