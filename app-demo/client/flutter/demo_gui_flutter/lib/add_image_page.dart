import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:universal_html/html.dart' as html;
import 'package:demo_gui_flutter/select_dataset_page.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class AddImagePage extends StatefulWidget {
  const AddImagePage({super.key});

  @override
  State<AddImagePage> createState() => _AddImagePageState();
}

class _AddImagePageState extends State<AddImagePage> {
  @override
  build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Add image to Dataset'),
        actions: [
          IconButton(
            onPressed: () {
              debugPrint('settings pressed');
            },
            icon: const Icon(
              Icons.settings,
            ),
          ),
        ],
      ),
      body: Row(
        mainAxisSize: MainAxisSize.max,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width / (3 / 2),
            ),
            child: const Padding(
              padding: EdgeInsets.all(8.0),
              child: CameraFeed(),
            ),
          ),
          Container(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width / 3,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Expanded(
                  child: SizedBox(
                    width: double.maxFinite,
                    child: Padding(
                      padding: const EdgeInsets.all(15),
                      child: ElevatedButton(
                        onPressed: () async {
                          String url = 'http://localhost:5000/capture';
                          var response = await http.get(Uri.parse(url));
                          var code = response.statusCode;
                          if (code == 403) {
                            const snackBar = SnackBar(
                              content: Text(
                                  'Face not detected! Please position your face at the middle of the screen and try again'),
                            );
                            ScaffoldMessenger.of(context)
                                .showSnackBar(snackBar);
                          }
                        },
                        child: const Text(
                          'Add Live Image',
                          textScaleFactor: 2.0,
                        ),
                      ),
                    ),
                  ),
                ),
                Expanded(
                  child: SizedBox(
                    width: double.maxFinite,
                    child: Padding(
                      padding: const EdgeInsets.all(15),
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (BuildContext context) {
                                return const AddImagePage();
                              },
                            ),
                          );
                        },
                        child: const Text(
                          'Add directory',
                          textScaleFactor: 2.0,
                        ),
                      ),
                    ),
                  ),
                ),
                Expanded(
                  child: SizedBox(
                    width: double.maxFinite,
                    child: Padding(
                      padding: const EdgeInsets.all(15),
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (BuildContext context) {
                                return const SelectDatasetPage();
                              },
                            ),
                          );
                        },
                        child: const Text(
                          'Process dataset',
                          textScaleFactor: 2.0,
                        ),
                      ),
                    ),
                  ),
                ),
                Expanded(
                  child: SizedBox(
                    width: double.maxFinite,
                    child: Padding(
                      padding: const EdgeInsets.all(15),
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (BuildContext context) {
                                return const SelectDatasetPage();
                              },
                            ),
                          );
                        },
                        child: const Text(
                          'Train Model',
                          textScaleFactor: 2.0,
                        ),
                      ),
                    ),
                  ),
                ),
                Expanded(
                  child: SizedBox(
                    width: double.maxFinite,
                    child: Padding(
                      padding: const EdgeInsets.all(15),
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (BuildContext context) {
                                return const SelectDatasetPage();
                              },
                            ),
                          );
                        },
                        child: const Text(
                          'Save Model',
                          textScaleFactor: 2.0,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class CameraFeed extends StatefulWidget {
  const CameraFeed({Key? key}) : super(key: key);

  @override
  CameraFeedState createState() => CameraFeedState();
}

class CameraFeedState extends State<CameraFeed> {
  late html.IFrameElement _element;

  @override
  void initState() {
    _element = html.IFrameElement()
      ..style.height = '100%'
      ..style.width = '100%'
      ..style.border = '0'
      ..srcdoc = """
          <img src="http://localhost:5000/video" width = "97%" height="auto"/>
        """;

    // ignore:undefined_prefixed_name
    ui.platformViewRegistry.registerViewFactory(
      'CameraView',
      (int viewId) => _element,
    );

    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width / 1.5,
      ),
      child: const HtmlElementView(viewType: 'CameraView'),
    );
  }
}
