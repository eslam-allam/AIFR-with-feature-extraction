import 'dart:io';
import 'dart:typed_data';
import 'package:http_parser/http_parser.dart';
import 'package:clay_containers/widgets/clay_container.dart';
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:universal_html/html.dart' as html;
import 'package:demo_gui_flutter/select_dataset_page.dart';
import 'package:http/http.dart' as http;
import 'package:rflutter_alert/rflutter_alert.dart';
import 'package:flutter_dropzone/flutter_dropzone.dart';

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
            child: Stack(
              alignment: Alignment.bottomCenter,
              children: [
                const Padding(
                  padding: EdgeInsets.all(8.0),
                  child: CameraFeed(),
                ),
                Padding(
                  padding: const EdgeInsets.only(bottom: 8.0),
                  child: IconButton(
                      onPressed: () => captureAlert(context, null),
                      icon: const Icon(Icons.camera_alt)),
                )
              ],
            ),
          ),
          Container(
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width / 3,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                ButtonWrap(
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
                ButtonWrap(
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
                ButtonWrap(
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
              ],
            ),
          ),
        ],
      ),
    );
  }
}

void captureAlert(
  BuildContext context,
  image, {
  bool fromfile = false,
  String name = 'unnamed.jpg',
}) async {
  http.Response response;
  int code;
  String url = 'http://localhost:5000/capture?save=False';
  if (!fromfile) {
    response = await http.get(Uri.parse(url));
  } else {
    assert(image != null, 'Passed image is null');

    response = await _asyncFileUpload(name, image, url);
  }

  code = response.statusCode;

  if (code == 403) {
    const snackBar = SnackBar(
      duration: Duration(seconds: 5),
      content: Text(
          'Face not detected! Please position your face at the middle of the screen and try again'),
    );
    ScaffoldMessenger.of(context).showSnackBar(snackBar);
  } else {
    Alert(
        style: const AlertStyle(
            backgroundColor: Colors.white70, isCloseButton: false),
        context: context,
        desc: "Would you like to save this image?",
        image: Ink(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.white, width: 5),
            borderRadius: BorderRadius.circular(15),
            image: DecorationImage(image: MemoryImage(response.bodyBytes)),
          ),
          height: 224,
          width: 224,
        ),
        buttons: [
          DialogButton(
            color: Colors.red,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                    padding: const EdgeInsets.only(right: 10),
                    child: const Text('Delete')),
                const Icon(
                  Icons.delete_forever,
                  color: Colors.white,
                ),
              ],
            ),
            onPressed: () => Navigator.of(context, rootNavigator: true).pop(),
          ),
          DialogButton(
            color: Colors.green,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                    padding: const EdgeInsets.only(right: 15),
                    child: const Text('Save')),
                const Icon(
                  Icons.check_circle_sharp,
                  color: Colors.white,
                ),
              ],
            ),
            onPressed: () {
              http.get(
                Uri.parse(
                  'http://localhost:5000/capture?save=True',
                ),
              );
              Navigator.of(context, rootNavigator: true).pop();
            },
          ),
        ]).show();
  }
}

class ButtonWrap extends StatelessWidget {
  const ButtonWrap({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: SizedBox(
        width: double.maxFinite,
        child: Padding(padding: const EdgeInsets.all(15), child: child),
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
  late DropzoneViewController controller;

  @override
  void initState() {
    _element = html.IFrameElement()
      ..style.height = '102%'
      ..style.width = '100%'
      ..style.border = '0'
      ..style.cursor = 'None'
      ..srcdoc = """
          <img src="http://localhost:5000/video" width = "100%" height="auto%"/>
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
      constraints:
          BoxConstraints(maxHeight: MediaQuery.of(context).size.height),
      child: ClayContainer(
        borderRadius: 10,
        color: const Color(0xff121212),
        child: Stack(
          children: [
            const HtmlElementView(viewType: 'CameraView'),
            DropzoneView(
                operation: DragOperation.copy,
                onCreated: (DropzoneViewController ctrl) => controller = ctrl,
                onDrop: (dynamic ev) async {
                  final image = await controller.getFileData(ev);
                  String name = await controller.getFilename(ev);

                  captureAlert(context, image, fromfile: true, name: name);
                }),
          ],
        ),
      ),
    );
  }
}

_asyncFileUpload(String name, Uint8List image, String url) async {
  //create multipart request for POST or PATCH method
  var request = http.MultipartRequest("POST", Uri.parse(url));
  //add text fields

  //create multipart using filepath, string or bytes
  var pic = http.MultipartFile.fromBytes('files.myImage', image,
      contentType: MediaType.parse('image/jpeg'), filename: name);
  //add multipart to request
  request.files.add(pic);

  http.StreamedResponse responsestream = await request.send();

  if (responsestream.statusCode == 200) {
    http.Response response = await http.get(Uri.parse('$url&getresult=True'));
    return response;
  } else {
    return http.Response('', 403);
  }
}
