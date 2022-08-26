import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:http_parser/http_parser.dart';
import 'package:clay_containers/widgets/clay_container.dart';
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:universal_html/html.dart' as html;
import 'package:demo_gui_flutter/select_dataset_page.dart';
import 'package:http/http.dart' as http;
import 'package:rflutter_alert/rflutter_alert.dart';
import 'package:flutter_dropzone/flutter_dropzone.dart';
import 'package:custom_radio_grouped_button/custom_radio_grouped_button.dart';

class AddImagePage extends StatefulWidget {
  const AddImagePage({super.key});

  @override
  State<AddImagePage> createState() => _AddImagePageState();
}

class _AddImagePageState extends State<AddImagePage> {
  double _accuracyThreshold = 89;
  double _variableDropout = 0.01;
  double _knnNeighbors = 5;
  double _initialDropout = 0.2;
  bool _loop = false,
      _earlyStop = true,
      _excelStats = true,
      _variableKNN = true;
  final _nameController = TextEditingController();
  final _ageController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  @override
  build(BuildContext context) {
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
          image: AssetImage(
            '/home/eslamallam/Python/AIFR-with-feature-extraction/app-demo/client/flutter/demo_gui_flutter/images/add_image_background.jpg',
          ),
        ),
      ),
      width: width,
      alignment: Alignment.center,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Expanded(
            flex: 4,
            child: Column(
              children: [
                Form(
                  key: _formKey,
                  child: Column(
                    children: [
                      Padding(
                        padding: EdgeInsets.only(left: edge20, top: edge20),
                        child: Align(
                          alignment: Alignment.centerLeft,
                          child: FractionallySizedBox(
                            widthFactor: 0.6,
                            child: TextNumberInput(
                              controller: _nameController,
                              label: 'Your Name:',
                              text: true,
                              maxlen: 15,
                            ),
                          ),
                        ),
                      ),
                      Padding(
                        padding: EdgeInsets.only(left: edge20, top: edge20),
                        child: Align(
                          alignment: Alignment.centerLeft,
                          child: FractionallySizedBox(
                            widthFactor: 0.6,
                            child: TextNumberInput(
                              controller: _ageController,
                              label: 'Age:',
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: EdgeInsets.only(top: edge20, bottom: edge20 * 3),
                  child: Text(
                    'Model Training preferences',
                    textScaleFactor: edge20 * 0.15,
                    style: const TextStyle(
                      fontFamily: 'moon',
                      color: Colors.deepPurpleAccent,
                    ),
                  ),
                ),
                FractionallySizedBox(
                  widthFactor: 1,
                  child: CustomCheckBoxGroup(
                    buttonTextStyle: ButtonTextStyle(
                      selectedColor: Colors.orange,
                      unSelectedColor: Colors.orange,
                      textStyle: TextStyle(
                        fontSize: edge20 * 0.8,
                      ),
                    ),
                    unSelectedColor: Theme.of(context).canvasColor,
                    defaultSelected: const [
                      "Early stop",
                      "Excel stats",
                      "Variable KNN",
                    ],
                    buttonLables: const [
                      "Loop",
                      "Early stop",
                      "Excel stats",
                      "Variable KNN",
                    ],
                    buttonValuesList: const [
                      "Loop",
                      "Early stop",
                      "Excel stats",
                      "Variable KNN",
                    ],
                    checkBoxButtonValues: (values) {
                      setState(() {
                        if (values.contains('Loop')) {
                          _loop = true;
                        } else {
                          _loop = false;
                        }
                        if (values.contains('Early stop')) {
                          _earlyStop = true;
                        } else {
                          _earlyStop = false;
                        }
                        if (values.contains('Excel stats')) {
                          _excelStats = true;
                        } else {
                          _excelStats = false;
                        }
                        if (values.contains('Variable KNN')) {
                          _variableKNN = true;
                        } else {
                          _variableKNN = false;
                        }
                      });
                    },
                    spacing: 0,
                    width: edge20 * 7.5,
                    horizontal: false,
                    enableButtonWrap: true,
                    absoluteZeroSpacing: false,
                    selectedColor: Colors.white38,
                    padding: edge20 * 0.5,
                    enableShape: true,
                  ),
                ),
                Padding(
                  padding: EdgeInsets.only(top: edge20 * 3),
                  child: Row(
                    children: [
                      Expanded(
                        child: Column(
                          children: [
                            const Text('Accuracy Threshold'),
                            Slider(
                              inactiveColor: Colors.orange.shade100,
                              activeColor: Colors.orange,
                              label: _accuracyThreshold.toStringAsFixed(2),
                              value: _accuracyThreshold,
                              min: 0.0,
                              max: 100.0,
                              divisions: 1000,
                              onChanged: (value) {
                                setState(
                                  () {
                                    _accuracyThreshold =
                                        double.parse(value.toStringAsFixed(2));
                                  },
                                );
                              },
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: Column(
                          children: [
                            const Text('Initial Dropout'),
                            Slider(
                              inactiveColor: Colors.orange.shade100,
                              activeColor: Colors.orange,
                              label: _initialDropout.toStringAsFixed(2),
                              value: _initialDropout,
                              min: 0.0,
                              max: 0.6,
                              divisions: 6,
                              onChanged: (value) {
                                setState(
                                  () {
                                    _initialDropout =
                                        double.parse(value.toStringAsFixed(2));
                                  },
                                );
                              },
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: EdgeInsets.only(top: edge20 * 3),
                  child: Row(
                    children: [
                      Expanded(
                        child: Column(
                          children: [
                            const Text('KNN Neighbors'),
                            Slider(
                              inactiveColor: Colors.orange.shade100,
                              activeColor: Colors.orange,
                              label: _knnNeighbors.toStringAsFixed(2),
                              value: _knnNeighbors,
                              min: 1,
                              max: 10,
                              divisions: 9,
                              onChanged: (value) {
                                setState(
                                  () {
                                    _knnNeighbors =
                                        double.parse(value.toStringAsFixed(2));
                                  },
                                );
                              },
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: Column(
                          children: [
                            const Text('Variable Dropout'),
                            Slider(
                              activeColor: Colors.orange,
                              inactiveColor: Colors.orange.shade100,
                              label: _variableDropout.toStringAsFixed(2),
                              value: _variableDropout,
                              min: 0.0,
                              max: 0.6,
                              divisions: 60,
                              onChanged: (value) {
                                setState(
                                  () {
                                    _variableDropout =
                                        double.parse(value.toStringAsFixed(2));
                                  },
                                );
                              },
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                TwoButtonRow(
                    _knnNeighbors,
                    _variableDropout,
                    _accuracyThreshold,
                    _initialDropout,
                    _loop,
                    _earlyStop,
                    _excelStats,
                    _variableKNN)
              ],
            ),
          ),
          Expanded(
            flex: 6,
            child:
                CameraBoxWithButtons(_formKey, _nameController, _ageController),
          ),
        ],
      ),
    );
  }
}

class CameraBoxWithButtons extends StatelessWidget {
  const CameraBoxWithButtons(
    this.formKey,
    this.nameController,
    this.ageController, {
    Key? key,
  }) : super(key: key);

  final TextEditingController nameController, ageController;
  final GlobalKey<FormState> formKey;

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    double height = MediaQuery.of(context).size.height;
    var padding = MediaQuery.of(context).viewPadding;
    double height_no_toolbar = height - padding.top - kToolbarHeight;

    double edge20 = (width * 0.01 + height_no_toolbar * 0.02) / 2;
    return ClayContainer(
      color: const Color(0xff121212),
      borderRadius: 10,
      child: FractionallySizedBox(
        alignment: Alignment.centerRight,
        widthFactor: 1,
        child: Stack(
          alignment: Alignment.topLeft,
          children: [
            Padding(
              padding: EdgeInsets.all(edge20 * 1.5),
              child: CameraFeed(formKey, nameController, ageController),
            ),
            Padding(
              padding: EdgeInsets.only(top: edge20 * 2, left: edge20 * 2),
              child: IconButton(
                onPressed: () {
                  if (!formKey.currentState!.validate()) {
                    // If the form is valid, display a snackbar. In the real world,
                    // you'd often call a server or save the information in a database.
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                          content: Text('Please fill the required fields')),
                    );
                  } else {
                    captureAlert(
                        context, null, nameController.text, ageController.text);
                  }
                },
                icon: const Icon(
                  Icons.camera_alt,
                  color: Colors.black,
                ),
                iconSize: edge20 * 3,
              ),
            )
          ],
        ),
      ),
    );
  }
}

class TwoButtonRow extends StatelessWidget {
  const TwoButtonRow(
    this.knnNeighbors,
    this.variableDropout,
    this.accuracyThreshold,
    this.initialDropout,
    this.loop,
    this.earlyStop,
    this.excelStats,
    this.variableKNN, {
    Key? key,
  }) : super(key: key);
  final double _buttonwidthfactor = 0.8;
  final double _buttonheightfactor = 0.8;
  final double _buttonborderradius = 50;

  final double knnNeighbors, variableDropout, accuracyThreshold, initialDropout;
  final bool loop, earlyStop, excelStats, variableKNN;

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    double edge20 = width * 0.01;
    return Expanded(
      flex: 1,
      child: FractionallySizedBox(
        heightFactor: 0.4,
        child: Row(
          mainAxisSize: MainAxisSize.max,
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            ButtonWrap(
              child: FractionallySizedBox(
                heightFactor: _buttonheightfactor,
                widthFactor: _buttonwidthfactor,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    primary: Colors.orange,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(
                          _buttonborderradius), //change border radius of this beautiful button thanks to BorderRadius.circular function
                    ),
                  ),
                  onPressed: () {
                    debugPrint(
                        'loop: $loop, Early Stop: $earlyStop\nExcel stats: $excelStats, Variable KNN: $variableKNN\n Accuracy Threshold: $accuracyThreshold, Initial Dropout: $initialDropout\nKNN neighbors: $knnNeighbors\nVariable Dropout: $variableDropout');
                  },
                  child: Text(
                    'Train Model',
                    textScaleFactor: edge20 * 0.085,
                  ),
                ),
              ),
            ),
            ButtonWrap(
              child: FractionallySizedBox(
                heightFactor: _buttonheightfactor,
                widthFactor: _buttonwidthfactor,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    primary: Colors.orange,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(
                          _buttonborderradius), //change border radius of this beautiful button thanks to BorderRadius.circular function
                    ),
                  ),
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (BuildContext context) {
                          return const SelectDatasetPage();
                        },
                      ),
                    );
                  },
                  child: Text(
                    'Save Model',
                    textScaleFactor: edge20 * 0.085,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

void captureAlert(
  BuildContext context,
  image,
  String name,
  String age, {
  bool fromfile = false,
}) async {
  http.Response response;
  int code;
  final String url =
      'http://localhost:5000/capture?save=False&name=${name.toString()}&age=${age.toString()}';
  if (!fromfile) {
    response = await http.get(Uri.parse(url));
  } else {
    assert(image != null, 'Passed image is null');

    response = await _asyncFileUpload(name, age, image, url);
  }

  code = response.statusCode;

  if (code == 403) {
    const snackBar = SnackBar(
      duration: Duration(seconds: 5),
      content: Text(
          'Face not detected! Please position your face at the middle of the screen and try again'),
    );
    // ignore: use_build_context_synchronously
    ScaffoldMessenger.of(context).showSnackBar(snackBar);
    return;
  } else if (code == 500) {
    const snackBar = SnackBar(
      duration: Duration(seconds: 5),
      content: Text('A server side error has occured. Please try again later.'),
    );
    // ignore: use_build_context_synchronously
    ScaffoldMessenger.of(context).showSnackBar(snackBar);
    return;
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
            onPressed: () async {
              response = await http.get(
                Uri.parse(
                  'http://localhost:5000/capture?save=True&name=$name&age=$age',
                ),
              );
              if (response.statusCode == 200) {
                const snackBar = SnackBar(
                  duration: Duration(seconds: 5),
                  content: Text('Image saved succesfully.'),
                );
                // ignore: use_build_context_synchronously
                ScaffoldMessenger.of(context).showSnackBar(snackBar);
              } else {
                const snackBar = SnackBar(
                  duration: Duration(seconds: 5),
                  content: Text('There was a problem saving your image.'),
                );
                // ignore: use_build_context_synchronously
                ScaffoldMessenger.of(context).showSnackBar(snackBar);
              }

              // ignore: use_build_context_synchronously
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
    double width = MediaQuery.of(context).size.width;
    double edge20 = width * 0.01;
    return Expanded(
      child: SizedBox(
        width: MediaQuery.of(context).size.width,
        child: Padding(padding: EdgeInsets.all(edge20 * 0.75), child: child),
      ),
    );
  }
}

class CameraFeed extends StatefulWidget {
  final TextEditingController nameController, ageController;
  final GlobalKey<FormState> formKey;
  const CameraFeed(this.formKey, this.nameController, this.ageController,
      {Key? key})
      : super(key: key);

  @override
  CameraFeedState createState() => CameraFeedState();
}

class CameraFeedState extends State<CameraFeed> {
  late html.ImageElement _element;
  late DropzoneViewController controller;

  @override
  void initState() {
    _element = html.ImageElement()
      ..style.height = '100%'
      ..style.width = '100%'
      ..style.border = '0'
      ..style.cursor = 'None'
      ..setAttribute('alt', 'Image not found. API must have crashed.')
      ..setAttribute('onerror',
          "this.onerror=null;this.src='images/facial-recognition-connected-real-estate.png';")
      ..src = "http://localhost:5000/video";

    // ignore:undefined_prefixed_name
    ui.platformViewRegistry
        .registerViewFactory('CameraView', (int viewId) => _element);

    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        const HtmlElementView(viewType: 'CameraView'),
        DropzoneView(
            operation: DragOperation.copy,
            onCreated: (DropzoneViewController ctrl) => controller = ctrl,
            onDrop: (dynamic ev) async {
              if (!widget.formKey.currentState!.validate()) {
                // If the form is valid, display a snackbar. In the real world,
                // you'd often call a server or save the information in a database.
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                      content: Text('Please fill the required fields')),
                );
              } else {
                final image = await controller.getFileData(ev);

                // ignore: use_build_context_synchronously
                captureAlert(context, image, widget.nameController.text,
                    widget.ageController.text,
                    fromfile: true);
              }
            }),
      ],
    );
  }
}

Future _asyncFileUpload(
    String name, String age, Uint8List image, String url) async {
  //create multipart request for POST or PATCH method
  var request = http.MultipartRequest("POST", Uri.parse(url));
  //add text fields

  //create multipart using filepath, string or bytes
  var pic = http.MultipartFile.fromBytes('files.myImage', image,
      contentType: MediaType.parse('image/jpeg'), filename: 'image.jpg');
  //add multipart to request

  request.fields['directory'] = '';
  request.files.add(pic);

  http.StreamedResponse responsestream = await request.send();

  if (responsestream.statusCode == 200) {
    http.Response response = await http.get(Uri.parse('$url&getresult=True'));
    return response;
  } else {
    return http.Response('', responsestream.statusCode);
  }
}

class TextNumberInput extends StatelessWidget {
  const TextNumberInput(
      {Key? key,
      required this.label,
      this.controller,
      this.value,
      this.onChanged,
      this.error,
      this.icon,
      this.allowDecimal = false,
      this.text = false,
      this.maxlen = 2})
      : super(key: key);

  final TextEditingController? controller;

  final String? value;

  final String label;

  final Function? onChanged;

  final String? error;

  final Widget? icon;

  final bool allowDecimal;

  final bool text;

  final int maxlen;

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      validator: (value) {
        if (value == null || value.isEmpty) {
          String _errorMessage() =>
              text ? 'Please enter your name' : 'Please enter your age';
          return _errorMessage();
        }
        return null;
      },
      maxLength: maxlen,
      controller: controller,
      initialValue: value,
      onChanged: onChanged as void Function(String)?,
      readOnly: false,
      keyboardType: text
          ? TextInputType.text
          : TextInputType.numberWithOptions(decimal: allowDecimal),
      inputFormatters: <TextInputFormatter>[
        FilteringTextInputFormatter.allow(RegExp(_getRegexString())),
        TextInputFormatter.withFunction(
          (oldValue, newValue) => newValue.copyWith(
            text: newValue.text.replaceAll('.', ','),
          ),
        ),
      ],
      decoration: InputDecoration(
        label: Text(label),
        errorText: error,
        icon: icon,
      ),
    );
  }

  String _getRegexString() => text
      ? r'[aA-zZ]+'
      : allowDecimal
          ? r'[0-9]+[,.]{0,1}[0-9]*'
          : r'[0-9]';
}
