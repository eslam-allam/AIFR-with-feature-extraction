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
      body: Container(
        constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width,
            maxHeight: MediaQuery.of(context).size.height),
        decoration: const BoxDecoration(
          image: DecorationImage(
            fit: BoxFit.fill,
            image: AssetImage(
              '/home/eslamallam/Python/AIFR-with-feature-extraction/app-demo/client/flutter/demo_gui_flutter/images/add_image_background.jpg',
            ),
          ),
        ),
        width: MediaQuery.of(context).size.width,
        alignment: Alignment.center,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            Expanded(
              flex: 4,
              child: Column(
                children: [
                  const Padding(
                    padding: EdgeInsets.only(top: 20, bottom: 40),
                    child: Text(
                      'Model Training preferences',
                      textScaleFactor: 3,
                      style: TextStyle(
                        fontFamily: 'moon',
                        color: Colors.deepPurpleAccent,
                      ),
                    ),
                  ),
                  FractionallySizedBox(
                    widthFactor: 1,
                    child: CustomCheckBoxGroup(
                      buttonTextStyle: const ButtonTextStyle(
                        selectedColor: Colors.red,
                        unSelectedColor: Colors.orange,
                        textStyle: TextStyle(
                          fontSize: 16,
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
                      width: 150,
                      horizontal: false,
                      enableButtonWrap: true,
                      absoluteZeroSpacing: false,
                      selectedColor: Colors.white,
                      padding: 10,
                      enableShape: true,
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(top: 40),
                    child: Row(
                      children: [
                        Expanded(
                          child: Column(
                            children: [
                              const Text('Accuracy Threshold'),
                              Slider(
                                label: _accuracyThreshold.toStringAsFixed(2),
                                value: _accuracyThreshold,
                                min: 0.0,
                                max: 100.0,
                                divisions: 1000,
                                onChanged: (value) {
                                  setState(
                                    () {
                                      _accuracyThreshold = double.parse(
                                          value.toStringAsFixed(2));
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
                                label: _initialDropout.toStringAsFixed(2),
                                value: _initialDropout,
                                min: 0.0,
                                max: 0.6,
                                divisions: 6,
                                onChanged: (value) {
                                  setState(
                                    () {
                                      _initialDropout = double.parse(
                                          value.toStringAsFixed(2));
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
                    padding: const EdgeInsets.only(top: 40),
                    child: Row(
                      children: [
                        Expanded(
                          child: Column(
                            children: [
                              const Text('KNN Neighbors'),
                              Slider(
                                label: _knnNeighbors.toStringAsFixed(2),
                                value: _knnNeighbors,
                                min: 1,
                                max: 10,
                                divisions: 9,
                                onChanged: (value) {
                                  setState(
                                    () {
                                      _knnNeighbors = double.parse(
                                          value.toStringAsFixed(2));
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
                                label: _variableDropout.toStringAsFixed(2),
                                value: _variableDropout,
                                min: 0.0,
                                max: 0.6,
                                divisions: 60,
                                onChanged: (value) {
                                  setState(
                                    () {
                                      _variableDropout = double.parse(
                                          value.toStringAsFixed(2));
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
                  Form(
                    key: _formKey,
                    child: Column(
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(left: 20, top: 20),
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
                          padding: const EdgeInsets.only(left: 20, top: 20),
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
                  TwoButtonRow(
                      _formKey,
                      _nameController,
                      _ageController,
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
            const Expanded(
              flex: 6,
              child: CameraBoxWithButtons(),
            ),
          ],
        ),
      ),
    );
  }
}

class CameraBoxWithButtons extends StatelessWidget {
  const CameraBoxWithButtons({
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ClayContainer(
      color: const Color(0xff121212),
      borderRadius: 10,
      child: FractionallySizedBox(
        alignment: Alignment.centerRight,
        widthFactor: 1,
        child: Stack(
          alignment: Alignment.topLeft,
          children: [
            const Padding(
              padding: EdgeInsets.all(30),
              child: CameraFeed(),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 40.0, left: 40.0),
              child: IconButton(
                onPressed: () => captureAlert(context, null),
                icon: const Icon(
                  Icons.camera_alt,
                  color: Colors.black,
                ),
                iconSize: 60,
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
    this.formKey,
    this.nameController,
    this.ageController,
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
  final TextEditingController? nameController, ageController;
  final double knnNeighbors, variableDropout, accuracyThreshold, initialDropout;
  final bool loop, earlyStop, excelStats, variableKNN;
  final GlobalKey<FormState> formKey;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      flex: 1,
      child: FractionallySizedBox(
        heightFactor: 0.2,
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
                    if (!formKey.currentState!.validate()) {
                      // If the form is valid, display a snackbar. In the real world,
                      // you'd often call a server or save the information in a database.
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                            content: Text('Please fill the required fields')),
                      );
                    } else {
                      debugPrint(
                          'Name: ${nameController?.text}\nAge: ${ageController?.text}\nloop: $loop, Early Stop: $earlyStop\nExcel stats: $excelStats, Variable KNN: $variableKNN\n Accuracy Threshold: $accuracyThreshold, Initial Dropout: $initialDropout\nKNN neighbors: $knnNeighbors\nVariable Dropout: $variableDropout');
                    }
                  },
                  child: const Text(
                    'Train Model',
                    textScaleFactor: 1.7,
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
                  child: const Text(
                    'Save Model',
                    textScaleFactor: 1.7,
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
  late html.ImageElement _element;
  late DropzoneViewController controller;

  @override
  void initState() {
    _element = html.ImageElement()
      ..style.height = '100%'
      ..style.width = '100%'
      ..style.border = '0'
      ..style.cursor = 'None'
      ..src = "http://localhost:5000/video";

    // ignore:undefined_prefixed_name
    ui.platformViewRegistry.registerViewFactory(
      'CameraView',
      (int viewId) => _element,
    );

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
              final image = await controller.getFileData(ev);
              String name = await controller.getFilename(ev);

              captureAlert(context, image, fromfile: true, name: name);
            }),
      ],
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
  request.fields['name'] = name;
  request.fields['directory'] = '';
  request.files.add(pic);

  http.StreamedResponse responsestream = await request.send();

  if (responsestream.statusCode == 200) {
    http.Response response = await http.get(Uri.parse('$url&getresult=True'));
    return response;
  } else {
    return http.Response('', 403);
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
