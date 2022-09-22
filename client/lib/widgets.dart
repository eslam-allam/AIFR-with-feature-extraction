import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_mjpeg/flutter_mjpeg.dart';
import 'package:http_parser/http_parser.dart';
import 'package:clay_containers/widgets/clay_container.dart';
// ignore: depend_on_referenced_packages
import 'package:http/http.dart' as http;
import 'package:rflutter_alert/rflutter_alert.dart';
import 'package:flutter/material.dart';

class VideoFeed extends StatelessWidget {
  const VideoFeed({
    Key? key,
    required this.heightNoToolbar,
    required this.edge20,
    required this.isRunning,
  }) : super(key: key);

  final double heightNoToolbar;
  final double edge20;
  final bool isRunning;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      flex: 10,
      child: Mjpeg(
        error: (context, error, stack) {
          return Image.asset(
            'images/video_error.png',
            fit: BoxFit.fill,
          );
        },
        fit: BoxFit.contain,
        isLive: isRunning,
        stream: 'http://localhost:5000/video',
      ),
    );
  }
}

class TwoButtonRow extends StatefulWidget {
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
  State<TwoButtonRow> createState() => _TwoButtonRowState();
}

class _TwoButtonRowState extends State<TwoButtonRow> {
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
                heightFactor: widget._buttonheightfactor,
                widthFactor: widget._buttonwidthfactor,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(widget
                          ._buttonborderradius), //change border radius of this beautiful button thanks to BorderRadius.circular function
                    ),
                  ),
                  onPressed: () async {
                    bool autosave;
                    if (widget.accuracyThreshold == 0) {
                      autosave = false;
                    } else {
                      autosave = true;
                    }
                    debugPrint(
                        'loop: ${widget.loop}, Early Stop: ${widget.earlyStop}\nExcel stats: ${widget.excelStats}, Variable KNN: ${widget.variableKNN}\n Accuracy Threshold: ${widget.accuracyThreshold}, Initial Dropout: ${widget.initialDropout}\nKNN neighbors: ${widget.knnNeighbors}\nVariable Dropout: ${widget.variableDropout}\nautosave: $autosave');
                    http.Response response = await http.get(Uri.parse(
                        'http://localhost:5000/trainmodel?loop=${widget.loop}&es=${widget.earlyStop}&estats=${widget.excelStats}&vknn=${widget.variableKNN}&at=${widget.accuracyThreshold}&dropout=${widget.initialDropout}&knn=${widget.knnNeighbors}&vdropout=${widget.variableDropout}&autosave=$autosave&save=False'));
                    if (response.statusCode == 200) {
                      String data = response.body;
                      // ignore: use_build_context_synchronously
                      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                        content: Text(data),
                        duration: const Duration(seconds: 10),
                      ));
                    } else {
                      // ignore: use_build_context_synchronously
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          duration: Duration(seconds: 10),
                          content: Text('An error occured during training'),
                        ),
                      );
                    }
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
                heightFactor: widget._buttonheightfactor,
                widthFactor: widget._buttonwidthfactor,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(widget
                          ._buttonborderradius), //change border radius of this beautiful button thanks to BorderRadius.circular function
                    ),
                  ),
                  onPressed: () async {
                    http.Response response = await http.get(Uri.parse(
                        'http://localhost:5000/trainmodel?loop=${widget.loop}&es=${widget.earlyStop}&estats=${widget.excelStats}&vknn=${widget.variableKNN}&at=${widget.accuracyThreshold}&dropout=${widget.initialDropout}&knn=${widget.knnNeighbors}&vdropout=${widget.variableDropout}&autosave=False&save=True'));
                    if (response.statusCode == 200) {
                      String data = response.body;
                      // ignore: use_build_context_synchronously
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(data),
                          duration: const Duration(seconds: 10),
                        ),
                      );
                    } else {
                      // ignore: use_build_context_synchronously
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                        content: Text('An error occured during saving'),
                        duration: Duration(seconds: 10),
                      ));
                    }
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

    response = await asyncFileUpload(name, age, image, url);
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

Future asyncFileUpload(
    String name, String age, Uint8List image, String url) async {
  //create multipart request for POST or PATCH method
  var request = http.MultipartRequest("POST", Uri.parse(url));
  //add text fields

  //create multipart using filepath, string or bytes
  var pic = http.MultipartFile.fromBytes('files.myImage', image,
      contentType: MediaType.parse('image/jpeg'), filename: name);
  //add multipart to request

  request.files.add(pic);

  http.StreamedResponse responsestream = await request.send();

  return responsestream;
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
          String errorMessage() =>
              text ? 'Please enter your name' : 'Please enter your age';
          return errorMessage();
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
