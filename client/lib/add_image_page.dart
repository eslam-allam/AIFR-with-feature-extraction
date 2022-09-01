import 'package:clay_containers/widgets/clay_container.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:client/widgets.dart';
import 'package:custom_radio_grouped_button/custom_radio_grouped_button.dart';
import 'package:file_picker/file_picker.dart';

// ignore: must_be_immutable
class AddImagePage extends StatefulWidget {
  const AddImagePage({super.key});

  @override
  State<AddImagePage> createState() => _AddImagePageState();
}

class _AddImagePageState extends State<AddImagePage> {
  double _accuracyThreshold = 0;
  double _variableDropout = 0.01;
  double _knnNeighbors = 3;
  double _initialDropout = 0.32;
  bool _loop = false,
      _earlyStop = true,
      _excelStats = true,
      _variableKNN = true;
  final _nameController = TextEditingController();
  final _ageController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool _dragging = false;
  bool _imageLoaded = true;
  final dynamic _imageFilesList = [];
  final List<Uint8List> _imageFilesBytesList = [];

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    double height = MediaQuery.of(context).size.height;
    var padding = MediaQuery.of(context).viewPadding;
    double heightNoToolbar = height - padding.top - kToolbarHeight;
    double edge20 = (width * 0.01 + heightNoToolbar * 0.02) / 2;

    getImageList(String type, List files) async {
      for (int i = 0; i < files.length; i++) {
        Uint8List currentImage = Uint8List(1);
        if (type == 'xfile') {
          currentImage = await files[i].readAsBytes();
        } else {
          currentImage = files[i].bytes;
        }
        _imageFilesBytesList.add(currentImage);
      }
      setState(() {});
    }

    return Container(
      constraints: BoxConstraints(maxWidth: width, maxHeight: heightNoToolbar),
      decoration: const BoxDecoration(
        image: DecorationImage(
          fit: BoxFit.fill,
          image: AssetImage(
            '/home/eslamallam/Python/AIFR_with_feature_extraction/client/images/add_image_background.jpg',
          ),
        ),
      ),
      width: width,
      alignment: Alignment.center,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Expanded(
            flex: 7,
            child: FractionallySizedBox(
              widthFactor: 0.9,
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
                    padding:
                        EdgeInsets.only(top: edge20 * 2, bottom: edge20 * 3),
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
          ),
          Expanded(
            flex: 9,
            child: Container(
              height: heightNoToolbar,
              decoration: BoxDecoration(
                  borderRadius: const BorderRadius.all(Radius.circular(10)),
                  color: !_dragging
                      ? Colors.grey.withOpacity(0.5)
                      : Colors.grey.withOpacity(1),
                  image: DecorationImage(
                      image: _imageFilesList.length == 0
                          ? const AssetImage(
                              'images/Drag_images_here_stripped.png')
                          : const AssetImage('images/empty.png'))),
              child: Stack(
                alignment: Alignment.topRight,
                children: [
                  FractionallySizedBox(
                    heightFactor: 1,
                    child: Padding(
                      padding: EdgeInsets.only(
                          bottom: edge20 * 0.5, top: edge20 * 0.5),
                      child: DropTarget(
                        onDragDone: (detail) {
                          setState(() {
                            _imageLoaded = false;
                            _imageFilesList.addAll(detail.files);
                            getImageList('xfile', detail.files);
                            _imageLoaded = true;
                          });
                        },
                        onDragEntered: (detail) {
                          setState(() {
                            _dragging = true;
                          });
                        },
                        onDragExited: (detail) {
                          setState(() {
                            _dragging = false;
                          });
                        },
                        child: ListView.builder(
                          itemExtent: edge20 * 10,
                          scrollDirection: Axis.vertical,
                          shrinkWrap: true,
                          itemCount: _imageFilesList.length,
                          itemBuilder: (context, index) {
                            return Card(
                              elevation: 8.0,
                              margin: EdgeInsets.symmetric(
                                  horizontal: edge20 * 0.5,
                                  vertical: edge20 * 0.3),
                              child: Container(
                                alignment: Alignment.center,
                                decoration: const BoxDecoration(
                                    borderRadius:
                                        BorderRadius.all(Radius.circular(10)),
                                    color: Color.fromRGBO(64, 75, 96, .9)),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  crossAxisAlignment: CrossAxisAlignment.center,
                                  children: [
                                    Expanded(
                                      flex: 2,
                                      child: Container(
                                        padding: EdgeInsets.only(
                                            right: edge20,
                                            left: edge20,
                                            top: edge20,
                                            bottom: edge20),
                                        decoration: BoxDecoration(
                                          border: Border(
                                            right: BorderSide(
                                                width: edge20 / 20,
                                                color: Colors.white24),
                                          ),
                                        ),
                                        child: _imageFilesBytesList.length >=
                                                index + 1
                                            ? Container(
                                                decoration: BoxDecoration(
                                                  border: Border.all(
                                                      width: 3,
                                                      color:
                                                          const Color.fromARGB(
                                                              255,
                                                              150,
                                                              148,
                                                              148)),
                                                  borderRadius:
                                                      const BorderRadius.all(
                                                          Radius.circular(20)),
                                                  image: DecorationImage(
                                                      image: MemoryImage(
                                                        _imageFilesBytesList[
                                                            index],
                                                      ),
                                                      fit: BoxFit.fitWidth,
                                                      alignment:
                                                          Alignment.center),
                                                ),
                                              )
                                            : const CircularProgressIndicator(),
                                      ),
                                    ),
                                    Expanded(
                                      flex: 6,
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        mainAxisAlignment:
                                            MainAxisAlignment.spaceBetween,
                                        children: [
                                          Expanded(
                                            child: FractionallySizedBox(
                                              heightFactor: 0.3,
                                              alignment: Alignment.bottomRight,
                                              child: Padding(
                                                padding: EdgeInsets.only(
                                                    left: edge20),
                                                child: Text(
                                                  '${_imageFilesList[index].name}',
                                                  style: TextStyle(
                                                      color: Colors.white,
                                                      fontWeight:
                                                          FontWeight.bold,
                                                      fontSize: edge20 * 0.9),
                                                ),
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            child: Padding(
                                              padding: EdgeInsets.only(
                                                  left: edge20 * 0.55),
                                              child: FractionallySizedBox(
                                                heightFactor: 1,
                                                alignment: Alignment.topLeft,
                                                child: Padding(
                                                  padding:
                                                      const EdgeInsets.only(
                                                          bottom: 50),
                                                  child: Row(
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .center,
                                                    mainAxisAlignment:
                                                        MainAxisAlignment
                                                            .center,
                                                    children: <Widget>[
                                                      const Expanded(
                                                        flex: 1,
                                                        child: Icon(
                                                            Icons.folder,
                                                            color: Colors
                                                                .yellowAccent),
                                                      ),
                                                      Expanded(
                                                        flex: 10,
                                                        child: Padding(
                                                          padding:
                                                              EdgeInsets.only(
                                                                  top: edge20 *
                                                                      0.2),
                                                          child: Text(
                                                            _imageFilesList[
                                                                    index]
                                                                .path,
                                                            style: TextStyle(
                                                                color: Colors
                                                                    .white,
                                                                fontSize:
                                                                    edge20 *
                                                                        0.75),
                                                          ),
                                                        ),
                                                      )
                                                    ],
                                                  ),
                                                ),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    Expanded(
                                      flex: 5,
                                      child: FractionallySizedBox(
                                        widthFactor: 1,
                                        child: Padding(
                                          padding: EdgeInsets.only(
                                              right: edge20 * 2.5),
                                          child: Row(
                                            mainAxisAlignment:
                                                MainAxisAlignment.start,
                                            crossAxisAlignment:
                                                CrossAxisAlignment.center,
                                            children: [
                                              const Expanded(
                                                child: TextNumberInput(
                                                  label: 'Age:',
                                                  allowDecimal: false,
                                                  text: false,
                                                  maxlen: 2,
                                                ),
                                              ),
                                              Padding(
                                                padding: EdgeInsets.only(
                                                    left: edge20 * 2),
                                                child: IconButton(
                                                  onPressed: () {},
                                                  icon: Icon(
                                                    Icons.check_box,
                                                    color: Colors.green,
                                                    size: edge20 * 2,
                                                  ),
                                                ),
                                              )
                                            ],
                                          ),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    ),
                  ),
                  FractionallySizedBox(
                    widthFactor: 1,
                    heightFactor: 1,
                    alignment: Alignment.center,
                    child: FractionallySizedBox(
                      widthFactor: 0.1,
                      heightFactor: 0.1,
                      child: Visibility(
                        visible: !_imageLoaded,
                        child: CircularProgressIndicator(
                          color: Colors.white,
                          strokeWidth: edge20 * 0.5,
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding:
                        EdgeInsets.only(top: edge20 * 0.8, right: edge20 * 0.8),
                    child: ElevatedButton.icon(
                        style: ButtonStyle(
                          backgroundColor:
                              const MaterialStatePropertyAll(Colors.white),
                          minimumSize: MaterialStatePropertyAll(
                            Size(edge20, edge20 * 3),
                          ),
                        ),
                        onPressed: (() async {
                          setState(() {
                            _imageLoaded = false;
                          });

                          FilePickerResult? result =
                              await FilePicker.platform.pickFiles(
                            withData: true,
                            allowMultiple: true,
                            type: FileType.custom,
                            allowedExtensions: ['jpg', 'png', 'JPG', 'PNG'],
                          );
                          if (result != null) {
                            List<PlatformFile> files = result.files;

                            setState(() {
                              _imageFilesList.addAll(files);
                              getImageList('platformfile', files);
                            });
                          } else {
                            // User canceled the picker
                          }
                          setState(() {
                            _imageLoaded = true;
                          });
                        }),
                        label: const Text('Pick Images'),
                        icon: const Icon(Icons.folder)),
                  ),
                ],
              ),
            ),
          )
        ],
      ),
    );
  }
}
