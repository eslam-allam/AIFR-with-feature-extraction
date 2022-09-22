import 'package:clay_containers/widgets/clay_container.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:client/widgets.dart';
import 'package:custom_radio_grouped_button/custom_radio_grouped_button.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;

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
  final List<TextEditingController> _ageControllers = [];
  bool _dragging = false;
  bool _imageLoaded = true;
  final List<String> _imageNameList = [];
  final List<String> _imagePathList = [];
  final List<Uint8List> _imageFilesBytesList = [];
  bool _validateName = true;
  final List<bool> _validateAge = [];
  final List<String> _inputAgeList = [];
  final List<bool> _limitExceeded = [];

  @override
  void dispose() {
    for (int i = 0; i < _ageControllers.length; i++) {
      _ageControllers[i].dispose();
    }
    _nameController.dispose();
    super.dispose();
  }

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
          opacity: 0.65,
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
                  Padding(
                    padding: EdgeInsets.only(left: edge20, top: edge20),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: FractionallySizedBox(
                        widthFactor: 0.6,
                        child: TextNumberInput(
                          error:
                              _validateName ? null : 'Please enter your name.',
                          controller: _nameController,
                          label: 'Your Name:',
                          text: true,
                          maxlen: 15,
                          onChanged: (String? newvalue) {
                            setState(() {
                              _validateName = true;
                            });
                          },
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding:
                        EdgeInsets.only(top: edge20 * 3, bottom: edge20 * 3),
                    child: Text(
                      'Model Training preferences',
                      textScaleFactor: edge20 * 0.15,
                      style: const TextStyle(
                        fontFamily: 'moon',
                        color: Colors.orange,
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
                      height: edge20 * 2,
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
                              Text(
                                'Accuracy Threshold',
                                style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: edge20 * 0.85),
                              ),
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
                              Text('Initial Dropout',
                                  style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: edge20 * 0.85)),
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
                    padding: EdgeInsets.only(top: edge20 * 2.5),
                    child: Row(
                      children: [
                        Expanded(
                          child: Column(
                            children: [
                              Text('KNN Neighbors',
                                  style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: edge20 * 0.85)),
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
                              Text('Variable Dropout',
                                  style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: edge20 * 0.85)),
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
                      image: _imageNameList.isEmpty
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
                            for (int i = 0; i < detail.files.length; i++) {
                              _imageNameList.add(detail.files[i].name);
                              _imagePathList.add(detail.files[i].path);
                              _ageControllers.add(TextEditingController());
                              _validateAge.add(true);
                              _inputAgeList.add('0');
                              _limitExceeded.add(false);
                            }

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
                          itemCount: _imageNameList.length,
                          itemBuilder: (context, index) {
                            return Card(
                              elevation: 8.0,
                              margin: EdgeInsets.symmetric(
                                  horizontal: edge20 * 0.5,
                                  vertical: edge20 * 0.3),
                              child: Container(
                                alignment: Alignment.center,
                                decoration: BoxDecoration(
                                    borderRadius: const BorderRadius.all(
                                        Radius.circular(10)),
                                    color: Colors.grey[850]),
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
                                                  _imageNameList[index],
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
                                                            _imagePathList[
                                                                index],
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
                                              Expanded(
                                                child: TextNumberInput(
                                                  error: _validateAge[index]
                                                      ? _limitExceeded[index]
                                                          ? 'Cannot have more than 6 \nimages with the same age.'
                                                          : null
                                                      : 'Please enter your age.',
                                                  onChanged:
                                                      (String? newvalue) {
                                                    setState(() {
                                                      _validateAge[index] =
                                                          true;
                                                      _limitExceeded[index] =
                                                          false;
                                                    });
                                                  },
                                                  controller:
                                                      _ageControllers[index],
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
                                                  onPressed: () {
                                                    String age =
                                                        _ageControllers[index]
                                                            .text;
                                                    String name =
                                                        _nameController.text;

                                                    _inputAgeList[index] =
                                                        '$name$age';

                                                    if (name.isEmpty |
                                                        age.isEmpty) {
                                                      setState(() {
                                                        if (name.isEmpty) {
                                                          _validateName = false;
                                                        }
                                                        if (age.isEmpty) {
                                                          _validateAge[index] =
                                                              false;
                                                        }
                                                      });
                                                    } else {
                                                      setState(() {
                                                        int numDuplicates =
                                                            _inputAgeList
                                                                .where((element) =>
                                                                    element.contains(
                                                                        name +
                                                                            age))
                                                                .length;
                                                        if (numDuplicates >=
                                                            2) {
                                                          switch (
                                                              numDuplicates) {
                                                            case 2:
                                                              int matchIndex =
                                                                  _inputAgeList
                                                                      .indexOf(
                                                                          name +
                                                                              age);
                                                              _imageNameList[
                                                                      matchIndex] =
                                                                  '${_nameController.text}A${age}a.jpg';
                                                              age = '${age}b';

                                                              break;
                                                            case 3:
                                                              age = '${age}c';
                                                              break;
                                                            case 4:
                                                              age = '${age}d';
                                                              break;
                                                            case 5:
                                                              age = '${age}e';
                                                              break;
                                                            case 6:
                                                              age = '${age}f';
                                                              break;

                                                            default:
                                                              _limitExceeded[
                                                                  index] = true;
                                                              return;
                                                          }
                                                        }

                                                        _validateName = true;
                                                        _validateAge[index] =
                                                            true;
                                                        _imageNameList[index] =
                                                            '${_nameController.text}A$age.jpg';
                                                      });
                                                    }
                                                  },
                                                  icon: Icon(
                                                    Icons.check_box,
                                                    color: Colors.white,
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
                              for (int i = 0; i < files.length; i++) {
                                _imageNameList.add(files[i].name);
                                _imagePathList.add(files[i].path!);
                                _ageControllers.add(TextEditingController());
                                _validateAge.add(true);
                                _inputAgeList.add('0');
                                _limitExceeded.add(false);
                              }
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
                  Align(
                    alignment: Alignment.bottomCenter,
                    child: Padding(
                      padding: EdgeInsets.only(bottom: edge20 * 0.8),
                      child: ElevatedButton.icon(
                        style: ButtonStyle(
                          backgroundColor:
                              const MaterialStatePropertyAll(Colors.green),
                          minimumSize: MaterialStatePropertyAll(
                            Size(edge20, edge20 * 3),
                          ),
                        ),
                        onPressed: () async {
                          String url = 'http://localhost:5000/uploadimage';
                          String message =
                              'Please confirm name and age of images';
                          for (int i = 0;
                              i < _imageFilesBytesList.length;
                              i++) {
                            if (_inputAgeList.isEmpty) break;
                            if (i == _inputAgeList.length) break;
                            if (_inputAgeList[i] == '0') {
                              message += '#$i,';
                              continue;
                            }

                            http.StreamedResponse response =
                                await asyncFileUpload(
                                    _imageNameList[i],
                                    _inputAgeList[i],
                                    _imageFilesBytesList[i],
                                    url);
                            if (response.statusCode == 404) {
                              SnackBar snackBar = SnackBar(
                                duration: const Duration(seconds: 10),
                                content: Text(
                                    'Could not find face in image ${_imageNameList[i]}'),
                              );
                              // ignore: use_build_context_synchronously
                              ScaffoldMessenger.of(context)
                                  .showSnackBar(snackBar);
                            }
                            setState(() {
                              _imageNameList.removeAt(i);
                              _imagePathList.removeAt(i);
                              _ageControllers.removeAt(i);
                              _validateAge.removeAt(i);
                              _inputAgeList.removeAt(i);
                              _limitExceeded.removeAt(i);
                              _imageFilesBytesList.removeAt(i);
                            });
                            i--;
                          }
                          if (message !=
                              'Please confirm name and age of images') {
                            SnackBar snackBar = SnackBar(
                              duration: const Duration(seconds: 10),
                              content: Text(message),
                            );
                            // ignore: use_build_context_synchronously
                            ScaffoldMessenger.of(context)
                                .showSnackBar(snackBar);
                          }
                        },
                        icon: const Icon(Icons.check_circle),
                        label: const Text("Submit Images"),
                      ),
                    ),
                  )
                ],
              ),
            ),
          )
        ],
      ),
    );
  }
}
