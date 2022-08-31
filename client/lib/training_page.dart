import 'package:clay_containers/widgets/clay_container.dart';
import 'package:client/add_image_page.dart';
import 'package:client/select_dataset_page.dart';
import 'package:flutter/material.dart';

class TrainingPage extends StatelessWidget {
  const TrainingPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.max,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Container(
          color: const Color(0xff121212),
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width / 2,
          ),
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: ClayContainer(
              color: const Color(0xff121212),
              child: Image.asset(
                'images/facial-recognition-connected-real-estate.png',
                alignment: Alignment.center,
              ),
            ),
          ),
        ),
        Container(
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width / 2,
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
                        'Select dataset Directory',
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
                      onPressed: () {},
                      child: const Text(
                        'Add image to dataset',
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
    );
  }
}
