import 'dart:math';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:sensors/sensors.dart';
import 'package:csv/csv.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:permission_handler/permission_handler.dart';
//https://protocoderspoint.com/flutter-google-fonts-package-library-or-use-custom-fonts-in-flutter/

List<List<dynamic>> rows = [];

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(

        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

  // variables for x, y, z acceleration
  double x, y, z;

  //variable for our start/stop button state
  bool btnON          = false;
  bool datacollection = false;

  // TensorFlow Lite Interpreter object
  Interpreter _interpreter;

  //initialize output data to save predictions
  var output = List<double>(3).reshape([1, 3]);

  @override
  void initState() {

    super.initState();

    rows=[];
    x = 0;
    y = 0;
    z = 0;

    //initialize output
    output[0][0] = 0.0;
    output[0][1] = 0.0;
    output[0][2] = 0.0;

    //listen to sensor event
    //ToDO: add a gyroscope
    accelerometerEvents.listen((AccelerometerEvent event) {
      setState(() {
        x = event.x;
        y = event.y;
        z = event.z;
      });

      //only when start recording, insert x, y, z values to row and rows,
      if(btnON == true) {

        print("collected");
        List<dynamic> row = [];
        row.add(DateTime.now());
        row.add(x);
        row.add(y);
        row.add(z);
        //row.add('\n');
        rows.add(row);

        int rowLength = rows.length;
        //print('Row length is length $rowLength');

        //if required segment length is reached
        //TODO: Adjust # of samples required to detect your gestures.
        //TODO: Develop a proper way to design evaluation on streaming data. Model inference might take some time and you might miss new data meanwhile
        if(rows.length >= 8) {

          //compute features [raw acc feature ]
          List<List<double>> featuresAll = [];
          //add raw acc features
          for (var i = 0; i < 8; i++) {
            List<double> feature = [];
            var currVal = rows[i];
            feature.add(currVal[1]);
            feature.add(currVal[2]);
            feature.add(currVal[3]);
            //TODO: add acc magnitude feature, other features, and/or gyroscope data

            featuresAll.add(feature);
          }

          var featureDim = [featuresAll].shape;
          print(featuresAll);
          //save the model inference to output
          _interpreter.run([featuresAll],output);
          print('Output is $output');

          //reset list to save accelerometer data
          rows = [];
        }
      }
    });
  }

  void _loadModel() async {
    // Creating the interpreter using Interpreter.fromAsset
    _interpreter = await Interpreter.fromAsset('gesturerecog_model.tflite');
    print('Interpreter loaded successfully');
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text("Gesture Recognition example app"),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Padding(
                padding: const EdgeInsets.all(10.0),
                child: Text(
                  "Model prediction",
                  style: TextStyle(fontSize: 18.0, fontWeight: FontWeight.w900),
                ),
              ),
              Table(
                border: TableBorder.all(
                    width: 2.0,
                    color: Colors.white,
                    style: BorderStyle.solid),
                children: [
                  TableRow(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(
                          "Class 1 : ",
                          style: TextStyle(fontSize: 20.0),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(output[0][0].toStringAsFixed(2), //trim the x axis value to 2 digit after decimal point
                            style: TextStyle(fontSize: 20.0)),
                      )
                    ],
                  ),
                  TableRow(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(
                          "Class 2 : ",
                          style: TextStyle(fontSize: 20.0),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(output[0][1].toStringAsFixed(2),  //trim the y axis value to 2 digit after decimal point
                            style: TextStyle(fontSize: 20.0)),
                      )
                    ],
                  ),
                  TableRow(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(
                          "Class 3 : ",
                          style: TextStyle(fontSize: 20.0),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(output[0][2].toStringAsFixed(2),   //trim the z axis value to 2 digit after decimal point
                            style: TextStyle(fontSize: 20.0)),
                      )
                    ],
                  ),
                ],
              ),


          Padding(
            padding: const EdgeInsets.all(8.0),
            child: RaisedButton(

              child: btnON ? Text("Stop Recognition") : Text("Start Recognition"),
              onPressed: () {
                if(btnON==false) {
                  print('Start recognition!!!');
                  //load tflite model
                  _loadModel();
                  print(btnON);
                }
                else {
                  print('Stop recognition!!!');
                  print(btnON);
                }

                //switching button state
                setState(() {
                  btnON  = !btnON;}
                );
              },

            ),
          ),
            ],
          ),

        ),
    );
  }
}
