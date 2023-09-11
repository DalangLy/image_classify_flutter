import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {

  late final int _interpreterAddress;
  late final Tensor _inputTensor;
  late final Tensor _outputTensor;
  late final List<String> _labels;

  @override
  void initState() {
    super.initState();
    
    _loadModel();
    _loadLabels();
  }
  
  Future<void> _loadModel() async {
    final options = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate());
    }

    // Use GPU Delegate
    // doesn't work on emulator
    // if (Platform.isAndroid) {
    //   options.addDelegate(GpuDelegateV2());
    // }

    // Use Metal Delegate
    if (Platform.isIOS) {
      options.addDelegate(GpuDelegate());
    }

    // load model
    final Interpreter interpreter = await Interpreter.fromAsset('assets/tnoat/model_unquant.tflite', options: options);
    //final Interpreter interpreter = await Interpreter.fromAsset('assets/mobilenet_v1_1.0_224_quant.tflite');
    _inputTensor = interpreter.getInputTensors().first;
    _outputTensor = interpreter.getOutputTensors().first;
    _interpreterAddress = interpreter.address;
  }
  
  Future<void> _loadLabels() async {
    // load label
    final labelTxt = await rootBundle.loadString('assets/tnoat/labels.txt');
    //final labelTxt = await rootBundle.loadString('assets/labels_mobilenet_quant_v1_224.txt');
    _labels = labelTxt.split('\n');
  }

  
  
  
  Uint8List? _imageBytes;
  List<MapEntry<String, double>> _result = [];
  bool _waiting = false;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: MediaQuery.of(context).size.width * 0.8,
                height: MediaQuery.of(context).size.width * 0.8,
                decoration: BoxDecoration(
                  border: Border.all()
                ),
                child: _imageBytes == null ? const Center(
                  child: Text("No Image"),
                ) : Image.memory(_imageBytes!, fit: BoxFit.contain,),
              ),
              ElevatedButton(
                onPressed: _waiting ? null : () async {
                  if(_imageBytes == null) {
                    showDialog(
                      context: context, builder: (context) {
                        return const AlertDialog(
                          content: Text("Please select image"),
                        );
                      },
                    );
                  }
                  _waiting = true;
                  final Map<String, double> classification = await _classifyImage(imageData: _imageBytes!, inputShape: _inputTensor.shape, outputShape:  _outputTensor.shape, interpreterAddress: _interpreterAddress, labels: _labels);
                  final List<MapEntry<String, double>> result = (classification.entries.toList()..sort(
                        (a, b) => a.value.compareTo(b.value),
                  )).reversed.take(3).toList();
                  _result = result;
                  _waiting = false;
                  setState(() {

                  });
                },
                child: const Text("Classify"),
              ),
              ..._result.map((e) => Container(
                padding: const EdgeInsets.all(8.0),
                margin: const EdgeInsets.symmetric(vertical: 10.0),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(5.0),
                  color: Colors.cyan,
                ),
                child: Text(
                  "${e.key} ${(e.value * 100).toStringAsFixed(0)}%",
                  style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white),
                ),
              )),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          _imageBytes = null;
          setState(() {

          });
          try{
            final Uint8List imageBytes = await _pickImage();
            setState(() {
              _imageBytes = imageBytes;
              _result.clear();
            });


          }catch(e){
            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.toString()),));
          }
        },
        child: const Icon(Icons.camera),
      ),
    );
  }

  Future<Uint8List> _pickImage() async {
    final ImagePicker picker = ImagePicker();
    // Pick an image.
    final XFile? xFile = await picker.pickImage(source: ImageSource.gallery);
    if(xFile == null) throw Exception("No image selected");
    final Uint8List imageData = File(xFile.path).readAsBytesSync();
    return imageData;
  }

  Future<Uint8List> _takePicture() async {
    final ImagePicker picker = ImagePicker();
    // Pick an image.
    final XFile? xFile = await picker.pickImage(source: ImageSource.camera);
    if(xFile == null) throw Exception("No image selected");
    final Uint8List imageData = File(xFile.path).readAsBytesSync();
    return imageData;
  }

  Future<Map<String, double>> _classifyImage({required Uint8List imageData, required List<int> inputShape, required List<int> outputShape, required int interpreterAddress, required List<String> labels,}) async {
    final image_lib.Image? img = image_lib.decodeImage(imageData);

    // resize original image to match model shape.
    final image_lib.Image imageInput = image_lib.copyResize(
      img!,
      width: inputShape[1],
      height: inputShape[2],
    );


    final List<List<List<num>>> imageMatrix = List.generate(
      imageInput.height,
          (y) => List.generate(
        imageInput.width,
            (x) {
          final image_lib.Pixel pixel = imageInput.getPixel(x, y);
          return [pixel.r, pixel.g, pixel.b];
        },
      ),
    );

    final input = [imageMatrix];
    // Set tensor output [1, 1001]
    final output = [List<double>.filled(outputShape[1],0.0)];
    // // Run inference
    final Interpreter interpreter = Interpreter.fromAddress(interpreterAddress);
    interpreter.run(input, output);
    // Get first output tensor
    final result = output.first;
    final double maxScore = result.reduce((a, b) => a + b);
    // Set classification map {label: points}
    final Map<String, double> classification = <String, double>{};
    for (var i = 0; i < result.length; i++) {
      if (result[i] != 0) {
        // Set label: points
        classification[labels[i]] =
            result[i].toDouble() / maxScore.toDouble();
      }
    }
    return classification;
  }
}