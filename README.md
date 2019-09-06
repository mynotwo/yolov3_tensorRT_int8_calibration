## Description

This code implements a full ONNX-based pipeline for performing inference with the YOLOv3 network, using int8 calibration. This code is partly based on the offical sample "yolov3\_onnx.py" given by Tensor RT. This sample is based on the [YOLOv3-608](https://pjreddie.com/media/files/papers/YOLOv3.pdf) paper.

**Note:** This sample is not supported on Ubuntu 14.04 and older. Additionally, the `yolov3_to_onnx.py` script does not support Python 3.

## Prerequisites

For specific software versions, see the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html).

1.  Install [ONNX-TensorRT: TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt). ONNX-TensorRT includes layer implementations for the required ONNX operators `Upsample` and `LeakyReLU`.

2.  Install the dependencies for Python.
	-   For Python 2 users, from the root directory, run:
	`python2 -m pip install -r requirements.txt`

	-   For Python 3 users, from the root directory, run:
	`python3 -m pip install -r requirements.txt`

## Running the sample

1.  Create an ONNX version of YOLOv3 with the following command. The Python script will also download all necessary files from the official mirrors (only once).
	`python yolov3_to_onnx.py`

    When running this sample you could get middle layer output by changing the definition of variable "output_layer_name", layer name can be get in yolov3.onnx

2.  Build a TensorRT engine from the generated ONNX file and run inference on a sample image, which will also be downloaded during the first run.
	`python onnx_to_tensorrt_int8.py`

    When running this sample you could get middle layer output by changing the definition of variable "output_layer_name", layer name can be found in yolov3.onnx

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:

# Additional resources

The following resources provide a deeper understanding about the model used in this sample, as well as the dataset it was trained on:

**Model**
- [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

**Dataset**
- [COCO dataset](http://cocodataset.org/#home)

**Documentation**
- [YOLOv3-608 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)


