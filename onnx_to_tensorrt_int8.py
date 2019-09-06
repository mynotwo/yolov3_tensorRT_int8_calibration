#!/usr/bin/env python2
from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

from yolov3_to_onnx import download_file
from yolov3_to_onnx import DarkNetParser
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
import calibrator as calibra
import glob
import time
import random
import math
from scipy import misc

TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

def load_random_batch(calib):
    # Load a random batch.
    batch = random.choice(calib.batch_files)
    _, data = calib.read_batch_file(batch)
    data = np.fromstring(data, dtype=np.float32)
    return data

# get darnet configuration
def get_darknet_params(cfg_file_path):
    supported_layers = ['net', 'convolutional', 'shortcut','route', 'upsample']
    parser = DarkNetParser(supported_layers)
    layer_configs = parser.parse_cfg_file(cfg_file_path) 
    del parser

    return layer_configs

# get kernel number of each layer to calculate output shape
def get_filter_num(cfg_file_path, layer_names):
    layer_configs = get_darknet_params(cfg_file_path)
    filter_nums = []
    for layer_name in layer_names:
        layer = layer_configs[layer_name]
        if 'filters' in layer:
            filter_nums.append(layer['filters'])
        else:
            filter_nums.append(filter_nums[-1])
    
    return filter_nums

def get_layer_num(cfg_file_path, layer_name):
    layer_configs = get_darknet_params(cfg_file_path)
    # It should be -1, set it as -2 to skip first element
    layer_num = -2
    for key in layer_configs:
        if "convolutional" in key:
            layer_num += 3
        else:
            layer_num += 1
        if key == layer_name:
            break

    return layer_num

def build_int8_engine(onnx_file_path, calib, cfg_file_path, output_layer_name="", engine_file_path=""):
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = calib.get_batch_size()
            builder.max_workspace_size = common.GiB(1)
            # Activate int8 mode
            builder.int8_mode = True
            builder.int8_calibrator = calib
            print("Builder works on Int8 mode.  Max batch size:{:d}, Max work space size:{:d}.".format(builder.max_batch_size, builder.max_workspace_size))
            # Parse model
            print("Parsing onnx model file...")
            with open(onnx_file_path, 'rb') as model:
                model_tensors = parser.parse(model.read())
                if output_layer_name!="":
                    print("Set output layer as {}".format(output_layer_name))
                    for layer in output_layer_name:
                       # Every normal layer consists of 3 sub-layer, except shortcut
                       print(get_layer_num(cfg_file_path, layer))
                       network.mark_output(network.get_layer(get_layer_num(cfg_file_path, layer)).get_output(0))
            print("Compelete parsing ONNX file.")
        # network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration
            print("Start building engine...")
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    onnx_file_path = 'yolov3.onnx'
    engine_file_path = 'yolo_in8.trt'
    cfg_file_path = "yolov3.cfg"
    
    input_image_path = download_file('dog.jpg',
        'https://github.com/pjreddie/darknet/raw/f86901f6177dfc6116360a13cc06ab680e0c86b0/data/dog.jpg', checksum_reference=None)
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (608, 608)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    middle_output_shapes = []

    # calibrator definition
    calibration_dataset_loc = "calibration_dataset/"
    calibration_cache = "yolo_calibration.cache"
    calib = calibra.PythonEntropyCalibrator(calibration_dataset_loc, cache_file=calibration_cache)
    
    # define the layer output you want to visualize
    output_layer_name = ["001_convolutional", "002_convolutional", "003_convolutional", "005_shortcut", "006_convolutional"]
    # get filter number of defined layer name
    filter_num = get_filter_num(cfg_file_path, output_layer_name)
    
    # Do inference with TensorRT
    trt_outputs = []
    with build_int8_engine(onnx_file_path, calib, cfg_file_path, output_layer_name, engine_file_path) as engine, engine.create_execution_context() as context: 
        start = time.time()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        # if batch size != 1 you can use load_random_batch to do test inference, here I just use 1 image as test set
        # inputs[0].host = load_random_batch(calib)
        inputs[0].host = image
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    end = time.time()
    print("Inference costs %.02f sec."%(end - start))
    for i, output in enumerate(trt_outputs[:len(filter_num)]):
        # length of inference output should be filter_num*h*h
        if "convolutional" in output_layer_name[i]:
            h = int(math.sqrt(output.shape[0]/filter_num[i]))
            w = h
        else:
            h = int(math.sqrt(output.shape[0]/filter_num[i]/2))
            w = 2*h
        middle_output_shapes.append((1, filter_num[i], w, h))
    # reshape
    middle_output = [output.reshape(shape) for output, shape in zip(trt_outputs[:len(filter_num)], middle_output_shapes)]
    # save middle output as grey image
    for name, output in zip(output_layer_name, middle_output):
        w, h = output.shape[2], output.shape[3]
        img = misc.toimage(output.sum(axis=1).reshape(w, h))
        img.save("{}.tiff".format(name)) 
    print("Saveing middle output {}".format(output_layer_name))
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs[len(filter_num):], output_shapes)]

    postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    output_image_path = 'dog_bboxes.png'
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__ == '__main__':
    main()
