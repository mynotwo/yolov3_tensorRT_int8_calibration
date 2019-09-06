import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
from data_processing import PreprocessYOLO

# For reading size information ftch.shaperom batches
import struct

class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_data_dir, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        # Get a list of all the batch files in the batch folder.
        self.batch_files = [os.path.join(batch_data_dir, f) for f in os.listdir(batch_data_dir)]

        # Find out the shape of a batch and then allocate a device buffer of that size.
        self.batch_size = 1
        self.batch_round = 100
        self.shape = self.read_batch_file(self.batch_files[0:self.batch_size]).shape
        print(self.shape)       
 # Each element of the calibration data is a float32.
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)

        # Create a generator that will give us batches. We can use next() to iterate over the result.
        def load_batches():
            start = 0
            for i in range(self.batch_round):
                print("Start Calibration using batch {:d}".format(i))
                yield self.read_batch_file(self.batch_files[start:start+self.batch_size])
                start = start + self.batch_size
        self.batches = load_batches()

    # This function is used to load calibration data from the calibration batch files.
    # In this implementation, one file corresponds to one batch, but it is also possible to use
    # aggregate data from multiple files, or use only data from portions of a file.
    def read_batch_file(self, filename):
        batch = []
        input_resolution_yolov3_HW = (608, 608)
        for img_path in filename:
            preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
            image = preprocessor.process(img_path)
            batch.append(image[1])
        batch = np.array(batch)
        batch.shape = self.batch_size, 3, 608, 608
            
        return batch

    def get_batch_size(self):
        return self.shape[0]

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Get a single batch.
            data = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
