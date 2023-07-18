# AutoDiCE Project.
[The source for this AutoDiCE project is available here][src].
![Python Logo](https://www.python.org/static/community_logos/python-logo.png "Sample inline image")

This repository provides instructions for installation, model generation, compilation, and conversion of models into NCNN param and bin files.
## 1. Preparation: 
### 1.1 Check MPI installation & Cluster Env (Hello world for mpirun, Important)
Before running into multi-node or single-node-multi-threads programs, we need to check if our MPI environment is installed correctly. 
Then compile and run it with the command:

    (shell)$ mpic++ helloworld.cpp -o helloworld && mpirun -np 2 ./helloworld

    
And if you have multiple devices, please check the official [MPI tutorials][mpi]. It introduces how to build a local LAN cluster with MPI. For a supercluster like Slurm, PBS, DAS-5, we refer to their official instructions for MPI configurations.

### 1.2 Download ONNX models 

In this tutorial, you need to download [AlexNet](https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx) for a simple demo to show how to define the Mapping Specification.
You can find more [onnx models](https://github.com/onnx/models) and download them according to your preferences.
Please note: we only test for opset version-9 onnx models.



## 2. Installation

First, you will need to install the `autodice` python package. You can do this with pip:

```bash
pip3 install autodice
sudo apt install libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev  ### (optional)

```
Next, download the NCNN package from the official NCNN [releases][ncnn] according to your platform and requirements...

...

```bash
git clone https://github.com/parrotsky/AutoDiCE_examples.git
cd AutoDiCE_examples
tar -xzf ncnn.tar.gz -C .  ### To current source directory (AutoDiCE_examples directory.)
```

Finally, download the Vulkan SDK from the [LunarG website][vulkan]. 

### Additional Notes (Optional)

If you need to apply horizontal partitioning, please download and compile the `autodice` branch named `hybridpartition`. 

You can clone the specific branch with the following command:

```bash
git clone -b hybridpartition https://github.com/parrotsky/AutoDiCE.git
```

## 3. How to use the Partition Package to generate submodels and cpp files. 
### **Quick Start**.

```bash
python3 01.alexnet.interface.py
# this is for the interface to parse models and platform/mapping specification files.
python3 02.alexnet.frontend.py
# this is for model splitting and code generations.
```

Now we can define the Mapping Specification or partitioning the models manually . We want to use two cpu cores in this demo to simulate a multi-node scenario. For this example, the hostname of machine is "lenovo". We define the two keys in the [mapping.json](https://github.com/parrotsky/AutoDiCE/blob/main/tools/distributed/vertical/mapping.json) file as: "lenovo_cpu0"  and "lenovo_cpu1".
Important: Modify the mapping.json file according to the hostname of your machine. Make sure to replace both occurrences of "lenovo" with the output of the `hostname` command. Then we can generate two submodels according to our mapping specification file:



### Prepare AlexNet Model and Deploy over two Computing nodes
#### 1. mapping.json Template (AlexNet):
```
### By changing the keys in the mapping.json.
# For a Single Device multiple CPU cores
  {"lenovo_cpu0": ["conv1_1", "conv1_2", "norm1_1", "pool1_1", "conv2_1", "conv2_2", "norm2_1", "pool2_1", "conv3_1", "conv3_2", "conv4_1", "conv4_2", "conv5_1", "conv5_2", "pool5_1", "OC2_DUMMY_0", "fc6_1", "fc6_2"], "lenovo_cpu1": ["fc6_3", "fc7_1", "fc7_2", "fc7_3", "fc8_1", "prob_1"]}
# For a Single Device CPU (all cores) + GPU
  {"lenovo_gpu": ["conv1_1", "conv1_2", "norm1_1", "pool1_1", "conv2_1", "conv2_2", "norm2_1", "pool2_1", "conv3_1", "conv3_2", "conv4_1", "conv4_2", "conv5_1", "conv5_2", "pool5_1", "OC2_DUMMY_0", "fc6_1", "fc6_2"], "lenovo_cpu012345": ["fc6_3", "fc7_1", "fc7_2", "fc7_3", "fc8_1", "prob_1"]}
# For Two Devices: One CPU (all cores) + The Other GPU
  {"lenovo0_gpu": ["conv1_1", "conv1_2", "norm1_1", "pool1_1", "conv2_1", "conv2_2", "norm2_1", "pool2_1", "conv3_1", "conv3_2", "conv4_1", "conv4_2", "conv5_1", "conv5_2", "pool5_1", "OC2_DUMMY_0", "fc6_1", "fc6_2"], "lenovo1_cpu012345": ["fc6_3", "fc7_1", "fc7_2", "fc7_3", "fc8_1", "prob_1"]}
``` 

```python
### Example Code for Code Generation...
import os,sys
 
from partition.interface import *

if __name__ == '__main__':

    output_dirs= './models'
    if not os.path.exists(output_dirs):
    # Create a new directory because it does not exist
        os.makedirs(output_dirs)
        print("The output directory %s is created!" % (output_dirs))

    origin_model = "bvlcalexnet-9.onnx"
    ### Can be downloaded from:
    # https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx
    input_model = format_onnx(origin_model)
    model =  onnx.load(input_model)
    # resourceid = { 1:'lenovo_cpu0', 2:'lenovo_cpu1'}  ### Given manually.
    platforms = ['lenovo']

    ### How the layers in alexnet model distribute over two computing nodes???
    random_map = {
            "lenovo_cpu0":
                ["conv1_1", "conv1_2", "norm1_1", "pool1_1", "conv2_1",
                "conv2_2", "norm2_1", "pool2_1", "conv3_1", "conv3_2",
                "conv4_1", "conv4_2", "conv5_1", "conv5_2", "pool5_1",
                "OC2_DUMMY_0", "fc6_1", "fc6_2"],
            "lenovo_cpu1":
                ["fc6_3", "fc7_1", "fc7_2", "fc7_3", "fc8_1", "prob_1"]
             }
    

    InputSpecs = Interface(model=input_model, mappings=random_map, platforms=platforms)
    #cppname, NodesList, ComputingNodes
    ### Generate Cpp Files for Multinode-Inference.
    GenerateCode = EngineCode(
        CppName = "./models/multinode",
        Platforms = InputSpecs.platforms,
        NodesList = InputSpecs.nodes,
        ComputingNodes = InputSpecs.computingnodes,
        ValueInfos = InputSpecs.value_map,
        Inputs = InputSpecs.inputs,
        Outputs = InputSpecs.outputs,
        Benchmark = False)

# ...
```

## 4. Compile and Execution 

After generating multiple submodels and essential files for multiple node inference, we have to compile and run inference.

First, we need to use onnx2ncnn tool to convert these onnx models to ncnn (.bin and .param) files for the ncnn engine to load and run.

Second, we compile with ncnn library and vulkan (if needed) to obtain our binary execution file (SPMD).

Third, Enjoy
```bash
cd models/ && cmake .. && make -j2 && cp ../dog.jpg . && cp ../synset_words.txt .
mpirun -rf rankfile ./multinode dog.jpg
```

### 4.1 Building the `onnx2ncnn` Tool (Optional)

To convert ONNX models to NCNN binary and parameter files, you need to build the `onnx2ncnn` tool. Follow these steps:

Navigate to the `tools` directory in the NCNN source tree:

    ```bash
    cd tools && protoc onnx.proto --cpp_out=.
    cmake . && make
    ```

The `onnx2ncnn` tool should now be compiled and ready for use.

### 4.2 Converting ONNX Models to NCNN Format manually; Compile!!!

After building the `onnx2ncnn` tool, you can use it to convert ONNX models to NCNN's binary and parameter files. Follow these steps:

1. Navigate to the location of the `onnx2ncnn` tool:

    ```bash
    cd tools/onnx2ncnn models/
    ```
2. Use the tool to convert the `.onnx` files:

    ```bash
    ./onnx2ncnn lenovo_cpu0.onnx 
    ```

    It would generate ` lenovo_cpu0.param`, and ` lenovo_cpu0.bin` under the models directory.

3. Once all the sub-models have been converted, compile them:

    ```bash
    cd models/
    cmake ..
    make
    ```

    This will compile the converted models in your `models` directory.


4. Now we can finally use the `mpirun` command to run the multi-node inference application.
```    
    (shell) $ cd models/ && cp ../dog.jpg . && cp ../synset_words.txt . && mpirun -rf rankfile ./multinode dog.jpg
    215 = 0.593469
    207 = 0.125113
    213 = 0.102682
    Brittany spaniel 
    golden retriever 
    Irish setter, red setter 
```    

# FAQ
## Protobuf, onnx2ncnn
```bash
fatal error: google/protobuf/port_def.inc: No such file or directory
   10 | #include <google/protobuf/port_def.inc>
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [CMakeFiles/onnx2ncnn.dir/build.make:84: CMakeFiles/onnx2ncnn.dir/onnx2ncnn.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/onnx2ncnn.dir/all] Error 2
You can regenerate the header files again with the "protoc onnx.proto --cpp_out=." in your terminal.

```
For more information we refer to the jupyter notebook [alexnet partition](https://github.com/parrotsky/AutoDiCE/blob/main/tools/distributed/vertical/vertical%20partition%20tutorial.ipynb). It shows how to generate multi-node inference c++ code file and corresponding sub-models in details.

[packaging guide]: https://packaging.python.
[mpi]: https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/
[distribution tutorial]: https://packaging.python.org/tutorials/packaging-projects/
[src]: https://github.com/parrotsky/AutoDiCE
[ncnn]: https://github.com/Tencent/ncnn/releases
[vulkan]: https://vulkan.lunarg.com/sdk/home
[rst]: http://docutils.sourceforge.net/rst.html
[md]: https://tools.ietf.org/html/rfc7764#section-3.5 "CommonMark variant"
[md use]: https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
