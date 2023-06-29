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
    # resourceid = { 1:'lenovo_cpu0', 2:'lenovo_cpu1'}
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



