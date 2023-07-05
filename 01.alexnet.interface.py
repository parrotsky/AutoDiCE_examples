import os,sys
 
from partition.interface import *
import os
import requests
import numpy as np

def download_model_if_not_exists(url, target_path):
    if not os.path.exists(target_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file from {url} to {target_path}")
    else:
        print(f"File {target_path} already exists. No need to download.")

if __name__ == '__main__':

    output_dirs= './models'
    if not os.path.exists(output_dirs):
    # Create a new directory because it does not exist
        os.makedirs(output_dirs)
        print("The output directory %s is created!" % (output_dirs))

    # Download the alexnet Model from onnx repo...
    url = 'https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx'
    origin_model = "bvlcalexnet-9.onnx"
    download_model_if_not_exists(url, origin_model)
    ### Can be downloaded from:
    # https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-9.onnx
    input_model = format_onnx(origin_model)
    model =  onnx.load(input_model)

    node_map = generate_node_dict(model.graph.node)
    ### Parse platform.txt file to generate a template mapping.
    file_path = 'platform.txt'
    parsed_data = parse_platform_file(file_path)

    ### Generate a template mapping 
    resourceid = ['lenovo_cpu0', 'lenovo_cpu1']   ### You need to modify it to your available machine resources.
    partition_num = len(resourceid)
    model_layer_num = len(node_map)
    
    ### Generate random splitting for the model.
    # Generate b-1 unique random numbers within the range [1, a]
    random_nums = np.random.choice(range(1, model_layer_num), partition_num-1, replace=False)
    # Add 0 at the start and a at the end and sort the array
    split_points = np.sort(np.concatenate(([0], random_nums, [model_layer_num])))

    # Generate the submodels accordingly 
    submodels = [list(node_map)[split_points[i]:split_points[i+1]] for i in range(partition_num)]
    mapping_dict = {}
    for i in range(partition_num):
        resource = resourceid[i]
        mapping_dict[resource] = submodels[i]
    
    save_json(mapping_dict, "mapping.json")
    print ("Exporting template mapping file for you...Done!")

