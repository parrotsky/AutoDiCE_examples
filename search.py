import optuna
from optuna_dashboard import run_server
# import pandas as pd
import numpy as np
from functools import partial
# Huggingface
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
import torchvision
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import vgg16_bn,VGG16_BN_Weights



# from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
# from val import validate_robustness
import argparse
import copy, pickle
import os,sys

# AutoDiCE
from partition.interface import *
import re
import subprocess
import requests


### Special warpper for transformer library.
### A simple trick to save model;
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        # Assuming 'logits' is the key you are interested in
        output = self.model(x)
        return output.logits

def download_model_if_not_exists(url, target_path):
    ### for downloading onnx model directly for the url and saved in target path.
    if not os.path.exists(target_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file from {url} to {target_path}")
    else:
        print(f"File {target_path} already exists. No need to download.")

# Function to read log file and extract relevant information
def read_log_and_extract(filename):
    with open(filename, 'r') as file:
        content = file.read()
    perf = 0.0
    mem = 0.0       
    # Regular expression patterns to match avg and memory usage
    avg_pattern = r'IRank: (\d+),min\s*=\s*([\d.]+), max\s*=\s*([\d.]+), avg\s*=\s*([\d.]+)'
    memory_pattern = r'IRank: (\d+), Physical Memory Usage \(KB\): (\d+)'

    # Search for avg and memory usage
    avg_matches = re.findall(avg_pattern, content)
    memory_matches = re.findall(memory_pattern, content)

    # Output the extracted values
    for rank, min_val, max_val, avg in avg_matches:
        avg =float(avg)
        # print(f"IRank {rank} has min = {min_val}, max = {max_val}, and avg = {avg}")
        if avg > perf:
            perf = avg

    for rank, memory in memory_matches:
        memory = int(memory) * 1.0 / 1024 # Kbytes ---> MBytes
        # print(f"IRank {rank} has a Physical Memory Usage of {memory} KB") 
        if memory > mem:
            mem = memory
    return mem, perf


def multi_objective(trial, resourceid, input_model):
    spoints = len(resourceid) - 1
    onnxmodel =  onnx.load(input_model)
    graph = onnxmodel.graph
    # for n in graph.node:
    #     n.name = str(n.output[0])
    #     n.name = n.name.replace('/','_')
    #     n.name = n.name.replace('.','_')

    node_map = generate_node_dict(onnxmodel.graph.node)
    model_layer_num = len(node_map)   # Get the total number of CNN layers.
    ### Generate random splitting for the model.
    # Generate b-1 unique random numbers within the range [1, a]
    spe = [trial.suggest_int(f'{i}', 0, model_layer_num) for i in range(spoints)]
    # random_nums = np.random.choice(range(1, model_layer_num), partition_num-1, replace=False)

    # Add 0 at the start and a at the end and sort the array
    split_points = np.sort(np.concatenate(([0], spe, [model_layer_num])))

    # Generate the submodels accordingly 
    submodels = [list(node_map)[split_points[i]:split_points[i+1]] for i in range(spoints+1)]
    mapping_dict = {}
    for i in range(spoints+1):
        resource = resourceid[i]
        mapping_dict[resource] = submodels[i]
    
    # save_json(mapping_dict, f"{args.save_path}/mapping.json")
    # print ("Exporting template mapping file for you...Done!")
    # random_map = load_json(f"{args.save_path}/mapping.json")

    InputSpecs = Interface(model=input_model, mappings=mapping_dict, platforms=resourceid)
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
        Benchmark = True)
    if True:
        os.system(f"cmake .&& cd models  && make && mpirun -rf rankfile ./multinode_bench  dog.jpg")
    os.system(f"cmake .&& cd models  && make && mpirun -rf rankfile ./multinode_bench  dog.jpg > ../{args.save_path}/test.log 2>&1  && cd ..")
    # command = f"cmake .&& cd models  && make && mpirun -rf rankfile ./multinode_bench  dog.jpg && cd .."
    # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    # Access the stdout output with result.stdout and stderr output with result.stderr
    # output = result.stdout
    # error_output = result.stderr

    memory, perf = read_log_and_extract(f"{args.save_path}/test.log")
    # result = subprocess.run(["bash", "-c", f"cmake . && cd models && make && mpirun -rf rankfile ./multinode_bench  dog.jpg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # result = subprocess.run(["bash", "-c", f"cmake . && cd models && make && mpirun -rf rankfile ./multinode_bench  dog.jpg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # output = result.stdout
    # print (output)
    # parse_stdout_bench(output)
    return memory, perf


# def multi_objective(trial, model, device_num, fail_num, flops_reduction, val_loader, dist_list):
    
#     dist = [trial.suggest_categorical(f'dist_{i}', dist_list[i].tolist()) for i in range(24)]
#     partition(dist, device_num, args.save_path)
#     worst_comb = worst_failure(device_num, fail_num, args.save_path)

#     flops = flop_eval(model, args.save_path, device_num)
#     if flops > flops_reduction:
#         raise optuna.TrialPruned()
#     # Assuming evaluate() returns flops and jsd_loss
#     try:
#         jsd_loss = jsd_eval(val_loader, 1, model, args.save_path, worst_comb)
#     except:
#         print ("jsd failed..")
#         raise optuna.TrialPruned()
#     # Restore original model weights.
#     return jsd_loss


def main():
    global args
    parser = argparse.ArgumentParser(description='Search Design space for distributed inference.')
    parser.add_argument('arch', type=str, help='Model Architecture..')
    parser.add_argument('study', type=str, help='The name of the study to search for.')
    parser.add_argument('save_path', type=str, help='The Path folder to save experiments results.')
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    if "resnet18" in args.arch:  
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    elif "alexnet" in args.arch:
        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights)
    elif "resnet101" in args.arch:
        weights = ResNet101_Weights.IMAGENET1K_V1
        model = resnet101(weights=weights)
    elif "vgg16_bn" in args.arch:
        weights=VGG16_BN_Weights.IMAGENET1K_V1
        model = vgg16_bn(weights=weights) 
    elif "vit" in args.arch:
        original_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        model = WrappedModel(original_model)

    else:
        raise ValueError(f"{args.arch} not support yet.")


    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        dynamic_axes = {'data_0': {0: 'batch_size'}, 'prob_1': {0: 'batch_size'}}
        input_names = ["data_0"]
        output_names = ["prob_1"]

        torch.onnx.export(model, x,
                        f"{args.save_path}/{args.arch}.onnx", 
                        export_params=True,
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                #   operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                        opset_version=12,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
                 )


# You could try disabling checking when tracing raises error
# mod = torch.jit.trace(net, x, check_trace=False)
    # mod = torch.jit.trace(model, x)
    # mod.save(f"{args.arch}.pt")
    # os.system(f"./pnnx {args.arch}.pt inputshape=[1,3,224,224]")
    # input_model = format_onnx(f"{args.arch}.pnnx.onnx")

    input_model = format_onnx(f"{args.save_path}/{args.arch}.onnx")
    model = onnx.load(input_model)
    inferred_model = onnx.shape_inference.infer_shapes(model)                                                                                                                                        
    graph = inferred_model.graph
    for n in graph.node:
        n.name = str(n.output[0])
        n.name = n.name.replace('/','_')
        n.name = n.name.replace('.','_')

    onnx.save(inferred_model, f"{args.save_path}/format_{args.arch}.onnx")

    node_map = generate_node_dict(inferred_model.graph.node)
    ### Parse platform.txt file to generate a template mapping.
    file_path = 'platform.txt'
    parsed_data = parse_platform_file(file_path)

    ### Generate a template mapping 
    resourceid = ['lenovo_cpu0123', 'lenovo_gpu']   ### You need to modify it to your available machine resources.
    partition_num = len(resourceid)
    model_layer_num = len(node_map)


    if False:  
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

    # device_num = 4 ## Number of Devices.
    # fail_num = 1
    # #val_loader = val_loader_only("/media/sky/DATASET/imagenet", 256, 4, False)

    # # storage = optuna.storages.InMemoryStorage()
    # # study = optuna.create_study(directions=['minimize', 'minimize'], storage = storage)
    # # study.optimize(partial(objective, model=model, device_num=device_num, fail_num = fail_num, val_loader=val_loader), n_trials=100)
    # # run_server(storage)
    # # sampler = optuna.samplers.NSGAIISampler()
    # # sampler = optuna.samplers.NSGAIIISampler()
    # # sampler = optuna.samplers.TPESampler()
    # # storage = optuna.storages.InMemoryStorage()

    # #study_name = 'my_study'
    # #storage_url = 'sqlite:///NAGAIIstudy.db'
    # #sampler = optuna.samplers.NSGAIISampler()
    # #if os.path.exists("NAGAIIstudy.db"):
    # #    print ("load previous study...")
    # #else:
    # ## Create study
    # #    study = optuna.create_study(sampler=sampler, directions=['minimize', 'minimize'], study_name=study_name, storage=storage_url)
    # #study_name = 'motpestudy'
    # #storage_url = 'sqlite:///MOTPEstudy.db'
    # ##sampler = optuna.samplers.TPESampler()
    # #sampler = optuna.samplers.MOTPESampler()
    # #if os.path.exists("MOTPEstudy.db"):
    # #    print ("load previous motpe study...")
    # #else:
    # ## Create study
    # if not os.path.exists('importance/vit/dist_list.pkl'):
    #     dist_list =  dist_matrix()
    # else:
    #     with open('importance/vit/dist_list.pkl', 'rb') as f:
    #         dist_list = pickle.load(f)
    study_name = args.study
    storage_url = f'sqlite:///{study_name}.db'
    #sampler = optuna.samplers.TPESampler()
    #sampler = optuna.samplers.TPESampler()
    if "nsga" in args.study:
        sampler = optuna.samplers.NSGAIISampler()
    elif "tpe" in args.study:
        sampler = optuna.samplers.TPESampler()
    elif "cmaes" in args.study:
        sampler = optuna.samplers.CmaEsSampler()
    else:
        raise ValueError("study not support.")

    if os.path.exists(f"{study_name}.db"):
        print (f"load previous {study_name} study...")
    else:
    # Create study
        study = optuna.create_study(sampler=sampler, directions=['minimize', 'minimize'], study_name=study_name, storage=storage_url)
        # study = optuna.create_study(sampler=sampler, directions=['minimize'], study_name=study_name, storage=storage_url)
    # Optimize in blocks of 100 trials
    for i in range(10):  # 10 blocks of 100 trials
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        #  fail_num=fail_num, flops_reduction=args.flops, 
        # study.optimize(partial(single_objective, model=model, ), n_trials=100)
        study.optimize(partial(multi_objective, resourceid=resourceid, 
            input_model= f"{args.save_path}/format_{args.arch}.onnx"),
            n_trials=100)


    # in terminal
    # optuna-dashboard sqlite:///db.sqlite3
    # You can analyze the Pareto front solutions afterward
    #pareto_solutions = study.get_pareto_front_trials()



    # Convert study trials to DataFrame
    # df = study.trials_dataframe(attrs=('number', 'params', 'values'))

    # # Filter to get Pareto optimal trials
    # def is_pareto_efficient(costs):
    #     is_efficient = np.ones(costs.shape[0], dtype=bool)
    #     for i, c in enumerate(costs):
    #         if is_efficient[i]:
    #             is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
    #             is_efficient[i] = True  # And keep self
    #     return is_efficient

    # pareto_trials = df.loc[is_pareto_efficient(df[['values_0', 'values_1']].values)]
    print ("---------")




if __name__ == '__main__':
    main()

