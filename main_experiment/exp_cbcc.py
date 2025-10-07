import sys
import os

import polars as pl
import pandas as pd
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MAX_ITER = 5
SEEDS = [12347, 77777, 2984, 298, 1102644]

from agg_methods_cbcc import AggregationMethod, get_aggregation_methods_CBCC
from scenario import create_ai_dataset
from io_utils import get_save_file_path, update_save_file, get_accuracy, get_biased_accuracy, load_dataset_profile, load_gt, load_human_responses

exp_df = pd.read_csv(f"{DIR_PATH}/exp.csv")

exp_params_list = exp_df.to_dict(orient="records")

def get_save_file_path_CBCC(dataset_profile, exp_params):
    dataset_name = dataset_profile["dataset_name"]
    n_classes = dataset_profile["n_classes"]
    scenario = exp_params["scenario"]
    r = exp_params["r"]
    ai_acc = exp_params["ai_acc"]

    file_name = f"{DIR_PATH}/results_cbcc/{dataset_name}_{scenario}_{r}_{ai_acc}.csv"
    if os.path.exists(file_name):
        assert False, "File already exists"
    with open(file_name, "w") as f:
        f.write("dataset,scenario,method,ai_acc,r,num_ai,iter,convergence,accuracy,biased_accuracy,uc_p,")
        for i in range(n_classes):
            f.write(f"uc_pih_{i},")
        for i in range(n_classes-1):
            f.write(f"uc_pia_{i},")
        f.write(f"uc_pia_{i+1}")
        f.write("\n")
    return file_name

for exp_params in exp_params_list:
    try:
        print(f"Running experiment with params: {exp_params}")
        dataset_name = exp_params["dataset"]
        dataset_profile = load_dataset_profile(dataset_name)
        gt, biased_tasks = load_gt(dataset_name)
        
        r = exp_params["r"]
        human = load_human_responses(dataset_name, r)
        ai_dataset_generator = create_ai_dataset(
            exp_params["scenario"], dataset_name, r, exp_params["ai_acc"], dataset_profile["n_classes"]
        )
        
        file_path = get_save_file_path_CBCC(dataset_profile, exp_params)
        
        for ai, num_ai in ai_dataset_generator:
            print(f"Number of AI workers: {num_ai}")
            for iter in range(MAX_ITER):
                print(f"Iteration: {iter}")
                methods = get_aggregation_methods_CBCC(dataset_profile["labels"], r=0.75, n_iter=100000)
                for method in methods:
                    print(f"Running method: {method.name}")   
                    try:                
                        ret = method.fit_predict(human, ai, seed=SEEDS[iter])
                        overall_accuracy = get_accuracy(ret, gt)
                        biased_accuracy = get_biased_accuracy(ret, gt, biased_tasks)
                        uc_text = method.get_uc_text(dataset_profile["n_classes"])
                        update_save_file(file_path, exp_params, method.name, exp_params["ai_acc"], r, num_ai,
                                        iter, method.is_converged(), overall_accuracy, biased_accuracy, uc_text)
                        print(f"Method: {method.name}, Overall Accuracy: {overall_accuracy}, Biased Accuracy: {biased_accuracy}")
                    except Exception as e:
                        uc_text = method.get_uc_text(dataset_profile["n_classes"])
                        update_save_file(file_path, exp_params, method.name, exp_params["ai_acc"], r, num_ai,
                                        iter, method.is_converged(), -1, -1, uc_text)
                        print(f"Method: {method.name}, Overall Accuracy: ERROR, Biased Accuracy: ERROR")
            if num_ai == exp_params["max_ai_num"]:
                print(f"Reached maximum number of AI workers: {num_ai}")
                break
    except Exception:
        continue