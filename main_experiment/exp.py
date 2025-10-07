import sys
import os

import polars as pl
import pandas as pd
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MAX_ITER = 5

from agg_methods import AggregationMethod, get_aggregation_methods
from scenario import create_ai_dataset
from io_utils import get_save_file_path, update_save_file, get_accuracy, get_biased_accuracy, load_dataset_profile, load_gt, load_human_responses

exp_df = pd.read_csv(f"{DIR_PATH}/exp.csv")

exp_params_list = exp_df.to_dict(orient="records")

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
        
        file_path = get_save_file_path(dataset_profile, exp_params)
        
        for ai, num_ai in ai_dataset_generator:
            print(f"Number of AI workers: {num_ai}")
            for iter in range(MAX_ITER):
                print(f"Iteration: {iter}")
                methods = get_aggregation_methods(dataset_profile["labels"], r=0.75, n_iter=100000)
                for method in methods:
                    print(f"Running method: {method.name}")                   
                    ret = method.fit_predict(human, ai)
                    overall_accuracy = get_accuracy(ret, gt)
                    biased_accuracy = get_biased_accuracy(ret, gt, biased_tasks)
                    uc_text = method.get_uc_text(dataset_profile["n_classes"])
                    update_save_file(file_path, exp_params, method.name, exp_params["ai_acc"], r, num_ai,
                                    iter, method.is_converged(), overall_accuracy, biased_accuracy, uc_text)
                    print(f"Method: {method.name}, Overall Accuracy: {overall_accuracy}, Biased Accuracy: {biased_accuracy}")
            if num_ai == exp_params["max_ai_num"]:
                print(f"Reached maximum number of AI workers: {num_ai}")
                break
    except Exception:
        # 実験が止まらないように
        continue