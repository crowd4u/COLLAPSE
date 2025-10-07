import sys
import os
import pandas as pd
import json

from sklearn.metrics import accuracy_score

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

def get_save_file_path(dataset_profile, exp_params):
    dataset_name = dataset_profile["dataset_name"]
    n_classes = dataset_profile["n_classes"]
    scenario = exp_params["scenario"]
    r = exp_params["r"]
    ai_acc = exp_params["ai_acc"]

    file_name = f"{DIR_PATH}/results/{dataset_name}_{scenario}_{r}_{ai_acc}.csv"
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

def update_save_file(file_path, exp_params, method, ai_acc, r, num_ai, iter, convergence, accuracy, biased_accuracy, uc_text):
    with open(file_path, "a") as f:
        f.write(f"{exp_params['dataset']},{exp_params['scenario']},{method},{ai_acc},{r},{num_ai},{iter},{convergence},{accuracy},{biased_accuracy},{uc_text}\n")

def get_accuracy(ret, gt):
    y_true = gt.sort_index().astype(str)
    y_pred = ret.sort_index().astype(str)
    acc = accuracy_score(y_true, y_pred)
    return acc

def get_biased_accuracy(ret, gt, biased_tasks):
    return get_accuracy(
        ret.loc[biased_tasks],
        gt.loc[biased_tasks]
    )     

def load_dataset_profile(dataset_name):
    with open(f"{DIR_PATH}/human_responses/{dataset_name}_dataset_profile.json", "r") as f:
        dataset_profile = json.load(f)
    return dataset_profile

def load_gt(dataset_name):
    gt = pd.read_csv(f"{DIR_PATH}/human_responses/{dataset_name}_gt.csv", dtype={"task": str, "gt": int})
    biased_tasks = gt[gt["gt"]==0]["task"].unique()
    gt = gt.set_index("task")
    return gt, biased_tasks

def load_human_responses(dataset_name, r):
    human = pd.read_csv(f"{DIR_PATH}/human_responses/{dataset_name}_r={r}.csv", dtype={"task": str, "worker": str, "label": int})
    return human