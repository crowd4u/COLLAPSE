import json

import numpy as np
import polars as pl

from ai_simulator import generate_confusion_matrix, get_dataset_human_acc, generate_ai_dataset

datasets = ["dog", "face", "tiny", "adult"]
ai_accs = ["-sigma", "mean", "+sigma", "max"]
num_AI = 30

for dataset_name in datasets:
    with open(f"../human_responses/{dataset_name}_dataset_profile.json", "r") as f:
        dataset_profile = json.load(f)

    r_range = dataset_profile["r_range"]
    n_classes = dataset_profile["n_classes"]
    labels = dataset_profile["labels"]

    for r in r_range:
        human_df = pl.read_csv(f"../human_responses/{dataset_name}_r={r}.csv").to_pandas()
        gt = pl.read_csv(f"../human_responses/{dataset_name}_gt.csv").to_pandas().set_index("task")
        human_acc, human_std = get_dataset_human_acc(human_df, gt)
        print(f"Dataset: {dataset_name}, r: {r}, human_acc: {human_acc:.3f}, human_std: {human_std:.3f}")

        for ai_acc in ai_accs:
            if ai_acc == "-sigma":
                ai_acc_value = human_acc - human_std
            elif ai_acc == "mean":
                ai_acc_value = human_acc
            elif ai_acc == "+sigma":
                ai_acc_value = human_acc + human_std
            elif ai_acc == "max":
                ai_acc_value = (n_classes - 1) / n_classes
            if ai_acc_value > (n_classes - 1) / n_classes:
                print(f"Skipping {dataset_name} r={r} {ai_acc} as it exceeds max accuracy.")
                continue

            h = (n_classes * ai_acc_value * (n_classes-1) - 1) / (n_classes *  (n_classes -2))
        
            print(f"Dataset: {dataset_name}, r: {r}, ai_acc: {ai_acc}, h: {h:.3f}")
            
            for fixed_target in range(1, n_classes):
                cm = generate_confusion_matrix(n_classes, r=h, fixed_index=[[0,fixed_target]])
                for i in range(num_AI):
                    ai_df = generate_ai_dataset(cm, gt, f"AI_{ai_acc}_{i}", labels)
                    ai_df.to_csv(f"../ai_responses/{dataset_name}_r={r}_ai={ai_acc}_target={fixed_target}_run={i}.csv", index=False)
