#%%

# === MAPPING FROM LABEL LETTERS TO NUMERICAL VALUES FOR EACH DATASET AND TASK ===
LABEL_MAPPING = {
    "dataset1" : {
        "task1" : {
            "RELEVANT" : 0,
            "IRRELEVANT" : 1,
        },
        "task2" : {
            "PROBLEM" : 0,
            "SOLUTION" : 1,
            "NEUTRAL" : 2,
        },
        "task4" : {
            "IN FAVOR OF" : 0,
            "AGAINST" : 1,
            "NEUTRAL" : 2,
        },
        "task5" : {
            "Section 230" : 0,
            "Trump ban" : 1,
            "Twitter Support" : 2,
            "Platform Policies" : 3,
            "Complaint" : 4,
            "Other" : 5,
        },
    },
    "dataset2" : {
        "task1" : {
            "RELEVANT" : 0,
            "IRRELEVANT" : 1,
        },
        "task2" : {
            "PROBLEM" : 0,
            "SOLUTION" : 1,
            "NEUTRAL" : 2,
        },
    },
    # "dataset3" : {
    #     "task1" : {
    #         "RELEVANT" : 0,
    #         "IRRELEVANT" : 1,
    #     },
    # },
    "dataset4" : {
        "task1" : {
            "RELEVANT" : 0,
            "IRRELEVANT" : 1,
        },
        "task2" : {
            "PROBLEM" : 0,
            "SOLUTION" : 1,
            "NEUTRAL" : 2,
        },
    },
}
# === END OF MAPPING ===

#%%
import os
from pathlib import Path
import pandas as pd
import polars as pl
import json

from utils_src import load_dataset_task_prompt_mappings, map_label_to_completion
from original_functions import process_output

#%%

class OutputFile:
    model_name: str
    dataset_id : int
    task_id : int
    sample : int
    epoch: int
    prompt_max_len : int
    batch_size : int
    grad_acc: int
    temperature: float

    def __str__(self):
       return json.dumps({
              "model_name": self.model_name,
              "dataset_id": self.dataset_id,
              "task_id": self.task_id,
              "sample": self.sample,
              "epoch": self.epoch,
              "prompt_max_len": self.prompt_max_len,
              "batch_size": self.batch_size,
              "grad_acc": self.grad_acc,
              "temperature": self.temperature,
       })
    
def separate_output_name(filename: str) -> OutputFile:
    parts = filename.split('_')
    output_file = OutputFile()
    output_file.model_name = "Llama-2-70b-chat-hf"
    if filename.endswith("eval_set.csv"):
        # e.g.) ds_4__task_1__sample_size_1500_eval_set.csv
        output_file.dataset_id = int(parts[1])
        output_file.task_id = int(parts[4])
        if parts[8] == "eval":
            output_file.sample = int(parts[7])
        else:
            output_file.sample = int(parts[8])
        output_file.epoch = -1
        output_file.prompt_max_len = -1
        output_file.batch_size = -1
        output_file.grad_acc = -1
        output_file.temperature = -1
    else:
        # e.g.) ds_4_task_2_sample_100_epochs_3_prompt_max_len_4096_batch_size_4_grad_acc_2_temp_0_05
        output_file.dataset_id = int(parts[1])
        output_file.task_id = int(parts[3])
        output_file.sample = int(parts[5])
        output_file.epoch = int(parts[7])
        output_file.prompt_max_len = int(parts[11])
        output_file.batch_size = int(parts[14])
        output_file.grad_acc = int(parts[17])
        if parts[19] == "0":
            tmp_str = "0." + parts[20].replace('.csv','')
            output_file.temperature = float(tmp_str)
        else:
            assert False, "Unexpected temperature format"
    return output_file

#%%
dataset_task_mappings_fp = "../../original/llm-predictions/dataset_task_mappings.csv"

# %%
# == MAIN ==
target_files = list(Path("../../original/llm-predictions/Llama-2-70b-chat-hf/all_predictions/") \
            .glob("*.csv"))

for p in target_files:
    output_file = separate_output_name(p.name)
    TASK_COL = "status_id" if output_file.dataset_id != 4 else "id" 
    
    try:
        mapping = LABEL_MAPPING[f"dataset{output_file.dataset_id}"][f"task{output_file.task_id}"]
    except KeyError:
        print(f"Skipping unsupported dataset/task combination: dataset {output_file.dataset_id}, task {output_file.task_id}")
        continue

    print(f"Processing file: {p.name}")

    # create and setup dirs
    parent = Path("../../")
    target = (parent / f"ds{output_file.dataset_id}task{output_file.task_id}") 
    target.mkdir(exist_ok=True)
    target /= "ai_workers"
    target.mkdir(exist_ok=True)
    target /= output_file.model_name
    target.mkdir(exist_ok=True)

    # read data
    df = pd.read_csv(p.absolute(), dtype={TASK_COL: str})
    # Get the expected labelset (from original paper codes)
    dataset_idx, dataset_task_mappings = load_dataset_task_prompt_mappings(
        dataset_num=output_file.dataset_id, task_num=output_file.task_id, dataset_task_mappings_fp=dataset_task_mappings_fp)
    label_column = dataset_task_mappings.loc[dataset_idx, "label_column"]
    labelset = dataset_task_mappings.loc[dataset_idx, "labelset_fullword"].split("; ")
    labelset = [label.strip() for label in labelset] 
    labelset += [label.upper() for label in labelset] 
    labelset_full_description = labelset
    num_rows = df.shape[0]

    worker_name = output_file.model_name + "___" + p.name.removesuffix(".csv")

    df["__label"] = df.prediction_ds.map(lambda x:
                                          process_output(x, output_file.task_id)
                                        )
    df["__gt"] = df[label_column].map(lambda label: map_label_to_completion(
        label=label, task_num=output_file.task_id, full_label=True))
    df["worker"] = worker_name
    df["task"] = df[TASK_COL].astype(str)
    new_df = df[["task", "worker", "__label", "__gt"]]
    new_df = pl.DataFrame(new_df)
    new_df = new_df.with_columns(pl.col("__label").map_elements(
        lambda x: mapping.get(x, None)
    ,return_dtype=pl.Int64).alias("label"))
    new_df = new_df.with_columns(pl.col("__gt").map_elements(
        lambda x: mapping.get(x, None)
    ,return_dtype=pl.Int64).alias("gt"))
    new_df = new_df.select([
        pl.col("task"),
        pl.col("worker"),
        pl.col("label"),
        pl.col("gt"),
    ])
    
    if new_df.filter(pl.col("task").str.contains("\.")).height > 0:
        print("Warning: Some labels could not be mapped to numerical values. Skipping file.")
        continue

    new_df.write_csv(
        target / (worker_name + "_transformed.csv")
    )
    with (target / (worker_name + "_metadata.json")).open('w') as f:
        f.write(str(output_file))



# %%
