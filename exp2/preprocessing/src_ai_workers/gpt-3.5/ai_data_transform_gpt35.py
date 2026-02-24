#%%
# === MAPPING FROM LABEL LETTERS TO NUMERICAL VALUES FOR EACH DATASET AND TASK ===
LABEL_MAPPING = {
    "dataset1" : {
        "task1" : {
            "A" : 0,
            "B" : 1,
        },
        "task2" : {
            "A" : 0,
            "B" : 1,
            "C" : 2,
        },
        "task4" : {
            "A" : 0,
            "B" : 1,
            "C" : 2,
        },
        "task5" : {
            "A" : 0,
            "B" : 1,
            "C" : 2,
            "D" : 3,
            "E" : 4,
            "F" : 5,
        },
    },
    "dataset2" : {
        "task1" : {
            "A" : 0,
            "B" : 1,
        },
        "task2" : {
            "A" : 0,
            "B" : 1,
            "C" : 2,
        },
    },
    # "dataset3" : {
    #     "task1" : {
    #         "A" : 0,
    #         "B" : 1,
    #     },
    # },
    "dataset4" : {
        "task1" : {
            "A" : 0,
            "B" : 1,
        },
        "task2" : {
            "A" : 0,
            "B" : 1,
            "C" : 2,
        },
    },
}

# === END OF MAPPING ===

#%%
import sys
from pathlib import Path
import pandas as pd
import polars as pl
import json

from utils_src import load_dataset_task_prompt_mappings, map_label_to_completion

#%%
class OutputFile:
    model_name: str
    dataset_id : int
    task_id : int
    samples : int
    is_full: bool

    def __str__(self):
       return json.dumps({
              "model_name": self.model_name,
              "dataset_id": self.dataset_id,
              "task_id": self.task_id,
              "samples": self.samples,
              "is_full": self.is_full,
       })
    
def separate_output_name(filename: str) -> OutputFile:
    parts = filename.split('_')
    output_file = OutputFile()
    output_file.model_name = "gpt3.5"
    output_file.dataset_id = int(parts[1])
    output_file.task_id = int(parts[3])
    tmp_samples = parts[5].split('--')[0]
    if tmp_samples != "full":
        output_file.samples = int(tmp_samples)
        output_file.is_full = False
    else:
        output_file.samples = -1
        output_file.is_full = True
    return output_file

#%%
dataset_task_mappings_fp = "../../original/llm-predictions/dataset_task_mappings.csv"
# %%
# == MAIN ==
for parent in Path("../../original/llm-predictions/GPT-3.5") \
        .glob("*/"):
    for p in parent.glob("*.csv"):
        output_file = separate_output_name(p.name)
        TASK_COL = "status_id" if output_file.dataset_id != 4 else "id" 
        
        try:
            mapping = LABEL_MAPPING[f"dataset{output_file.dataset_id}"][f"task{output_file.task_id}"]
        except KeyError:
            print(f"Skipping unsupported dataset/task combination: dataset {output_file.dataset_id}, task {output_file.task_id}")
            continue

        print(f"Processing file: {p.name}")

        worker_name = output_file.model_name + "_samples_" + str(output_file.samples)

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

        zero_shot =  "full" in p.name
        # Get the expected labelset
        dataset_idx, dataset_task_mappings = load_dataset_task_prompt_mappings(
            dataset_num=output_file.dataset_id, task_num=output_file.task_id, dataset_task_mappings_fp=dataset_task_mappings_fp)
        label_column = dataset_task_mappings.loc[dataset_idx, "label_column"]
        labelset = dataset_task_mappings.loc[dataset_idx, "labelset"].split(",")
        labelset = [label.strip() for label in labelset]
        labelset_full_description = dataset_task_mappings.loc[dataset_idx, "labelset_fullword"].split("; ")

    if zero_shot:
        continue  # Currently skipping zero-shot files
    
    df["__label"] = df['prediction'].map(lambda label: map_label_to_completion(
    label=label, task_num=output_file.task_id, full_label=True))
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
