import json

import numpy as np
import pandas as pd
import polars as pl

np.random.seed(777)

def change_r(df: pl.DataFrame, new_r,*,seed=777):
    now_r = df.select(pl.len() / df.unique("task").select(pl.len())).to_numpy()[0][0]
    print("Now Reducdancy:", now_r)

    rand_df = df.filter(
        pl.int_range(pl.len()).shuffle(seed=seed).over("task") < new_r
    )
    
    new_r = rand_df.select(pl.len() / rand_df.unique("task").select(pl.len())).to_numpy()[0][0]
    print("New Reducdancy:", new_r)
    return rand_df

def write_data(name, raw_df_pl, r_range, gt):
    for r in r_range:
        df = change_r(raw_df_pl, r)
        df.write_csv(f"../human_responses/{name}_r={r}.csv")
    gt.write_csv(f"../human_responses/{name}_gt.csv")

def save_dataset_profile(name, r_range, n_classes, labels):
    data = {}
    data["dataset_name"] = name
    data["r_range"] = r_range
    data["n_classes"] = n_classes
    data["labels"] = labels
    json.dump(data, open(f"../human_responses/{name}_dataset_profile.json", "w"), indent=4)

if __name__ == "__main__":
    ## Dog dataset
    print("Processing Dog dataset")
    raw_df_pl = pl.read_csv(f"./raw_datasets/s4_Dog data/dogs-1.merged.label.tsv",
                separator="\t",
                has_header=False,
                new_columns=["task","worker","label"]
        )
    gt = pl.read_csv(f"./raw_datasets/s4_Dog data/truth.csv",
                    new_columns=["task","gt"]
            )
    n_classes = 4
    r_range = [3,5,10]
    labels = [0,1,2,3]
    name = "dog"
    write_data(name, raw_df_pl, r_range, gt)
    save_dataset_profile(name, r_range, n_classes, labels)
    ## Face dataset
    print("Processing Face dataset")
    raw_df_pl = pl.read_csv(f"./raw_datasets/s4_Face Sentiment Identification/answer.csv",
                     separator=",",
                     has_header=True,
                     new_columns=["task","worker","label"]
                )
    gt = pl.read_csv(f"./raw_datasets/s4_Face Sentiment Identification/truth.csv",
                    new_columns=["task","gt"]
            )
    n_classes = 4
    r_range = [3,5] # Avg. 8.96
    labels = [0,1,2,3]
    name = "face"
    write_data(name, raw_df_pl, r_range, gt)
    save_dataset_profile(name, r_range, n_classes, labels)
    ## Tiny
    print("Processing Tiny dataset")
    raw_df_pl = pl.read_csv(f"./raw_datasets/tiny_imagenet.csv",
                separator=",",
                has_header=True,
        )
    gt = raw_df_pl[["task", "gt"]].unique()
    raw_df_pl = raw_df_pl[["task", "worker", "label"]]
    n_classes = 5
    r_range = [2,]
    labels = [0,1,2,3,4]
    name = "tiny"
    write_data(name, raw_df_pl, r_range, gt)
    save_dataset_profile(name, r_range, n_classes, labels)
    ## Adult
    print("Processing Adult dataset")
    np.random.seed(777)
    df = pd.read_csv("./raw_datasets/tlkaggftrs.csv", dtype={"task" : str, "worker" :str ,"label" : str ,"gt" : str})
    gt = df.filter(["task","gt"]).drop_duplicates(keep='last')
    df = df.drop(["gt"], axis=1)

    n_classes = 5
    r_range = [3,5]
    labels = [0,1,2,3,4]
    name = "adult"

    df["label"] = (df["label"].astype(int) - 1).astype(str)
    gt["gt"] = (gt["gt"].astype(int) - 1).astype(str)

    df = pl.from_pandas(df)
    gt = pl.from_pandas(gt)

    raw_df_pl = df
    write_data(name, raw_df_pl, r_range, gt)
    save_dataset_profile(name, r_range, n_classes, labels)
