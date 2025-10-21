import numpy as np
import pandas as pd

def generate_confusion_matrix(n_classes, r=0.75, n_affected_class=1, *, flat=False, fixed_index=None):
    l = (1-r)/(n_classes-1)
    flat_cm = np.full((n_classes, n_classes), l)
    np.fill_diagonal(flat_cm, r)
    if flat:
        return flat_cm
    cm  = flat_cm
    if fixed_index is not None:
        for idx in fixed_index:
            cm[idx[0], idx[1]] = r
            cm[idx[0], idx[0]] = l
        return cm
    affected_classes = np.random.choice(n_classes, n_affected_class, replace=False)
    affected_cols = np.random.choice(n_classes-1, n_affected_class, replace=True)
    for i in range(n_affected_class):
        cm[affected_classes[i], :] = l
        col_index = 0
        if affected_cols[i] < affected_classes[i]:
            col_index = affected_cols[i]
        else:
            col_index = affected_cols[i] + 1
        cm[affected_classes[i], col_index] = r
    return cm

def get_dataset_human_acc(df, gt):
    df = df.merge(gt, on="task", how="left")
    df["correct"] = (df["label"] == df["gt"]).astype(int)
    acc = df.groupby("worker")["correct"].mean().reset_index()
    acc.columns = ["worker", "accuracy"]
    acc_mean = acc["accuracy"].mean()
    acc_std = acc["accuracy"].std()
    return acc_mean, acc_std

def generate_confusion_matrix_neo(k, human_acc_mean, human_acc_std, mode, *, fixed_target=1):
    assert k > 2 
    a = None
    if mode == "-sigma":
        a = human_acc_mean - human_acc_std
    if mode == "-2sigma":
        a = human_acc_mean - 2*human_acc_std
    if mode == "-3sigma":
        a = human_acc_mean - 3*human_acc_std
    elif mode == "mean":
        a = human_acc_mean
    elif mode == "+sigma":
        a = human_acc_mean + human_acc_std
        assert a <= (k-1)/k
    elif mode == "max":
        a = (k-1)/k
    assert a is not None
    r = (k * a * (k-1) - 1) / (k *  (k -2))
    return generate_confusion_matrix(k, r=r, fixed_index=[[0,fixed_target]])

def generate_ai_dataset(cm, gt, ai_name, labels):
    rows=[]
    for index,record in gt.reset_index().iterrows():
        tid = record["task"]
        truth = record["gt"]
        prod = cm[truth]
        label = np.random.choice(labels,size=1,p=prod)[0]
        row = {
            "task" : tid,
            "worker": ai_name,
            "label" : label
        }
        rows.append(row)
    return pd.DataFrame(rows)


"""
import polars as pl
df = pl.read_csv("datasets/crowd_truth_infer/s4_Dog data/dogs-1.merged.label.tsv",
                     separator="\t",
                     has_header=False,
                     new_columns=["task","worker","label"]
                )
gt = pl.read_csv("datasets/crowd_truth_infer/s4_Dog data/truth.csv",
                     new_columns=["task","gt"]
                )
gt = gt.to_pandas().set_index("task")
#cm = generate_confusion_matrix(4, r=0.75, n_affected_class=4)
acc_mean, acc_std = get_dataset_human_acc(df.to_pandas(), gt)
print(f"acc_mean: {acc_mean}, acc_std: {acc_std}")
cm = generate_confusion_matrix_neo(4, acc_mean, acc_std, "-sigma")
print(cm)
ai_name = "ai1"
ai_df = generate_ai_dataset(cm, gt, ai_name, labels=[0,1,2,3])
print(ai_df, gt)
from dataset_preprocessing import get_ha_count_and_ratio
get_ha_count_and_ratio(df, ai_df)
# """