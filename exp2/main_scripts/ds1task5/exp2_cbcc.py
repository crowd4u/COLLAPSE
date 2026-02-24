
TARGET = "ds1task5"
LABELS = {
    0: "section_230",
    1: "trump_ban",
    2: "twitter_support",
    3: "platform_polices",
    4: "complaint",
    5: "other",
}
EXCEPT_LABELS = [5,]

SEEDS = [12347, 77777, 2984, 298, 1102644]

WRITE_APPEND = False

import sys
import os
from pathlib import Path
import polars as pl
import pandas as pd

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MAX_ITER = 5
n_classes = len(LABELS.keys())

sys.path.append(f"{DIR_PATH}/../")
from agg_methods_cbcc import get_aggregation_methods_CBCC
from io_utils import get_accuracy, get_recall

## Create result file
RESULT_CSV_PATH = f"{TARGET}_aggregation_results_cbcc.csv"
if not WRITE_APPEND:
    assert not os.path.exists(RESULT_CSV_PATH), f"{RESULT_CSV_PATH} already exists."
else:
    print(f"Appending to existing file: {RESULT_CSV_PATH}")
    
inst = pl.read_csv(f"{DIR_PATH}/insts.csv", truncate_ragged_lines=True)

def main():
    for ai_name, num_ai in inst.iter_rows():
        print(f"Processing combination: {ai_name}_{num_ai}")
        if num_ai != 0:
            for iter in range(MAX_ITER):
                human = pl.read_csv("datasets/human.csv")
                gt = human.select(["task", "gt"]).unique().sort("task").to_pandas()
                human = human.select(["task", "worker", "label"]).to_pandas()
                if ai_name == "mix":
                    AIs = ["f100","o250","l250"]
                else:
                    AIs = [ai_name]
                ai_cnt = 0
                for i in range(num_ai):
                    for ai in AIs:
                        aidf = pl.read_csv(f"datasets/{ai}.csv")
                        aidf = aidf.with_columns(
                            (pl.col("worker") + pl.lit(f"_wc_{ai_cnt}")).alias("worker")
                        )
                        ai_cnt += 1
                        if ai_cnt == 1:
                            ai_data = aidf
                        else:
                            ai_data = pl.concat([ai_data, aidf])
                ai = ai_data.to_pandas()
                #ai.to_csv("debug_ai.csv", index=False)
                #human.to_csv("debug_human.csv", index=False)
                # get aggregation methods
                agg_methods = get_aggregation_methods_CBCC(list(LABELS.keys()))
                for method in agg_methods:
                    print(f"Method: {method.name}, Iteration: {iter}")
                    ret = method.fit_predict(human, ai, seed=SEEDS[iter], human_only=False)
                    acc = get_accuracy(ret, gt.set_index("task"))
                    recalls = []
                    for key in LABELS.keys():
                        if key in EXCEPT_LABELS:
                            continue
                        recall = get_recall(
                            ret,
                            gt.set_index("task"),
                            biased_tasks=gt[gt["gt"]==key]["task"].to_list()
                        )
                        recalls.append(recall)
                    uc_text = method.get_uc_text(n_classes)
                    # write results
                    f.write(f"{ai_name},{num_ai},{method.name},{iter},{acc},")
                    for recall in recalls:
                        f.write(f"{recall},")
                    f.write(f"{uc_text}\n")
                    f.flush()
        else:
            for iter in range(MAX_ITER):
                # only human data
                human = pl.read_csv("datasets/human.csv")
                gt = human.select(["task", "gt"]).unique().sort("task").to_pandas()
                df = human.select(["task", "worker", "label"]).to_pandas()
                # get aggregation methods
                agg_methods = get_aggregation_methods_CBCC(list(LABELS.keys()))
                for method in agg_methods:
                    print(f"Method: {method.name}, Iteration: {iter}")
                    ret = method.fit_predict(df, ai=None, seed=SEEDS[iter], human_only=True)
                    acc = get_accuracy(ret, gt.set_index("task"))
                    recalls = []
                    for key in LABELS.keys():
                        if key in EXCEPT_LABELS:
                            continue
                        recall = get_recall(
                            ret,
                            gt.set_index("task"),
                            biased_tasks=gt[gt["gt"]==key]["task"].to_list()
                        )
                        recalls.append(recall)
                    uc_text = method.get_uc_text(n_classes)
                    # write results
                    f.write(f"{ai_name},{num_ai},{method.name},{iter},{acc},")
                    for recall in recalls:
                        f.write(f"{recall},")
                    f.write(f"{uc_text}\n")
                    f.flush()

if not WRITE_APPEND:
    with open(RESULT_CSV_PATH, "w") as f:
        header = "ai_name,num_ai,method,iter,accuracy,"
        f.write(header)
        for key in LABELS.keys():
            if key in EXCEPT_LABELS:
                continue
            f.write(f"recall_{LABELS[key]},")
        for i in range(n_classes):
            f.write(f"uc_pih_{i},")
        for i in range(n_classes-1):
            f.write(f"uc_pia_{i},")
        f.write(f"uc_pia_{i+1}")
        f.write("\n")
        main()
else:
    with open(RESULT_CSV_PATH, "a") as f:
        main()