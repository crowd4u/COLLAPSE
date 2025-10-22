from pathlib import Path

import streamlit as st

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

METHODS = {
    "BDS(iter_sampling=1000)": "lightcoral",
    "BDS(iter_sampling=2000)": "red",
    "BDS(iter_sampling=3000)": "darkred",
    "HSDS_MCMC(iter_sampling=1000)": "lightblue",
    "HSDS_MCMC(iter_sampling=2000)": "blue",
    "HSDS_MCMC(iter_sampling=3000)": "darkblue",
    "HSDS_EM": "purple",
    "EMDS": "black",
    "OneCoinDS": "pink",
    "GLAD": "green",
    "CBCC_C=2" : "yellow",
    "CBCC_C=4" : "orange",
    "CBCC_C=8" : "darkorange",
}

st.title("Interactive Results Viewer for the Main Experiment")

st.markdown("## Configuration")

### Load the data
f = {
    "dog": 0,
    "face": 0, 
    "tiny": 0,
    "adult": 0,
}
dfs = {}
for p in Path("../experimental_results/results").glob("*.csv"):
    dataset = p.stem.split("_")[0]
    if f[dataset] == 0:
        dfs[dataset] = pl.read_csv(p, has_header=True, encoding="utf-8")
        f[dataset] += 1
    else:
        dfs[dataset] = pl.concat([dfs[dataset], pl.read_csv(p, has_header=True, encoding="utf-8")])
        f[dataset] += 1

for p in Path("../experimental_results/results_cbcc").glob("*.csv"):
    dataset = p.stem.split("_")[0]
    if f[dataset] == 0:
        dfs[dataset] = pl.read_csv(p, has_header=True, encoding="utf-8")
        # Results of CBCC may have negative accuracy due to estimation errors, so replace them with 0
        dfs[dataset] = dfs[dataset].with_columns([
            pl.when(pl.col("accuracy") < 0).then(0).otherwise(pl.col("accuracy")).alias("accuracy"),
            pl.when(pl.col("recall") < 0).then(0).otherwise(pl.col("recall")).alias("recall"),
        ])
        f[dataset] += 1
    else:
        dfs[dataset] = pl.concat([dfs[dataset], pl.read_csv(p, has_header=True, encoding="utf-8")])
        # Results of CBCC may have negative accuracy due to estimation errors, so replace them with 0
        dfs[dataset] = dfs[dataset].with_columns([
            pl.when(pl.col("accuracy") < 0).then(0).otherwise(pl.col("accuracy")).alias("accuracy"),
            pl.when(pl.col("recall") < 0).then(0).otherwise(pl.col("recall")).alias("recall"),
        ])
        f[dataset] += 1

## delete uc_ columns if exist 
for dataset in dfs:
    uc_cols = dfs[dataset].columns
    uc_cols = [col for col in uc_cols if col.startswith("uc_")]
    if uc_cols:
        dfs[dataset] = dfs[dataset].drop(uc_cols)

## Load human results
for i,p in enumerate(Path("../experimental_results/results_human").glob("*.csv")):
    if i == 0:
        human_df = pl.read_csv(p, has_header=True, encoding="utf-8")
        uc_cols = human_df.columns
        uc_cols = [col for col in uc_cols if col.startswith("uc_")]
        if uc_cols:
            human_df = human_df.drop(uc_cols)
    else:
        tmp = pl.read_csv(p, has_header=True, encoding="utf-8")
        uc_cols = tmp.columns
        uc_cols = [col for col in uc_cols if col.startswith("uc_")]
        if uc_cols:
            tmp = tmp.drop(uc_cols)
        human_df = pl.concat([human_df, tmp])

for p in Path("../experimental_results/results_human_cbcc").glob("*.csv"):
    tmp = pl.read_csv(p, has_header=True, encoding="utf-8")
    tmp = tmp.with_columns([
        pl.when(pl.col("accuracy") < 0).then(0).otherwise(pl.col("accuracy")).alias("accuracy"),
        pl.when(pl.col("recall") < 0).then(0).otherwise(pl.col("recall")).alias("recall"),
    ])
    uc_cols = tmp.columns
    uc_cols = [col for col in uc_cols if col.startswith("uc_")]
    if uc_cols:
        tmp = tmp.drop(uc_cols)
    human_df = pl.concat([human_df, tmp])


## Process human_df to expand "*" values
if 'human_df' in locals():
    expanded_human_dfs = []
    
    all_scenarios = []
    all_ai_accs = []
    
    for dataset_name, df in dfs.items():
        scenarios = df.select("scenario").unique().to_series().to_list()
        ai_accs = df.select("ai_acc").unique().to_series().to_list()
        all_scenarios.extend(scenarios)
        all_ai_accs.extend(ai_accs)
    
    all_scenarios = list(set(all_scenarios))
    all_ai_accs = list(set(all_ai_accs))
    
    for _, row in human_df.to_pandas().iterrows():
        row_data = row.to_dict()
        
        if 'scenario' in row_data and row_data['scenario'] == '*':
            scenarios_to_expand = all_scenarios
        else:
            scenarios_to_expand = [row_data.get('scenario', 'homo')]
        
        if 'ai_acc' in row_data and row_data['ai_acc'] == '*':
            ai_accs_to_expand = all_ai_accs
        else:
            ai_accs_to_expand = [row_data.get('ai_acc', 'mean')]
        
        for scenario in scenarios_to_expand:
            for ai_acc in ai_accs_to_expand:
                expanded_row = row_data.copy()
                expanded_row['scenario'] = scenario
                expanded_row['ai_acc'] = ai_acc
                expanded_row['num_ai'] = 0
                expanded_human_dfs.append(expanded_row)
    
    if expanded_human_dfs:
        expanded_human_df = pl.DataFrame(expanded_human_dfs)
        
        for dataset_name in dfs.keys():
            dataset_human = expanded_human_df.filter(pl.col("dataset") == dataset_name)
            if not dataset_human.is_empty():
                dfs[dataset_name] = pl.concat([dfs[dataset_name], dataset_human])

metric = st.selectbox(
    "Metric", ["accuracy", "recall"]
)

compare_target = st.selectbox(
    "Comparison", ["dataset", "r", "ai_acc", "scenario"]
)

st.markdown("Please specify params other than those specified in `Comparison` below.\n")
st.markdown("**Note that impossible experimental conditions will produce broken plots, e.g., results for `tiny` data other than `r=2`, or results for `+sigma` other than `face` data.**")

dataset = st.selectbox(
    "Dataset", ["dog", "face", "tiny", "adult"]
)

if dataset == "dog":
    r_options = [3,5,10]
elif dataset == "face":
    r_options = [3,5]
elif dataset == "tiny":
    r_options = [2]
elif dataset == "adult":
    r_options = [3,5]

r_num = st.selectbox("Number of human workers per task (redundancy; `r` / $r$)", r_options)

ai_acc = st.selectbox(
    "Performance Level of Biased_AI (`ai_acc` / $a_{AI}$)", ["-sigma", "mean", "+sigma", "max"]
)

scenario = st.selectbox(
    "Scenario (`scenario`)", ["homo", "hetero", "hybrid"]
)

st.markdown("## Results")

fdfs = []
if compare_target == "dataset":
    options = dfs.keys()
    for dataset in options:
        fdf = dfs[dataset].filter(
            (pl.col("r") == r_num) & (pl.col("ai_acc") == ai_acc) & (pl.col("scenario") == scenario)
        )
        
        fdf = fdf.group_by(["dataset","r", "num_ai", "method"]).agg(
            [
                pl.col("accuracy").mean().alias("mean_accuracy"),
                pl.col("accuracy").std().alias("std_accuracy"),
                pl.col("recall").mean().alias("mean_recall"),
                pl.col("recall").std().alias("std_recall"),
            ]
        ).to_pandas()
        fdfs.append(fdf)
else:
    mdf = dfs[dataset]
    if compare_target == "r":
        mdf = mdf.filter(
            (pl.col("dataset") == dataset) & (pl.col("ai_acc") == ai_acc) & (pl.col("scenario") == scenario)
        )
    elif compare_target == "ai_acc":
        mdf = mdf.filter(
            (pl.col("dataset") == dataset) & (pl.col("r") == r_num) & (pl.col("scenario") == scenario)
        )
    elif compare_target == "scenario":
        mdf = mdf.filter(
            (pl.col("dataset") == dataset) & (pl.col("r") == r_num) & (pl.col("ai_acc") == ai_acc)
        )
    options = mdf.select(pl.col(compare_target)).unique().sort(compare_target).to_series().to_list()
    for option in options:
        fdf = mdf.filter(pl.col(compare_target) == option)
        fdf = fdf.group_by(["r", "num_ai", "method"]).agg(
            [
                pl.col("accuracy").mean().alias("mean_accuracy"),
                pl.col("accuracy").std().alias("std_accuracy"),
                pl.col("recall").mean().alias("mean_recall"),
                pl.col("recall").std().alias("std_recall"),
            ]
        ).to_pandas()
        fdfs.append(fdf)

# Plotting
figs = []
chart_data_list = [] 

for i, (fdf, opt) in enumerate(zip(fdfs, options)):
    fig, ax = plt.subplots(figsize=(5, 3))
    if fdf.empty:
        chart_data_list.append(None)
        figs.append(fig) 
        continue
    
    used_data = []
    
    for method, color in METHODS.items():
        tdf = fdf[fdf["method"] == method]
        if not tdf.empty:
            tdf = tdf.sort_values("num_ai")
            method_new = method.replace("Two-Step", "HS-DS")
            method_new = method_new.replace("TwoStep", "HS-DS")
            
            for _, row in tdf.iterrows():
                used_data.append({
                    "method": method,
                    "num_ai": row["num_ai"],
                    "mean_accuracy": row["mean_recall"] if metric == "recall" else row["mean_accuracy"],
                    "std_accuracy": row["std_recall"] if metric == "recall" else row["std_accuracy"]
                })
            
            ax.errorbar(
                tdf["num_ai"], 
                tdf["mean_recall"] if metric == "recall" else tdf["mean_accuracy"], 
                yerr=tdf["std_recall"] if metric == "recall" else tdf["std_accuracy"], 
                label=method_new,
                color=color, 
                capsize=5,
                fmt='o-',
            )
    
    chart_data_list.append(pd.DataFrame(used_data) if used_data else None)
    
    ax.set_title(f"{compare_target}: {opt}")        
    ax.set_xlabel("Number of AI Workers")
    ax.set_ylabel("Recall" if metric == "recall" else "Overall Accuracy")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 35)
    ax.grid(True)
    figs.append(fig)  

for i, (fig, opt) in enumerate(zip(figs, options)):
    st.pyplot(fig, clear_figure=True)
    
    with st.expander(f"Formatted Data: {compare_target} = {opt}"):
        if chart_data_list[i] is not None:
            chart_data = chart_data_list[i]
            
            st.dataframe(chart_data, use_container_width=True)
            
            csv = chart_data.to_csv(index=False)
            st.download_button(
                label=f"Download CSV ({compare_target}={opt})",
                data=csv,
                file_name=f"chart_data_{compare_target}_{opt}_{metric}.csv",
                mime="text/csv",
                key=f"download_{i}_{compare_target}_{opt}_{metric}"
            )
        else:
            st.write("No data available")
    
    st.markdown("---")



