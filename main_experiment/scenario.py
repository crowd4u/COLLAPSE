#%%
import os
import polars as pl

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MAX_RUN = 30 - 1

#%%
def create_ai_dataset(scenario: str, dataset_name: str, r : int, ai_acc: str, n_classes: int):
    i = n_classes - 1
    while True:
        ai_num = 0
        if scenario == "homo":
            for j in range(i):
                tmp_df = pl.read_csv(
                    f"{DIR_PATH}/ai_responses/{dataset_name}_r={r}_ai={ai_acc}_target={1}_run={j}.csv",
                    dtypes={"task": pl.String, "worker": pl.String, "label": pl.Int64}
                )
                print("Loading: ", f"{dataset_name}_r={r}_ai={ai_acc}_target={1}_run={j}.csv")
                ai_num += 1
                if j == 0:
                    df = tmp_df
                else:
                    df = pl.concat([df, tmp_df], how="vertical")
            yield df.to_pandas(), ai_num
        elif scenario == "hetero":
            dfs = []
            for j in range(i//(n_classes-1)):
                for k in range(1, n_classes):
                    tmp_df = pl.read_csv(
                        f"{DIR_PATH}/ai_responses/{dataset_name}_r={r}_ai={ai_acc}_target={k}_run={j}.csv",
                        dtypes={"task": pl.String, "worker": pl.String, "label": pl.Int64}
                    )
                    tmp_df = tmp_df.with_columns((pl.col("worker") + f"_{k}").alias("worker"))
                    print("Loading: ", f"{dataset_name}_r={r}_ai={ai_acc}_target={k}_run={j}.csv")
                    ai_num += 1
                    dfs.append(tmp_df)
            df = pl.concat(dfs, how="vertical")
            yield df.to_pandas(), ai_num
        elif scenario == "hybrid":
            if i % (2 * (n_classes - 1)) == 0:
                dfs = []
                for j in range(i//(2*(n_classes-1))):
                    for k in range(1, n_classes):
                        tmp_df = pl.read_csv(
                            f"{DIR_PATH}/ai_responses/{dataset_name}_r={r}_ai={ai_acc}_target={k}_run={j}.csv",
                            dtypes={"task": pl.String, "worker": pl.String, "label": pl.Int64}
                        )
                        tmp_df = tmp_df.with_columns((pl.col("worker") + f"_{k}_1").alias("worker"))
                        print("Loading: ", f"{dataset_name}_r={r}_ai={ai_acc}_target={k}_run={j}.csv")
                        ai_num += 1
                        dfs.append(tmp_df)
                    for k in range(1, n_classes):
                        index = MAX_RUN - ((n_classes - 1) * j + k) + 1
                        tmp_df = pl.read_csv(
                            f"{DIR_PATH}/ai_responses/{dataset_name}_r={r}_ai={ai_acc}_target={1}_run={index}.csv",
                            dtypes={"task": pl.String, "worker": pl.String, "label": pl.Int64}
                        )
                        tmp_df = tmp_df.with_columns((pl.col("worker") + f"_{k}_2").alias("worker"))
                        print("Loading: ", f"{dataset_name}_r={r}_ai={ai_acc}_target={1}_run={index}.csv")
                        ai_num += 1
                        dfs.append(tmp_df)
                df = pl.concat(dfs, how="vertical")
                yield df.to_pandas(), ai_num
        else:
            raise ValueError("Invalid scenario type. Choose from 'homo', 'hetero', or 'hybrid'.")
        i += (n_classes - 1)


# %%
# a = create_dataset("hybrid", "face", 5, "mean", 4)
# for i, (df, ai_num) in enumerate(a):
#     print(i, ai_num)
#     if i == 4:
#         break
# # %%
# DIR_PATH
# # %%
