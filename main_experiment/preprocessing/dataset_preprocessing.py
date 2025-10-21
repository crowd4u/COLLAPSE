import polars as pl
import pandas as pd

def change_r(df: pl.DataFrame, new_r,*,seed=777):
    now_r = df.select(pl.len() / df.unique("task").select(pl.len())).to_numpy()[0][0]
    print("Now Reducdancy:", now_r)

    rand_df = df.filter(
        pl.int_range(pl.len()).shuffle(seed=seed).over("task") < new_r
    )
    
    new_r = rand_df.select(pl.len() / rand_df.unique("task").select(pl.len())).to_numpy()[0][0]
    print("New Reducdancy:", new_r)
    return rand_df

def get_ha_count_and_ratio(human_df: pd.DataFrame, ai_df: pd.DataFrame):
    human_count:int = len(human_df)
    ai_count = len(ai_df)
    print("Human Count:", human_count)
    print("AI Count:", ai_count)
    ha_count = human_count + ai_count
    ha_ratio = human_count / ha_count
    ah_ratio = ai_count / ha_count
    print("%Human: ", ha_ratio)
    print("%AI: ", ah_ratio)
    return human_count, ai_count, ha_count, ha_ratio, ah_ratio