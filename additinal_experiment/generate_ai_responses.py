import numpy as np
import polars as pl
import pandas as pd

import sys
sys.path.append(f"../main_experiment/preprocessing/")

from ai_simulator import generate_ai_dataset

np.random.seed(777)

df = pd.read_csv("human_responses_with_gt.csv", 
                 dtype={"task" : str, "worker" :str ,"label" : str ,"gt" : int})

gt = df.filter(["task","gt"]).drop_duplicates(keep='last').set_index("task")

#flan 250 CM
CM = np.array([
            [0.99319728, 0.        , 0.        , 0.        , 0.        ,  0.00680272],
            [0.        , 1.        , 0.        , 0.        , 0.        ,  0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ,  1.        ],
            [0.12      , 0.12      , 0.        , 0.        , 0.36      ,  0.4       ],
            [0.03846154, 0.34615385, 0.        , 0.        , 0.11538462 ,  0.5      ],
            [0.        , 0.        , 0.        , 0.        , 0.         ,  0.       ],
       ])

NUM_AI = 10

for i in range(NUM_AI):
    ai_df = generate_ai_dataset(CM, gt, f"AI_{i}", [0,1,2,3,4,5])
    ai_df.to_csv(f"AI_{i}.csv", index=False)
