#%%

__all__ = [
    'CBCC',
]

import os
import subprocess

from typing import List, Optional
from numpy.typing import NDArray

import attr
import numpy as np
import pandas as pd

from crowdkit.aggregation.classification.majority_vote import MajorityVote
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.aggregation.utils import get_most_probable_labels, named_series_attrib


class CBCC():
    r"""
    CBBC crowd-kit wrapper
    Original Paper: https://dl.acm.org/doi/10.1145/2566486.2567989
    Implementation:
        Paper: https://dl.acm.org/doi/10.14778/3055540.3055547
        Repository: https://github.com/zhydhkcws/crowd_truth_infer/tree/master/methods/l_CBCC
    """

    def __init__(self, labels, C=4) -> None:
        self.labels = labels    
        self.K = len(labels)
        self.C = C
        print("CAUTION: DO NOT EXECUTE THIS PROGRAM DUPULICATELY. \n BECAUSE IT USES FIXED FILE PATH.")

    def fit_predict(self, data: pd.DataFrame, seed=12347) -> pd.Series:
        # Convert to CBCC format from crowd-kit format
        n = len(data['task'].unique())
        m = len(data['worker'].unique())
        K = self.K

        csv = []
        taskid2int = {task:i for i,task in enumerate(data['task'].unique())}
        workerid2int = {worker:i for i,worker in enumerate(data['worker'].unique())}
        label2int = {label:i for i,label in enumerate(self.labels)}

        for t,w,c in zip(data['task'], data['worker'], data['label']):
            csv.append({
                "worker": workerid2int[w],
                "task": taskid2int[t],
                "label": label2int[c]
            })
        csv = pd.DataFrame(csv)
        # Save to fixed path
        cf_path = os.path.dirname(__file__) + "/CBCC/Data/CF.csv"
        csv.to_csv(cf_path, index=False, header=False)
        exe_path = os.path.dirname(__file__) + f"/CBCC/CBCC.exe"
        print(exe_path)
        # Run CBCC
        subprocess.run([exe_path, str(seed), str(self.C)], cwd=os.path.dirname(__file__) + "/CBCC/")
        # Read result
        endpoint_path = os.path.dirname(__file__) + "/CBCC/Results/endpoints.csv"
        result = pd.read_csv(endpoint_path, header=None, names=["task",]+[f"c{i}" for i in range(K)])
        result = result.set_index("task")
        # Select maximum class
        result = get_most_probable_labels(result)
        # Replace correct label
        result = result.replace({f"c{v}":k for k,v in label2int.items()})
        # Replace task id
        result = result.rename(index={v:k for k,v in taskid2int.items()})
        # Clean up
        os.remove(cf_path)
        os.remove(endpoint_path)
        return result
   