
import sys, os
import pandas as pd
import subprocess
from pathlib import Path

PY2_BIN = r"/usr/bin/python2"
PY3_BIN = r"/usr/local/bin/python"

name2path = {
    "CATD" : r"/app/exp2/truth_infer_methods/c_CATD/method.py",
    "PM-CRH" : r"/app/exp2/truth_infer_methods/c_PM-CRH/method.py",
    "LFC" : r"/app/exp2/truth_infer_methods/l_LFCmulti/method.py",
    "ZC" : r"/app/exp2/truth_infer_methods/l_ZenCrowd/method.py",
    "LA1" : r"/app/exp2/truth_infer_methods/LA/data_pipeline.py",
    "LA2" : r"/app/exp2/truth_infer_methods/LA/data_pipeline.py",
}

class AggregationMethod:

    def __init__(self, name: str, workdir: str,  n_pass=None):
        self.name = name
        assert name in name2path.keys()
        if not name.startswith("LA"):
            self.python_version = 2
            assert n_pass is None
            self.n_pass = None
        else:
            self.python_version = 3
            assert n_pass in ["one", "two"]
            if n_pass == "one":
                self.n_pass = 1
            else:
                self.n_pass = 2
        self.workdir = workdir
        (Path(workdir) / "results").mkdir(exist_ok=True)
        (Path(workdir) / "datasets").mkdir(exist_ok=True)
        
    
    
    def fit_predict(self, df, iter, ds_name):
        seed = str(iter)*3
        df = df.rename(columns={"task":"question", "label":"answer"})
        datapath = str((Path(self.workdir) / "datasets" / (ds_name + ".csv")).absolute())
        df.to_csv(datapath, index=False)
        if self.n_pass is None and self.python_version == 2:
            filepath = str((Path(self.workdir) / "results" / f"{self.name}_{iter}_results.csv").absolute())
            subprocess.run(
                [
                    PY2_BIN,
                    name2path[self.name],
                    datapath,
                    "categorical",
                    filepath,
                    seed,
                ]
            )
        else:
            p = (Path(self.workdir) / "results" / f"{self.name}{self.n_pass}_{iter}_results.csv").absolute()
            filepath = str(p)
            if not p.exists():
                p1 = (Path(self.workdir) / "results" / f"{self.name}1_{iter}_results.csv").absolute()
                p2 = (Path(self.workdir) / "results" / f"{self.name}2_{iter}_results.csv").absolute()
                subprocess.run(
                [
                    PY3_BIN,
                    name2path[self.name],
                    datapath,
                    str(p1),
                    str(p2),
                    seed,
                ]
                ) 
            else:
                print("INFO: result file aleady exists. Reuse it.")
        result_df = pd.read_csv(filepath)
        os.remove(filepath)
        result_df = result_df.set_index("task")
        return result_df

    def get_uc_text(self, n_classes: int) -> str:
        return ("-1,"*(n_classes*2+1))[:-1]

    def is_converged(self):
        return -1

def get_aggregation_methods(workdir):
    return [
        AggregationMethod("CATD", workdir=workdir),
        AggregationMethod("LFC", workdir=workdir),
        AggregationMethod("ZC", workdir=workdir),
        AggregationMethod("PM-CRH", workdir=workdir),
        AggregationMethod("LA1", workdir=workdir, n_pass="one"),
        AggregationMethod("LA2", workdir=workdir, n_pass="two"),
    ]
