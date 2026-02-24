#%%
import sys, os
import pandas as pd
from crowdkit.aggregation import DawidSkene, GLAD, OneCoinDawidSkene

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append(f"{DIR_PATH}/../methods")
from hsds_stan import SeparatedBDS, HSDS_Stan
from hsds_em import HSDS_EM

class AggregationMethod:

    def __init__(self, name: str, is_separated: bool, is_em: bool, obj: object):
        self.name = name
        self.is_separated = is_separated
        self.is_em = is_em
        self.obj = obj

    def fit_predict(self, human, ai):
        self.is_fitted = True
        if not self.is_em:
            self.__clear_output_dir()
        if self.is_separated:
            ret = self.obj.fit_predict(human, ai)
        else:
            ret = self.obj.fit_predict(pd.concat([human, ai], axis=0))
        return ret

    def get_uc_text(self, n_classes: int) -> str:
        assert self.is_fitted, "Method must be fitted before getting UC text."
        if not self.is_em:
            text = f"{self.obj.p_unconverged_count},"
            for c in self.obj.pih_unconverged_count:
                text += f"{c},"
            for c in self.obj.pia_unconverged_count:
                text += f"{c},"
            return text[:-1]
        else:
            return ("-1,"*(n_classes*2+1))[:-1]

    def __clear_output_dir(self):
        # remove all files in "./outputs/"
        output_dir = f"{DIR_PATH}/outputs/"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    print(f"Error: {e}")
        else:
            os.makedirs(output_dir)
    
    def is_converged(self):
        if not self.is_em:
            return  1 if self.obj.convergence else 0
        else:
            return -1

def get_BDS_instance(labels, iter_warmup, iter_sampling, r):
    infer_params = {
        "iter_warmup": iter_warmup,
        "iter_sampling": iter_sampling,
    }
    return SeparatedBDS(
        labels=labels,
        algorithm="mcmc",
        init_worker_accuracy=r,
        infer_params=infer_params,
    )

def get_HSDS_Stan_instance(labels, iter_warmup, iter_sampling, r):
    infer_params = {
        "iter_warmup": iter_warmup,
        "iter_sampling": iter_sampling,
    }
    return HSDS_Stan(
        labels=labels,
        algorithm="mcmc",
        init_worker_accuracy=r,
        infer_params=infer_params
    )

def get_aggregation_methods(labels, r=0.75, n_iter=100000):
    return [
        AggregationMethod("EMDS", False, True, DawidSkene(n_iter=n_iter)),
        AggregationMethod("GLAD", False, True, GLAD(n_iter=n_iter)),
        AggregationMethod("OneCoinDS", False, True, OneCoinDawidSkene(n_iter=n_iter)),
        AggregationMethod("HSDS_EM", True, True, HSDS_EM(n_iter=n_iter, r=r)),
        AggregationMethod("BDS(iter_sampling=1000)", True, False,
                          get_BDS_instance(labels, iter_warmup=500, iter_sampling=1000, r=r)),
        AggregationMethod("BDS(iter_sampling=2000)", True, False,
                          get_BDS_instance(labels, iter_warmup=1000, iter_sampling=2000, r=r)),
        AggregationMethod("BDS(iter_sampling=3000)", True, False,
                          get_BDS_instance(labels, iter_warmup=1500, iter_sampling=3000, r=r)),
        AggregationMethod("HSDS_MCMC(iter_sampling=1000)", True, False,
                          get_HSDS_Stan_instance(labels, iter_warmup=500//2, iter_sampling=1000//2, r=r)),
        AggregationMethod("HSDS_MCMC(iter_sampling=2000)", True, False,
                          get_HSDS_Stan_instance(labels, iter_warmup=1000//2, iter_sampling=2000//2, r=r)),
        AggregationMethod("HSDS_MCMC(iter_sampling=3000)", True, False,
                          get_HSDS_Stan_instance(labels, iter_warmup=1500//2, iter_sampling=3000//2, r=r)),
    ]
# %%
"""
a = get_aggregation_methods([0,1,2,3,4], r=0.75, n_iter=100000)
df = pd.read_csv(f"{DIR_PATH}/human_responses/face_r=5.csv")
ai = pd.read_csv(f"{DIR_PATH}/ai_responses/face_r=5_ai=mean_target=1_run=0.csv")
for method in a:
    print(f"Method: {method.name}")
    ret = method.fit_predict(df, ai)
    print(ret)
    print(method.get_uc_text(5))
# %%
"""