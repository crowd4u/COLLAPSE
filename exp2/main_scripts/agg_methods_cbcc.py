#%%
import sys, os
import pandas as pd

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append(f"{DIR_PATH}/../../methods")

from cbcc_wrapper import CBCC

class AggregationMethod:

    def __init__(self, name: str, is_separated: bool, is_em: bool, obj: object):
        self.name = name
        self.is_separated = is_separated
        self.is_em = is_em
        self.obj = obj

    def fit_predict(self, human, ai, human_only=False, seed=12345):
        self.is_fitted = True
        if not self.is_em:
            self.__clear_output_dir()
        if human_only==False:
            if self.is_separated:
                ret = self.obj.fit_predict(human, ai, seed=seed)
            else:
                ret = self.obj.fit_predict(pd.concat([human, ai], axis=0), seed=seed)
        else:
            assert self.is_separated==False, "Human-only fitting only supported for non-separated methods."
            ret = self.obj.fit_predict(human, seed=seed)
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

def get_aggregation_methods_CBCC(labels, r=0.75, n_iter=100000):
    return [
        AggregationMethod("CBCC_M=2", False, True, CBCC(labels, C=2)),
        AggregationMethod("CBCC_M=4", False, True, CBCC(labels, C=4)),
        AggregationMethod("CBCC_M=8", False, True, CBCC(labels, C=8)),
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