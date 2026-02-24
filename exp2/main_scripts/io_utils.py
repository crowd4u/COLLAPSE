
import sys
import os
import pandas as pd
import json

from sklearn.metrics import accuracy_score

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

def get_accuracy(ret, gt):
    y_true = gt.sort_index().astype(str)
    y_pred = ret.sort_index().astype(str)
    acc = accuracy_score(y_true, y_pred)
    return acc

def get_recall(ret, gt, biased_tasks):
    return get_accuracy(
        ret.loc[biased_tasks],
        gt.loc[biased_tasks]
    )     

