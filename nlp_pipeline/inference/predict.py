"""
Get ordered predictions - needed for both Kaggle and evaluation of your mdoels.
"""
import numpy as np
import torch.nn as nn


def get_ordered_preds(learner, ds_type):
    preds = learner.get_preds(ds_type=ds_type, activ=nn.Sigmoid())
    sampler = [i for i in learner.data.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    preds = [p[reverse_sampler] for p in preds]
    return preds
