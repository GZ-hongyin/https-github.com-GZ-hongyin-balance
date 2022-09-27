import torch
import numpy as np
from matplotlib import pyplot as plt


def acc(output,label):
    count = label.shape[0]
    true_count = (output == label).sum()
    acc = true_count/count
    return acc