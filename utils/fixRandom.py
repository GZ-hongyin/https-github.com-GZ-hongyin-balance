# 固定随机数
# 已经验证，和在代码中直接使用的效果一致
#
# YZ 2020/03/21

import random
import numpy as np
import torch

def fix_random(seed=0):
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 针对数据的分布自动实现寻找卷积效率最高的方式(如果是true)
        torch.backends.cudnn.deterministic = True  # 实现卷积的方式是巩固定的
    setup_seed(seed)