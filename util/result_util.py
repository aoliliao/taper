import os
import random

import numpy as np
import torch


def set_fixed_seed(seed):
    # seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True

