import torch
import numpy as np

ascend = np.arange(10)
descend = np.arange(10,0,-1)

ascend = torch.from_numpy(ascend).unsqueeze(1)
descend = torch.from_numpy(descend).unsqueeze(1)

print(torch.min(ascend,descend))
