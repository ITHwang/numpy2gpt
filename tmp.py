import numpy as np

import torch

x: torch.Tensor = torch.tensor(data=np.array([1, 2, 3]))
y: torch.Tensor = torch.tensor(data=np.array([4, 5, 6]))
z = x * y
if isinstance(z, torch.Tensor):
    z.backward()

print(z)
