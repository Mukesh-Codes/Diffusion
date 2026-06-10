import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class OUDataset(Dataset):
    def __init__(self, theta=1, sigma=1, dt=0.05, length=64, count=10000):
        print(f"Generating {count} OU trajectories")
        a = np.exp(-theta * dt)
        q = (sigma ** 2) / (2 * theta)
        step_std = np.sqrt(q * (1 - a**2))
        dataset = np.zeros((count, length))
        dataset[:, 0] = 0.0

        for t in range(1, length):
            epsilon = np.random.normal(0, 1, size=count)
            dataset[:, t] = (a * dataset[:, t-1]) + (step_std * epsilon)
        self.data = torch.tensor(dataset, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(batch_size=64, **kwargs):
    dataset = OUDataset(**kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)