import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
import random
from datasets import load_dataset

class OUDataset(Dataset):
    def __init__(self, count=10000):
        theta = 1
        sigma =1
        dt = 0.05
        length=64
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

class AudioSetStream(IterableDataset):
    def __init__(self, seq_len=4096, target_class=None):
        super().__init__()
        self.seq_len = seq_len
        self.dataset = load_dataset("agkphysics/AudioSet", split="train", streaming=True, token=True)

    def __iter__(self):
        for item in self.dataset:
            if self.target_class:
                labels = item.get('human_labels', [])
                labels_lower = [label.lower() for label in labels]
                if not any(self.target_class in label for label in labels_lower):
                    continue
            audio_array = item['audio']['array']
            tensor = torch.tensor(audio_array, dtype=torch.float32)
            length = tensor.shape[0]
            if length < self.seq_len:
                pad_size = self.seq_len - length
                tensor = torch.nn.functional.pad(tensor, (0, pad_size))
            elif length > self.seq_len:
                start = random.randint(0, length - self.seq_len)
                tensor = tensor[start : start + self.seq_len]
            yield tensor.unsqueeze(0) * 0.9

def get_dataloader(dataset_name, batch_size=64, seq_len=100, count=10000, target_class=None):
    if dataset_name == "ou":
        dataset = OUDataset(count=count)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name == "audioset":
        dataset = AudioSetStream(seq_len=seq_len, target_class=target_class)
        return DataLoader(dataset, batch_size=batch_size)

    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported! Try 'ou' or 'audioset'.")