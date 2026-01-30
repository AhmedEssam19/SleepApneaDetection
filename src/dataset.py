import torch

from torch.utils.data import Dataset
from scipy import signal
from config import CONFIG

class SleepApneaDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        _, _, spectrogram = signal.spectrogram(
            sample,
            fs=CONFIG.spectrogram.sample_rate,
            nperseg=CONFIG.spectrogram.window_size,
            noverlap=CONFIG.spectrogram.window_overlap
        )
        label = self.labels[idx]
        if self.transform:
            spectrogram = self.transform(torch.tensor(spectrogram)).float()
        return spectrogram, label
    
    def get_statistics(self):
        _, _, spectrogram = signal.spectrogram(
            self.data,
            fs=CONFIG.spectrogram.sample_rate,
            nperseg=CONFIG.spectrogram.window_size,
            noverlap=CONFIG.spectrogram.window_overlap
        )

        if self.transform:
            transformed_data = self.transform(torch.tensor(spectrogram))
        else:
            transformed_data = spectrogram

        mean_val = torch.mean(transformed_data)
        std_val = torch.std(transformed_data)

        return mean_val, std_val


class EEGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(torch.tensor(sample)).float()
        return torch.tensor(sample, dtype=torch.float32), label
    
class HeartRateDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        f, _, spectrogram = signal.spectrogram(
            sample,
            fs=CONFIG.spectrogram.sample_rate,
            nperseg=CONFIG.spectrogram.window_size,
            noverlap=CONFIG.spectrogram.window_overlap
        )

        freq_mask = (f >= CONFIG.spectrogram.min_freq) & (f <= CONFIG.spectrogram.max_freq)
        spectrogram = spectrogram[freq_mask, :]
        label = self.labels[idx]
        spectrogram = torch.tensor(spectrogram).unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(spectrogram).float()

        return spectrogram, torch.tensor(label, dtype=torch.float32)
    
    def get_statistics(self):
        f, _, spectrogram = signal.spectrogram(
            self.data,
            fs=CONFIG.spectrogram.sample_rate,
            nperseg=CONFIG.spectrogram.window_size,
            noverlap=CONFIG.spectrogram
            .window_overlap
        )

        freq_mask = (f >= CONFIG.spectrogram.min_freq) & (f <= CONFIG.spectrogram.max_freq)
        spectrogram = spectrogram[:, freq_mask, :]
        spectrogram = torch.tensor(spectrogram).unsqueeze(1)

        if self.transform:
            transformed_data = self.transform(spectrogram)
        else:
            transformed_data = spectrogram

        mean_val = torch.mean(transformed_data)
        std_val = torch.std(transformed_data)

        return mean_val, std_val
