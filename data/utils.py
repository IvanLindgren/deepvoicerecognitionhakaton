import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None, target_sr=16000):
        self.data_info = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sr = target_sr
        self.filter_data()

    def filter_data(self):
        # Убираем файлы, которых нет или которые некорректны
        self.data_info = self.data_info[
            self.data_info.iloc[:, 0].apply(lambda x: os.path.exists(os.path.join(self.audio_dir, x)))
        ]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data_info.iloc[idx, 0])
        label = self.data_info.iloc[idx, 1]
        audio = self.load_and_process_audio(audio_path)
        if self.transform:
            audio = self.transform(audio)
        return audio, label

    def load_and_process_audio(self, audio_path):
        try:
            audio, _ = librosa.load(audio_path, sr=self.target_sr)
            if len(audio) < self.target_sr:  # Минимальная длина 1 секунда
                raise ValueError("Audio too short")
            return audio
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return np.zeros(self.target_sr)


def to_spectrogram_tensor(audio, n_fft=2048, hop_length=512, n_mels=128):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return torch.tensor(spectrogram_db, dtype=torch.float32).T

def truncate_or_pad(tensor, length):
    """
    Обрезает или дополняет тензор до заданной длины.

    Args:
        tensor (torch.Tensor): Входной тензор.
        length (int): Требуемая длина.

    Returns:
        torch.Tensor: Тензор фиксированной длины.
    """
    if tensor.size(0) > length:
        return tensor[:length]
    else:
        padding = torch.zeros(length - tensor.size(0), tensor.size(1))
        return torch.cat((tensor, padding), dim=0)


def collate_fn(batch, max_audio_length=500):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0)

    audio, labels = zip(*batch)

    # Приводим все аудио к одной длине
    def truncate_or_pad(tensor, length):
        if tensor.size(0) > length:
            return tensor[:length]
        else:
            padding = torch.zeros(length - tensor.size(0), tensor.size(1))
            return torch.cat((tensor, padding), dim=0)

    audio = torch.stack([truncate_or_pad(x, max_audio_length) for x in audio])
    labels = torch.tensor(labels, dtype=torch.float32)
    return audio, labels


def load_data(data_dir, csv_file, batch_size, shuffle=True, collate_fn=None):
    dataset = AudioDataset(csv_file=csv_file, audio_dir=data_dir, transform=to_spectrogram_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
