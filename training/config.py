# training/config.py
import torch
import os

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TrainingConfig:
    def __init__(self):
        self.project_root = get_project_root()
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(self.project_root, "models/best_model.pth")
        self.input_dim = 128  # Размерность мел-спектрограммы
        self.d_model = 256   # Размерность модели трансформера
        self.nhead = 8       # Количество "голов" в механизме внимания
        self.num_layers = 4   # Количество слоев трансформера
        self.num_classes = 1  # Один выход для вероятности дипфейка
        self.dropout = 0.1    # Dropout
        self.max_audio_length = 500  # Максимальная длина аудио (в количестве временных шагов)
        self.data_dir = os.path.join(self.project_root, "data/raw")
        self.train_csv = ""
        self.val_csv = ""