# data/generate_csv.py
import os
import pandas as pd
import random

def generate_csv(data_dir, output_csv, val_ratio=0.2, shuffle=True):
    """
    Генерирует CSV файл с путями к аудиофайлам и метками.

    Args:
        data_dir (str): Путь к директории с данными (поддиректории REAL и FAKE).
        output_csv (str): Путь к выходному CSV файлу.
        val_ratio (float): Доля данных для валидации.
        shuffle (bool): Перемешивать ли данные.
    """

    real_dir = os.path.join(data_dir, "REAL")
    fake_dir = os.path.join(data_dir, "FAKE")

    real_files = [(os.path.join("REAL", f), 0) for f in os.listdir(real_dir) if f.endswith('.wav')]
    fake_files = [(os.path.join("FAKE", f), 1) for f in os.listdir(fake_dir) if f.endswith('.wav')]

    all_files = real_files + fake_files

    if shuffle:
        random.shuffle(all_files)

    val_size = int(len(all_files) * val_ratio)
    train_files = all_files[:-val_size]
    val_files = all_files[-val_size:]

    train_df = pd.DataFrame(train_files, columns=['filename', 'label'])
    val_df = pd.DataFrame(val_files, columns=['filename', 'label'])

    train_csv_path = output_csv.replace(".csv", "_train.csv")
    val_csv_path = output_csv.replace(".csv", "_val.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print(f"Train CSV saved to: {train_csv_path}")
    print(f"Validation CSV saved to: {val_csv_path}")

if __name__ == "__main__":
    data_dir = "data/raw"
    output_csv = "data/data.csv"
    generate_csv(data_dir, output_csv)