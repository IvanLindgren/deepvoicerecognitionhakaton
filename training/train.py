import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import TransformerClassifier
from data.utils import load_data, collate_fn
from training.config import TrainingConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from functools import partial
import numpy as np


def train(config):
    device = config.device

    # Обертка для передачи max_audio_length в collate_fn
    custom_collate_fn = partial(collate_fn, max_audio_length=config.max_audio_length)

    # Загрузка данных с пользовательской функцией collate_fn
    train_dataloader = load_data(
        config.data_dir, config.train_csv, batch_size=config.batch_size, collate_fn=custom_collate_fn
    )
    val_dataloader = load_data(
        config.data_dir, config.val_csv, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    # Инициализация модели
    model = TransformerClassifier(
        input_dim=config.input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()  # Для бинарной классификации

    best_val_f1 = 0  # Лучшая F1-метрика
    patience = 5
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        # Режим обучения
        model.train()
        train_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        for data, labels in train_dataloader:
            if data.dim() != 3:
                print(f"Invalid input shape: {data.shape}")
                continue

            data, labels = data.to(device), labels.unsqueeze(1).to(device)  # Добавляем ось для меток
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_train_labels.extend(labels.cpu().numpy().flatten())
            all_train_preds.extend(outputs.cpu().detach().numpy().flatten())

        avg_train_loss = train_loss / len(train_dataloader)
        train_f1 = f1_score(all_train_labels, (np.array(all_train_preds) > 0.5).astype(int), zero_division=0)

        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")

        # Режим валидации
        model.eval()
        val_loss = 0.0
        epoch_val_labels = []
        epoch_val_preds = []

        with torch.no_grad():
            for data, labels in val_dataloader:
                if data.dim() != 3:
                    print(f"Invalid input shape: {data.shape}")
                    continue

                data, labels = data.to(device), labels.unsqueeze(1).to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Преобразуем метки и предсказания в плоский список
                epoch_val_labels.extend(labels.cpu().numpy().flatten())
                epoch_val_preds.extend(outputs.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(epoch_val_labels, (np.array(epoch_val_preds) > 0.5).astype(int))
        val_precision = precision_score(epoch_val_labels, (np.array(epoch_val_preds) > 0.5).astype(int), zero_division=0)
        val_recall = recall_score(epoch_val_labels, (np.array(epoch_val_preds) > 0.5).astype(int), zero_division=0)
        val_f1 = f1_score(epoch_val_labels, (np.array(epoch_val_preds) > 0.5).astype(int), zero_division=0)

        print(f"Epoch {epoch + 1}/{config.epochs}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        # Сохранение лучшей модели
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config.model_path)
            epochs_no_improve = 0
            print(f"Saved best model at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1

        # Ранняя остановка
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training finished.")


if __name__ == "__main__":
    config = TrainingConfig()
    config.train_csv = os.path.join(config.project_root, "data", "data_train.csv")
    config.val_csv = os.path.join(config.project_root, "data", "data_val.csv")

    if not os.path.exists(config.train_csv) or not os.path.exists(config.val_csv):
        raise FileNotFoundError("CSV файлы не найдены. Убедитесь, что они существуют.")

    print("Конфигурация:")
    print(f"  train_csv: {config.train_csv}")
    print(f"  val_csv: {config.val_csv}")
    print(f"  data_dir: {config.data_dir}")
    print(f"  model_path: {config.model_path}")

    train(config)
