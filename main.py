import os
import tkinter as tk
from tkinter import filedialog, messagebox
from training.train import train
from inference.modelpredict import predict
from data.generate_csv import generate_csv
from training.config import TrainingConfig
import random


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

class DeepFakeApp:
    def __init__(self, master):
        self.master = master
        master.title("Обнаружение дипфейков аудио")

        self.project_root = get_project_root()
        self.config = TrainingConfig()

        # Инициализируем пути с использованием project_root
        self.data_dir = tk.StringVar(value=os.path.join(self.project_root, "data/raw"))
        self.output_csv = tk.StringVar(value=os.path.join(self.project_root, "data/data.csv"))
        self.audio_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Фрейм для работы с CSV
        csv_frame = tk.LabelFrame(self.master, text="Генерация CSV")
        csv_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        tk.Label(csv_frame, text="Директория с данными:").grid(row=0, column=0, sticky="w")
        tk.Entry(csv_frame, textvariable=self.data_dir, width=30).grid(row=0, column=1)
        tk.Button(csv_frame, text="Обзор", command=self.browse_data_dir).grid(row=0, column=2)

        tk.Label(csv_frame, text="Выходной CSV:").grid(row=1, column=0, sticky="w")
        tk.Entry(csv_frame, textvariable=self.output_csv, width=30).grid(row=1, column=1)

        tk.Button(csv_frame, text="Создать CSV", command=self.generate_csv).grid(row=2, column=1)

        # Фрейм для обучения
        training_frame = tk.LabelFrame(self.master, text="Обучение модели")
        training_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        tk.Button(training_frame, text="Обучить модель", command=self.train_model).grid(row=0, column=0)

        # Фрейм для предсказания
        prediction_frame = tk.LabelFrame(self.master, text="Предсказание")
        prediction_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        tk.Label(prediction_frame, text="Аудиофайл:").grid(row=0, column=0, sticky="w")
        tk.Entry(prediction_frame, textvariable=self.audio_path, width=30).grid(row=0, column=1)
        tk.Button(prediction_frame, text="Обзор", command=self.browse_audio_file).grid(row=0, column=2)

        tk.Button(prediction_frame, text="Предсказать", command=self.predict_audio).grid(row=1, column=1)

    def browse_data_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir.set(directory)

    def browse_audio_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Аудиофайлы", "*.wav")])
        if filepath:
            self.audio_path.set(filepath)

    def generate_csv(self):
        try:
            generate_csv(self.data_dir.get(), self.output_csv.get())
            messagebox.showinfo("Успех", "CSV файлы успешно созданы!")
            # Обновляем пути в config после генерации CSV
            self.config.train_csv = os.path.join(self.project_root, "data/data_train.csv")
            self.config.val_csv = os.path.join(self.project_root, "data/data_val.csv")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании CSV: {e}")

    def train_model(self):
        try:
            print(self.config)
            train(self.config)  # Передаем config в функцию train
            messagebox.showinfo("Успех", "Модель успешно обучена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении модели: {e}")

    def predict_audio(self):
        try:
            audio_path = self.audio_path.get()
            if not audio_path:
                 # Выбираем случайный файл из датасета
                real_files = [os.path.join(self.data_dir.get(), "REAL", f) for f in os.listdir(os.path.join(self.data_dir.get(), "REAL")) if f.endswith('.wav')]
                fake_files = [os.path.join(self.data_dir.get(), "FAKE", f) for f in os.listdir(os.path.join(self.data_dir.get(), "FAKE")) if f.endswith('.wav')]
                all_files = real_files + fake_files
                if all_files:
                    audio_path = random.choice(all_files)
                else:
                    raise ValueError("В датасете нет файлов для анализа.")
                self.audio_path.set(audio_path) # устанавливаем путь в GUI

            probability = predict(audio_path)

            if probability is not None:
                result = f"Вероятность дипфейка: {probability:.4f}\n"
                if probability > 0.5:
                    result += "Обнаружен дипфейк"
                else:
                    result += "Распознано как настоящее аудио"
                messagebox.showinfo("Результат предсказания", result)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при предсказании: {e}")

root = tk.Tk()
app = DeepFakeApp(root)
root.mainloop()