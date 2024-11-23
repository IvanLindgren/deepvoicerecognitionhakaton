import librosa
import torch
from models.transformer import TransformerClassifier
from data.utils import to_spectrogram_tensor, truncate_or_pad
from training.config import TrainingConfig

def process_audio(audio_path, target_sr=16000, min_duration_sec=1):
    """
    Загружает и обрабатывает аудиофайл.

    Args:
        audio_path (str): Путь к аудиофайлу.
        target_sr (int): Частота дискретизации.
        min_duration_sec (int): Минимальная длительность аудио в секундах.

    Returns:
        np.ndarray: Обработанный аудиофайл.
        None: Если файл не удалось загрузить или он слишком короткий.
    """
    try:
        audio, _ = librosa.load(audio_path, sr=target_sr)
        if len(audio) < min_duration_sec * target_sr:  # Проверяем минимальную длину
            raise ValueError(f"Audio too short: {len(audio)} samples, expected at least {min_duration_sec * target_sr}")
        return audio
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None


def predict(audio_path):
    config = TrainingConfig()
    device = config.device

    # Загрузка модели
    model = TransformerClassifier(
        input_dim=config.input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.eval()

    # Предобработка аудио
    audio = process_audio(audio_path, target_sr=16000)
    if audio is None:
        print(f"Failed to process audio: {audio_path}")
        return None

    # Преобразование в спектрограмму
    spectrogram = to_spectrogram_tensor(audio).to(device)

    # Приведение к фиксированной длине (например, 5000, как на этапе обучения)
    max_audio_length = 5000  # Обязательно проверьте, какое значение использовалось при обучении
    spectrogram = truncate_or_pad(spectrogram, max_audio_length).unsqueeze(0)  # Добавляем размерность батча

    # Инференс
    with torch.no_grad():
        output = model(spectrogram)
        probability = output.item()

    return probability
