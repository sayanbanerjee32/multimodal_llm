import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

class AudioTranscriptionPipeline:
    def __init__(self, model_name="openai/whisper-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def load_audio(self, file_path, sample_rate=16000):
        audio, _ = librosa.load(file_path, sr=sample_rate)
        return audio

    def transcribe(self, audio):
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]

    def __call__(self, audio_file):
        audio = self.load_audio(audio_file)
        return self.transcribe(audio)

# Usage example:
# audio_pipeline = AudioTranscriptionPipeline()
# transcription = audio_pipeline("path/to/your/audio/file.wav")
# print("Transcription:", transcription)
