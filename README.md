# arabic-dialect-dataset
# 🗣️ Arabic Dialect Voice Dataset Builder 🎙️

This repository contains a complete pipeline to **collect**, **clean**, **transcribe**, **split**, and **convert** Arabic dialect audio samples into training-ready data for machine learning models.

The final goal is to build a diverse and high-quality dataset for Arabic **dialect classification** or **speech-related** deep learning tasks.

---

## 📦 Features

- ✅ YouTube audio downloader for each dialect
- ✅ Voice Activity Detection (VAD) using WebRTC
- ✅ Whisper-based Arabic transcription
- ✅ Smart silence removal + padding
- ✅ Dataset splitting: train / val / test
- ✅ Mel Spectrogram generation (for CNN input)
- ✅ Metadata file including all transcript & path info

---

## 🧱 Dataset Structure

After running the full pipeline, the folder structure for each dialect looks like this:

YourDataset/
            Lebanese/
                     train/
                           wav/ # Cleaned .wav clips
                           mel/ # Mel spectrogram images (.png)
                     val/
                           wav/
                           mel/
                     test/
                           wav/
                           mel/
                     Lebanese_metadata.csv 

---

 
Each metadata CSV contains columns:
- `sample_id`: unique ID
- `filename`: name of the audio file
- `dialect`: e.g., Lebanese, Egyptian
- `duration`: in seconds
- `source_url`: YouTube source
- `start_time_ms`, `end_time_ms`: original clip location
- `language`: Whisper-detected language
- `avg_logprob`: average transcription confidence
- `transcription`: Whisper output
- `whisper_model`: model used (e.g., `base`)
- `split`: `train`, `val`, or `test`
- `mel_path`: spectrogram image path

---

## 🛠️ Requirements

Install the following before running the notebook:

```bash
pip install torch torchvision torchaudio
pip install openai-whisper
pip install pytube
pip install pydub
pip install webrtcvad
pip install librosa
pip install matplotlib
pip install pandas scikit-learn
brew install ffmpeg     # On macOS (for audio processing)

---

 💬 Supported Dialects
You can customize the dataset to collect samples from the following dialects:

🇱🇧 Lebanese

🇪🇬 Egyptian

🇸🇾 Syrian

🇵🇸 Palestinian

🇯🇴 Jordanian

🇸🇦 Saudi

🇦🇪 Emirati

🇮🇶 Iraqi


