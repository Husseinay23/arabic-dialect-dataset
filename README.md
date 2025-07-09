 # 🗣️ Arabic Dialect Speech Dataset Pipeline

This repository contains a complete pipeline to create a clean, labeled, and ready-to-train dataset of Arabic dialect speech audio samples extracted from YouTube. It supports multiple dialects and generates Mel spectrograms to be used in deep learning models.

---

## 🌍 Supported Dialects

The pipeline is designed to support any number of dialects. It currently includes:

- 🇱🇧 Lebanese
- 🇪🇬 Egyptian
- 🇯🇴 Jordanian
- 🇸🇾 Syrian
- 🇮🇶 Iraqi
- 🇵🇸 Palestinian
- 🇸🇦 Saudi
- 🇦🇪 Emirati

Each dialect has its own folder and is processed independently.

---

## 🛠️ Pipeline Overview

### 1. 🎧 YouTube Audio Downloader
Downloads full audio from YouTube videos using `yt-dlp`. Files are saved in a `full_audio/` folder per dialect.

```python
download_youtube_audio("https://youtube.com/...", dialect_dir="Dataset/Lebanese")
````

* Format: `.mp3`
* Trims the first 60 seconds to skip intros

---

### 2. 🧠 Voice Activity Detection (VAD)

Applies WebRTC VAD to extract clean voice chunks (default 7 seconds each). Adds padding before and after speech.

```python
segments = vad_collector(audio)
```

* Removes silence and background noise
* Segments are saved as `.wav` files
* Each dialect has its own CSV metadata file

---

### 3. 📄 Metadata Creation

Each saved segment is labeled with:

* `sample_id`
* `filename`
* `dialect`
* `start_time_ms`, `end_time_ms`
* `start_time_str`, `end_time_str`
* `source_url`

**Example metadata row:**

```csv
sample_id,filename,dialect,duration_sec,start_time_ms,end_time_ms,start_time_str,end_time_str,source_url
abc12345,lebanese_chunk_0001.wav,Lebanese,6.95,61000,68000,01:01,01:08,https://youtube.com/...
```

---

### 4. 📦 Dataset Split

Automatically splits each dialect dataset into:

* `train` (70%)
* `val` (15%)
* `test` (15%)

Files are **moved** to:

```
Dataset/
└── Lebanese/
    ├── train/wav/
    ├── val/wav/
    └── test/wav/
```

The metadata CSV is updated with a `split` column.

---

### 5. 🎼 Mel Spectrogram Generation

Generates Mel spectrograms (`.png`) from `.wav` files using Librosa and Matplotlib.

```python
mel = librosa.feature.melspectrogram(y, sr=16000, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)
```

* Output saved to:

  ```
  Dataset/Lebanese/train/mel/lebanese_chunk_0001.png
  ```
* Metadata CSV is updated with `mel_path`



---

## 📁 Final Folder Structure

```
Dataset/
├── Lebanese/
│   ├── train/
│   │   ├── wav/
│   │   └── mel/
│   ├── val/
│   │   ├── wav/
│   │   └── mel/
│   ├── test/
│   │   ├── wav/
│   │   └── mel/
│   └── Lebanese_metadata.csv
├── Egyptian/
│   └── ...
```

---

## ▶️ How to Run

Run the full pipeline in a Jupyter Notebook using:

1. `process_multiple_youtube_links(dialect, links)`
2. `split_all_dialects(base_dir="./Dataset")`
3. `generate_mel_spectrograms(dialect_dir, dialect)`

Or loop through all dialects:

```python
dialects = ['Lebanese', 'Egyptian', 'Jordanian', 'Syrian', 'Iraqi', 'Palestinian', 'Saudi', 'Emirati']
for dialect in dialects:
    generate_mel_spectrograms(os.path.join('./Dataset', dialect), dialect)
```

---

## 📦 Dependencies

Install with pip:

```bash
pip install -r requirements.txt
```

**Main libraries:**

* `yt-dlp` (for YouTube downloads)
* `pydub` (audio processing)
* `webrtcvad` (voice activity detection)
* `librosa`, `matplotlib`, `numpy`, `pandas` (for spectrograms and metadata)
* `scikit-learn` (for splitting)

---

## 🚀 Future Improvements

* [ ] Save spectrograms as `.npy` arrays instead of `.png`
* [ ] Add automatic dialect detection or label verification
* [ ] Include command-line interface (CLI) for automation
* [ ] Use Whisper or DeepSpeech to align transcription (optional)

---

## 🙌 Acknowledgments

This project was developed by Hussein Ayoub for academic and research purposes in Arabic speech processing and dialect AI modeling. Data is sourced from publicly available YouTube videos 
