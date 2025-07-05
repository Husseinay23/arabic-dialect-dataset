import os
import random
import uuid
import csv
import subprocess
from pytube import YouTube
from pydub import AudioSegment
import webrtcvad
import whisper

# ---------------------------
# CONFIGURATION
# ---------------------------
CHUNK_DURATION_MS = 7000                   # Desired chunk length
SILENCE_THRESHOLD_DB = -40                 # Unused, but reserved for silence logic
VAD_MODE = 2                               # WebRTC VAD aggressiveness (0 = loose, 3 = strict)
TARGET_DBFS = -20.0                        # Normalize loudness
SAMPLE_RATE = 16000                        # Audio sample rate (Hz)
CHANNELS = 1                               # Mono channel
PADDING_MS = 500                           # Padding before/after detected speech (ms)
MAX_AUDIO_DURATION_MIN = 30                # Cap audio length to first 30 minutes
WHISPER_MODEL_SIZE = "base"                # Whisper model size ("base", "small", "medium")
OUTPUT_BASE = "./YourDataset"              # Base output directory

# ---------------------------
# LOAD WHISPER MODEL ONCE
# ---------------------------
print(f"🧠 Loading Whisper model: {WHISPER_MODEL_SIZE}...")
model = whisper.load_model(WHISPER_MODEL_SIZE)

# ---------------------------
# Download YouTube Audio
# ---------------------------
def download_youtube_audio(youtube_url, output_dir, filename="full_audio.mp3"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    print(f"🔽 Downloading audio from YouTube: {youtube_url}")
    command = [
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--output", output_path, youtube_url
    ]

    subprocess.run(command, check=True)
    print(f"🎧 Audio downloaded and saved to {output_path}")
    return output_path

# ---------------------------
# Prepare and Trim Audio
# ---------------------------
def prepare_audio(input_path):
    print("🎚️  Preparing audio (mono + 16kHz)...")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(CHANNELS).set_frame_rate(SAMPLE_RATE)

    max_duration_ms = MAX_AUDIO_DURATION_MIN * 60 * 1000
    if len(audio) > max_duration_ms:
        print("⏱️ Trimming audio to 30 minutes max")
        return audio[:max_duration_ms]
    return audio

# ---------------------------
# Normalize audio volume
# ---------------------------
def normalize_audio(audio_segment, target_dBFS=TARGET_DBFS):
    change = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change)

# ---------------------------
# Slice audio into raw frames for VAD
# ---------------------------
def make_frames(audio_segment, sample_rate, frame_duration_ms):
    frame_len = int(sample_rate * frame_duration_ms / 1000.0) * 2
    audio_bytes = audio_segment.raw_data
    frames = []
    for i in range(0, len(audio_bytes), frame_len):
        frame = audio_bytes[i:i + frame_len]
        if len(frame) == frame_len:
            timestamp = int(i / (sample_rate * 2) * 1000)
            frames.append((timestamp, frame))
    return frames

# ---------------------------
# Detect voiced segments with WebRTC VAD
# ---------------------------
def vad_collector(audio_segment, sample_rate=SAMPLE_RATE, chunk_ms=30, vad_mode=VAD_MODE):
    print("🗣️  Detecting voiced segments using WebRTC VAD...")
    vad = webrtcvad.Vad(vad_mode)
    frames = make_frames(audio_segment, sample_rate, chunk_ms)
    segments = []
    voiced = []

    for i, (timestamp, frame) in enumerate(frames):
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            voiced.append((timestamp, frame))
        elif voiced:
            start_ms = frames[i - len(voiced)][0]
            end_ms = timestamp

            # Apply padding around detected speech
            chunk_start = max(0, start_ms - PADDING_MS)
            chunk_end = min(len(audio_segment), end_ms + PADDING_MS)
            chunk = audio_segment[chunk_start:chunk_end]

            if len(chunk) >= CHUNK_DURATION_MS:
                segments.append((chunk[:CHUNK_DURATION_MS], chunk_start, chunk_end))
                print(f"🎙️  Segment extracted: {chunk_start}ms → {chunk_end}ms")

            voiced = []

    print(f"✅ VAD found {len(segments)} voiced segments")
    return segments

# ---------------------------
# Transcribe segments and write to metadata
# ---------------------------
def transcribe_and_save(segments, output_dir, dialect, source_url, quota=100):
    print(f"📝 Transcribing and saving segments for dialect: {dialect}")
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, f"{dialect}_metadata.csv")

    with open(metadata_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "filename", "dialect", "duration", "source_url",
            "start_time_ms", "end_time_ms", "language", "avg_logprob",
            "transcription", "whisper_model"
        ])

        count = 0
        random.shuffle(segments)

        for seg, start, end in segments:
            if count >= quota:
                break
            
            seg = seg.low_pass_filter(3400).high_pass_filter(300)  # ✅ Bandpass filtering
            seg = normalize_audio(seg)


            seg = normalize_audio(seg)
            temp_path = os.path.join(output_dir, "temp.wav")
            seg.export(temp_path, format="wav")

            result = model.transcribe(temp_path, language="ar", fp16=False)
            transcript = result["text"].strip()
            language = result.get("language", "")
            avg_logprob = result.get("avg_logprob", -10.0)

            if transcript and language == "ar" and avg_logprob > -1.0:
                filename = f"{dialect}_chunk_{uuid.uuid4().hex[:8]}.wav"
                final_path = os.path.join(output_dir, filename)
                seg.export(final_path, format="wav")

                sample_id = uuid.uuid4().hex[:12]
                writer.writerow([
                    sample_id, filename, dialect, round(seg.duration_seconds, 2),
                    source_url, start, end, language, round(avg_logprob, 3),
                    transcript, WHISPER_MODEL_SIZE
                ])
                print(f"✅ [{count+1}] Saved: {filename} | 🗣️ {transcript}")
                count += 1
            else:
                print("❌ Skipped low-quality or non-Arabic sample")

        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"📁 Metadata written to: {metadata_path}")
    print(f"🎉 Total usable segments saved: {count}")

# ---------------------------
# Main pipeline: Multiple videos per dialect
# ---------------------------
process_multiple_youtube_links(dialect, eval(f"{dialect.lower()}_links"), quota=10):

    print(f"\n🌍 Starting dataset build for: {dialect}")
    dialect_dir = os.path.join(OUTPUT_BASE, dialect)
    os.makedirs(dialect_dir, exist_ok=True)

    all_segments = []
    quota_per_link = int(quota * 1.5)  # Over-sample, filter later

    for i, url in enumerate(links):
        print(f"\n🔗 Processing video {i+1}/{len(links)}: {url}")
        try:
            audio_path = download_youtube_audio(url, dialect_dir, f"audio_{i+1}.mp3")
            audio = prepare_audio(audio_path)
            segments = vad_collector(audio)
            for seg in segments:
                all_segments.append((*seg, url))  # save URL too
        except Exception as e:
            print(f"⚠️ Error with link {url} — {e}")
            continue

    if not all_segments:
        print(f"❌ No valid segments found for {dialect}")
        return

    print(f"🧮 Total segments collected: {len(all_segments)}")
    selected_segments = all_segments[:quota * 2]

    # Drop URL (we already passed "multiple" as source_url for now)
    final_segments = [(seg, start, end) for seg, start, end, _ in selected_segments]

    transcribe_and_save(final_segments, dialect_dir, dialect, source_url="multiple", quota=quota)
    print(f"🏁 Finished dialect: {dialect}")
