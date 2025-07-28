import os
import csv
import soundfile as sf
from typing import Dict

# 저장 위치 지정
SAVE_DIR = "./LibriSpeech/manifest_bpe"
os.makedirs(SAVE_DIR, exist_ok=True)

# 데이터셋 경로 설정
DATASETS: Dict[str, str] = {
    "train.tsv": "./LibriSpeech/train-clean-100",
    "dev.tsv": "./LibriSpeech/dev-clean",
    "test.tsv": "./LibriSpeech/test-clean"
}

def get_duration(file_path):
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate

def generate_manifest(tsv_name: str, dataset_dir: str):
    entries = []

    for root, _, files in os.walk(dataset_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        for txt_file in txt_files:
            txt_path = os.path.join(root, txt_file)
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", maxsplit=1)
                    if len(parts) != 2:
                        continue
                    file_id, text = parts
                    flac_path = os.path.join(root, file_id + ".flac")
                    if not os.path.exists(flac_path):
                        continue
                    duration = get_duration(flac_path)
                    entries.append([flac_path, duration, text])

    # 저장 위치로 파일 작성
    save_path = os.path.join(SAVE_DIR, tsv_name)
    with open(save_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "duration", "text"])
        writer.writerows(entries)
    print(f"{tsv_name} saved at {save_path} with {len(entries)} entries.")

# 전체 데이터셋 처리
for tsv_name, dataset_dir in DATASETS.items():
    generate_manifest(tsv_name, dataset_dir)
