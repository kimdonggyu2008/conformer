import os
import csv
import soundfile as sf
import pandas as pd
import json

# ✅ 1. 경로 설정
BASE_DIR = "/home/kimdonggyu/asr_test/conformer-main/LibriSpeech"
SAVE_DIR = os.path.join(BASE_DIR, "manifest_char")
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 데이터셋 매핑
DATASETS = {
    "train.tsv": os.path.join(BASE_DIR, "train-clean-100"),
    "dev.tsv": os.path.join(BASE_DIR, "dev-clean"),
    "test.tsv": os.path.join(BASE_DIR, "test-clean")
}

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<BLANK>"]

# ✅ 2. 오디오 길이 계산 함수
def get_duration(file_path):
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate

# ✅ 3. Manifest 생성
def generate_manifest(tsv_name, dataset_dir):
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
                    entries.append([flac_path, duration, text.lower()])  # ✅ 소문자로 통일

    save_path = os.path.join(SAVE_DIR, tsv_name)
    with open(save_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["path", "duration", "text"])
        writer.writerows(entries)
    print(f"[INFO] {tsv_name} saved at {save_path} ({len(entries)} samples)")

# ✅ 모든 split 처리
for tsv_name, dataset_dir in DATASETS.items():
    generate_manifest(tsv_name, dataset_dir)

# ✅ 4. Vocab 생성 (train 기반)
train_path = os.path.join(SAVE_DIR, "train.tsv")
train_df = pd.read_csv(train_path, sep="\t")
all_texts = train_df["text"].dropna().astype(str).tolist()

char_set = sorted(set("".join(all_texts)))
vocab_tokens = SPECIAL_TOKENS + char_set

vocab_dir = os.path.join(SAVE_DIR, "tokenizer_char")
os.makedirs(vocab_dir, exist_ok=True)
vocab_path = os.path.join(vocab_dir, "vocab.txt")

with open(vocab_path, "w", encoding="utf-8") as f:
    for token in vocab_tokens:
        f.write(token + "\n")

print(f"[INFO] vocab.txt saved at {vocab_path} (size: {len(vocab_tokens)})")

# ✅ 5. Tokenizing (input_ids 추가)
def tokenize_and_save(tsv_file, vocab_dict):
    df = pd.read_csv(tsv_file, sep="\t")
    def encode(text):
        return [vocab_dict[c] for c in text if c in vocab_dict]
    df["input_ids"] = df["text"].apply(lambda x: json.dumps(encode(str(x))))
    tokenized_path = tsv_file.replace(".tsv", "_char.tsv")
    df.to_csv(tokenized_path, sep="\t", index=False)
    print(f"[INFO] Tokenized file saved at {tokenized_path}")

# ✅ Vocab dict
vocab_dict = {token: idx for idx, token in enumerate(vocab_tokens)}

for split in ["train.tsv", "dev.tsv", "test.tsv"]:
    tokenize_and_save(os.path.join(SAVE_DIR, split), vocab_dict)

