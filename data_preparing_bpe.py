import pandas as pd
import os
import ast
from transformers import AutoTokenizer

# 1. 경로 설정
base_dir = "./LibriSpeech/manifest_bpe"
splits = ["train", "dev", "test"]
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base")

all_token_ids = set()
all_texts = []

for split in splits:
    print(f"\n[STEP 1] Tokenizing {split}.tsv ...")

    raw_path = os.path.join(base_dir, f"{split}.tsv")
    tokenized_path = os.path.join(base_dir, f"tokenized_bpe_{split}.tsv")
    merged_path = os.path.join(base_dir, f"tokenized_bpe_{split}_merged.tsv")

    df = pd.read_csv(raw_path, sep="\t")

    # BPE 토크나이즈
    df["input_ids"] = df["text"].apply(
        lambda x: tokenizer(x, add_special_tokens=False)["input_ids"]
    )

    # 저장
    df.to_csv(tokenized_path, sep="\t", index=False)
    print(f"  → BPE tokenized saved: {tokenized_path}")

    ### [STEP 2] MERGE TSV
    print(f"[STEP 2] Merging input_ids for: {split} ...")

    # 문자열 리스트를 공백으로 합치기
    df["input_ids"] = df["input_ids"].apply(
        lambda x: " ".join(map(str, x)) if isinstance(x, list) else ""
    )

    # 필요한 열만 유지
    df = df.loc[:, ["path", "duration", "text", "input_ids"]]
    df.to_csv(merged_path, sep="\t", index=False)
    print(f"  → Merged saved: {merged_path}")

    # 토큰 및 텍스트 수집
    for line in df["input_ids"]:
        tokens = line.strip().split()
        all_token_ids.update(map(int, tokens))
    all_texts.extend(df["text"].dropna().astype(str).tolist())

print(f"\n[INFO] All splits done. Unique BPE token IDs: {len(all_token_ids)}")

# 3. vocab.txt (문자 기반)
print("[STEP 3] Creating vocab.txt (char-based) ...")

vocab_dir = os.path.join(base_dir, "tokenizer_bpe")
os.makedirs(vocab_dir, exist_ok=True)
vocab_path = os.path.join(vocab_dir, "vocab.txt")

char_set = sorted(set(''.join(all_texts)))
special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<BLANK>"]
all_tokens = special_tokens + char_set

with open(vocab_path, "w", encoding="utf-8") as f:
    for token in all_tokens:
        f.write(token + "\n")

print(f"[INFO] vocab.txt saved to: {vocab_path}")
print(f"→ 총 vocab 크기 (문자 기반): {len(all_tokens)}")
