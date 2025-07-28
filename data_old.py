import math
import pandas as pd
import torch
from tokenizer import ITokenizer
from utils import IPipeline
from pathlib import Path
from typing import List, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


# ✅ BaseData 그대로 사용
class BaseData:
    def __init__(self, text_pipeline, audio_pipeline, tokenizer, sampling_rate, hop_length, fields_sep, csv_file_keys):
        self.text_pipeline = text_pipeline
        self.audio_pipeline = audio_pipeline
        self.tokenizer = tokenizer
        self.max_len = 0
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.sep = fields_sep
        self.csv_file_keys = csv_file_keys

    def _get_padded_aud(self, aud_path: Union[str, Path], max_duration: float) -> Tensor:
        max_len = 1 + math.ceil(max_duration * self.sampling_rate / self.hop_length)
        self.max_len = max_len
        aud = self.audio_pipeline.run(aud_path)
        assert aud.shape[0] == 1, f"expected 1 channel, got {aud.shape[0]}"
        return self.pad_mels(aud, max_len)

    def _get_padded_tokens(self, text: str) -> Tensor:
        text = self.text_pipeline.run(text)
        tokens = self.tokenizer.tokens2ids(text)
        return torch.LongTensor(tokens), len(tokens)

    def pad_mels(self, mels: Tensor, max_len: int) -> Tensor:
        n = max_len - mels.shape[1]
        zeros = torch.zeros(size=(1, n, mels.shape[-1]))
        return torch.cat([zeros, mels], dim=1)

    def pad_tokens(self, tokens: list, max_len: int) -> Tensor:
        return tokens + [self.tokenizer.special_tokens.pad_id] * (max_len - len(tokens))


# ✅ Dataset 구현 (BaseData 상속)
class AudioTextDataset(Dataset, BaseData):
    def __init__(self, file_path, text_pipeline, audio_pipeline, tokenizer, sampling_rate, hop_length, fields_sep, csv_file_keys):
        BaseData.__init__(self, text_pipeline, audio_pipeline, tokenizer, sampling_rate, hop_length, fields_sep, csv_file_keys)
        self.df = pd.read_csv(file_path,sep="\t",encoding="utf-8")
        self.df[self.csv_file_keys.path] = self.df[self.csv_file_keys.path].apply(lambda p: p.replace("\\", "/"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row[self.csv_file_keys.path]
        text = row[self.csv_file_keys.text]
        duration = row[self.csv_file_keys.duration]

        mel = self._get_padded_aud(audio_path, duration)  # [1, T, F]
        tokens, token_len = self._get_padded_tokens(text)
        return mel.squeeze(0), tokens, token_len  # 텍스트는 필요시 추가 가능


def collate_fn(batch, pad_id=0):
    """
    Collate function for ASR CTC training.
    Args:
        batch: list of tuples (mel, tokens, token_length)
        pad_id: padding token id
    Returns:
        padded_mels: (B, T_mel, F)
        (padded_tokens, target_lengths)
    """
    # 1. Unpack
    mels, tokens, lengths = zip(*batch)  # tokens: list of Tensor

    # 2. Mel padding
    mel_lengths = [m.shape[0] for m in mels]
    max_mel_len = max(mel_lengths)
    feature_dim = mels[0].shape[1]
    padded_mels = torch.zeros(len(batch), max_mel_len, feature_dim)
    for i, mel in enumerate(mels):
        padded_mels[i, :mel.shape[0]] = mel

    # 3. Token padding
    max_token_len = max(lengths)
    padded_tokens = torch.full((len(batch), max_token_len), fill_value=pad_id, dtype=torch.long)
    for i, t in enumerate(tokens):
        if torch.is_tensor(t):
            t_list = t.tolist()
        else:
            t_list = t
        padded_tokens[i, :len(t_list)] = torch.tensor(t_list, dtype=torch.long)

    # 4. target_lengths tensor
    target_lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_mels, (padded_tokens, target_lengths)



# ✅ 외부 DataLoader 클래스 (기존 이름 유지)
class DataLoader:
    def __init__(self, file_path, text_pipeline, audio_pipeline, tokenizer,
                 batch_size, sampling_rate, hop_length, fields_sep, csv_file_keys,
                 num_workers=4):
        self.loader = TorchDataLoader(
            AudioTextDataset(file_path, text_pipeline, audio_pipeline, tokenizer,
                             sampling_rate, hop_length, fields_sep, csv_file_keys),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  # ✅ 멀티프로세싱 적용
            pin_memory=True,
            collate_fn=collate_fn
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

