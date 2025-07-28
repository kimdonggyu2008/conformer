import math
import pandas as pd
import torch
from tokenizer import ITokenizer
from utils import IPipeline
from pathlib import Path
from typing import List, Union
from torch import Tensor
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence # ✅ pad_sequence 추가

# ✅ BaseData (함수명 유지, 내용 변경)
class BaseData:
    def __init__(self, text_pipeline, audio_pipeline, tokenizer, sampling_rate, hop_length, fields_sep, csv_file_keys):
        self.text_pipeline = text_pipeline
        self.audio_pipeline = audio_pipeline
        self.tokenizer = tokenizer
        # self.max_len = 0 # ✅ 이 변수는 개별 샘플의 패딩 길이를 저장했는데, 배치 단위 패딩으로 바뀌면서 필요 없어졌습니다.
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.sep = fields_sep
        self.csv_file_keys = csv_file_keys

    # ✅ _get_padded_aud 함수 (함수명 유지, 내용 변경)
    # 이 함수는 더 이상 패딩을 수행하지 않습니다. 'padded'라는 이름은 오해를 줄 수 있지만,
    # 함수명을 바꾸지 말라는 요청에 따라 그대로 유지합니다.
    # 대신, 원본 멜 스펙트로그램 특징만 반환합니다.
    def _get_padded_aud(self, aud_path: Union[str, Path], max_duration: float) -> Tensor: # max_duration은 더 이상 내부에서 사용되지 않음
        aud = self.audio_pipeline.run(aud_path) # [1, T, F] 또는 [T, F] 형태의 멜 스펙트로그램 반환 예상
        # 오디오 특징이 [1, T, F] 형태라면 [T, F]로 차원을 축소합니다.
        if aud.ndim == 3 and aud.shape[0] == 1:
            aud = aud.squeeze(0) # [1, T, F] -> [T, F]
        assert aud.ndim == 2, f"예상치 못한 오디오 특징 차원입니다. 2D (시간, 특징 차원)를 예상했지만 {aud.ndim}D를 받았습니다."
        return aud # ✅ 패딩되지 않은 멜 스펙트로그램 텐서 반환

    # ✅ _get_padded_tokens 함수 (함수명 유지, 내용 변경)
    # 이 함수도 더 이상 패딩을 수행하지 않습니다. 'padded'라는 이름은 오해를 줄 수 있지만,
    # 함수명을 바꾸지 말라는 요청에 따라 그대로 유지합니다.
    # 대신, 토큰 ID 리스트와 그 길이를 반환합니다.
    def _get_padded_tokens(self, text: str) -> (List[int], int): # 반환 타입 주석 변경
        text = self.text_pipeline.run(text) # 텍스트 전처리
        tokens = self.tokenizer.tokens2ids(text) # 텍스트를 토큰 ID 리스트(Python list[int])로 변환
        return tokens, len(tokens) # ✅ 토큰 ID 리스트(list[int])와 길이 반환

    # ✅ pad_mels 함수 (함수명 유지, 더 이상 직접 사용되지 않음)
    # 이 함수는 이제 AudioTextDataset.__getitem__이나 collate_fn에서 직접 호출되지 않습니다.
    # collate_fn의 pad_sequence가 이 역할을 대신합니다.
    def pad_mels(self, mels: Tensor, max_len: int) -> Tensor:
        n = max_len - mels.shape[1]
        zeros = torch.zeros(size=(1, n, mels.shape[-1]))
        return torch.cat([zeros, mels], dim=1)

    # ✅ pad_tokens 함수 (함수명 유지, 더 이상 직접 사용되지 않음)
    # 이 함수도 이제 AudioTextDataset.__getitem__이나 collate_fn에서 직접 호출되지 않습니다.
    # collate_fn의 pad_sequence가 이 역할을 대신합니다.
    def pad_tokens(self, tokens: list, max_len: int) -> Tensor:
        return tokens + [self.tokenizer.special_tokens.pad_id] * (max_len - len(tokens))


# ✅ Dataset 구현 (AudioTextDataset) (함수명 유지, 내용 변경)
class AudioTextDataset(Dataset, BaseData):
    def __init__(self, file_path, text_pipeline, audio_pipeline, tokenizer, sampling_rate, hop_length, fields_sep, csv_file_keys, max_audio_len=None):
        BaseData.__init__(self, text_pipeline, audio_pipeline, tokenizer, sampling_rate, hop_length, fields_sep, csv_file_keys)
        self.df = pd.read_csv(file_path,sep="\t",encoding="utf-8")
        self.df[self.csv_file_keys.path] = self.df[self.csv_file_keys.path].apply(lambda p: p.replace("\\", "/"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row[self.csv_file_keys.path]
        text = row[self.csv_file_keys.text]
        # duration = row[self.csv_file_keys.duration] # ✅ duration은 이제 개별 샘플 패딩에 사용되지 않으므로 필요 없습니다.

        # ✅ 수정: _get_padded_aud는 이제 패딩되지 않은 멜 스펙트로그램을 반환합니다.
        mel = self._get_padded_aud(audio_path, row[self.csv_file_keys.duration]) # 함수명은 유지하되, 내부적으로 duration 사용 안함
        
        # ✅ 수정: _get_padded_tokens는 이제 토큰 ID 리스트(list[int])를 반환합니다.
        tokens_list, token_len = self._get_padded_tokens(text) # tokens_list: list[int]

        # collate_fn의 `pad_sequence`는 텐서 리스트를 기대하므로, 여기서 토큰 리스트를 텐서로 변환합니다.
        tokens_tensor = torch.LongTensor(tokens_list)

        # ✅ 최종 반환값 형태: (멜 텐서, 토큰 텐서, 토큰 길이 정수, 원본 텍스트 문자열)
        # 이 형태는 아래 `collate_fn`이 배치로 묶을 때 사용됩니다.
        return mel, tokens_tensor, token_len, text, audio_path


# ✅ collate_fn 함수 (함수명 유지, 내용 변경)
# 이 함수는 `torch.nn.utils.rnn.pad_sequence`를 사용하여 메모리 효율적으로 변경되었습니다.

def collate_fn(batch, pad_id=0):
    """
    ASR CTC 훈련을 위한 collate 함수입니다.
    Args:
        batch: __getitem__에서 반환된 튜플들의 리스트 (mel_tensor, token_tensor, token_length_int, original_text_str$        pad_id: 패딩 토큰 ID
    Returns:
        padded_mels: (B, T_mel, F) 형태의 패딩된 멜 스펙트로그램 텐서
        (padded_tokens, target_lengths, texts_list): 다음을 포함하는 튜플
            padded_tokens: (B, T_token) 형태의 패딩된 토큰 텐서
            target_lengths: (B,) 형태의 실제 토큰 길이 텐서
            texts_list: 원본 텍스트 문자열들의 리스트 (WER/CER 계산용)
    """
    # 1. 배치 언팩: __getitem__에서 반환된 4개의 요소를 받습니다.
    mels_list, tokens_list, lengths_list, texts_list, audio_paths = zip(*batch)

    # 2. 멜 스펙트로그램 패딩: pad_sequence를 사용하여 효율적으로 패딩합니다.
    # mels_list는 __getitem__에서 반환된 [T, F] 형태의 멜 텐서들의 리스트입니다.
    # batch_first=True는 결과 텐서의 첫 번째 차원이 배치 크기가 되도록 합니다 (B, T_mel, F).
    # padding_value=0.0은 멜 스펙트로그램의 패딩 값을 0으로 설정합니다.
    padded_mels = pad_sequence(mels_list, batch_first=True, padding_value=0.0)

    # 3. 토큰 패딩: pad_sequence를 사용하여 효율적으로 패딩합니다.
    # tokens_list는 __getitem__에서 반환된 LongTensor 형태의 토큰 텐서들의 리스트입니다.
    # padding_value는 함수 인자로 받은 pad_id를 사용합니다.
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=pad_id)

    # 4. 실제 토큰 길이 텐서 생성: 이미 __getitem__에서 길이를 가져왔으므로 그대로 텐서로 만듭니다.
    target_lengths = torch.tensor(lengths_list, dtype=torch.long)

    # 5. 최종 반환값: Trainer가 기대하는 형태 (x, (y, target_lengths, texts))에 맞춰 반환합니다.
    return padded_mels, (padded_tokens, target_lengths, texts_list, list(audio_paths))



# ✅ 외부 DataLoader 클래스 (함수명 유지, 내용 변경 없음)
# 이 클래스 자체는 변경되지 않았지만, 이제 내부적으로 변경된 AudioTextDataset과 collate_fn을 사용합니다.
class DataLoader:
    def __init__(self, file_path, text_pipeline, audio_pipeline, tokenizer,
                 batch_size, sampling_rate, hop_length, fields_sep, csv_file_keys,
                 num_workers=4):
        self.loader = TorchDataLoader(
            AudioTextDataset(file_path, text_pipeline, audio_pipeline, tokenizer,
                             sampling_rate, hop_length, fields_sep, csv_file_keys),
            batch_size=batch_size,
            shuffle=True, # 학습 시에만 True, 검증/테스트 시에는 일반적으로 False가 좋습니다.
            num_workers=num_workers, # 멀티프로세싱으로 데이터 로딩 속도 향상
            pin_memory=True, # CPU에서 GPU로 데이터 전송을 효율적으로 만듭니다.
            collate_fn=collate_fn # ✅ 최적화된 collate_fn을 사용하도록 지정
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
