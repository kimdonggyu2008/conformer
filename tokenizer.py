from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import (
    Callable,
    List,
    Tuple,
    Union
    )
from os import PathLike
from data_loaders import JSONLoader
from utils import save_json
from functools import wraps

# bpe 토크나이저
import sentencepiece as spm

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
BLANK = '<BLANK>'


def check_token(token: str) -> Callable:
    """To check if a token exists or not

    Args:
        token ([type]): the token to be checked
    """
    def decorator(func):
        @wraps(func)
        def wrapper(obj, token=token):
            if token in obj._token_to_id:
                return obj._token_to_id[token]
            return func(obj, token)
        return wrapper
    return decorator


@dataclass
class SpecialTokens:
    _pad: Tuple[str, int] = (None, None)
    _blank: Tuple[str, int] = (None, None)
    _sos: Tuple[str, int] = (None, None)
    _eos: Tuple[str, int] = (None, None)

    @property
    def pad_id(self):
        return self._pad[1]

    @property
    def pad_token(self):
        return self._pad[0]

    @property
    def blank_id(self):
        return self._blank[1]

    @property
    def blank_token(self):
        return self._blank[0]

    @property
    def sos_id(self):
        return self._sos[1]

    @property
    def sos_token(self):
        return self._sos[0]

    @property
    def eos_id(self):
        return self._eos[1]

    @property
    def eos_token(self):
        return self._eos[0]

    @property
    def mask_id(self):
        return self._mask[1]

    @property
    def mask_token(self):
        return self._mask[0]


class ITokenizer(ABC):

    @abstractmethod
    def ids2tokens(self):
        pass

    @abstractmethod
    def tokens2ids(self):
        pass

    @abstractmethod
    def set_tokenizer(self):
        pass

    @abstractmethod
    def save_tokenizer(self):
        pass

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def add_token(self):
        pass

    @abstractmethod
    def preprocess_tokens(self):
        pass

    @abstractmethod
    def batch_tokenizer(self):
        pass

    @abstractproperty
    def vocab_size(self):
        pass

    @abstractmethod
    def get_tokens(self):
        pass


class BaseTokenizer(ITokenizer):
    _pad_key = 'pad'
    _sos_key = 'sos'
    _eos_key = 'eos'
    _blank_key = 'blank'
    _token_to_id_key = 'token_to_id'
    _special_tokens_key = 'special_tokens'

    def __init__(self) -> None:
        super().__init__()
        self._token_to_id = dict()
        self._id_to_token = dict()
        self.special_tokens = SpecialTokens()

    @property
    def vocab_size(self):
        return len(self._token_to_id)

    def add_token(self, token: str):
        token_id = self.vocab_size
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        return token_id

    @check_token(PAD)
    def add_pad_token(self, token=PAD) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._pad = (token, token_id)
        return self

    @check_token(BLANK)
    def add_blank_token(self, token=BLANK) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._blank = (token, token_id)
        return self

    @check_token(SOS)
    def add_sos_token(self, token=SOS) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._sos = (token, token_id)
        return self

    @check_token(EOS)
    def add_eos_token(self, token=EOS) -> ITokenizer:
        token_id = self.add_token(token)
        self.special_tokens._eos = (token, token_id)
        return self

    def _reset_id_to_token(self) -> None:
        self._id_to_token = dict(zip(
            self._token_to_id.values(),
            self._token_to_id.keys()
            ))

    def __set_special_tokens_dict(self, data: dict) -> None:
        if self._pad_key in data:
            self.special_tokens._pad = tuple(data[self._pad_key])
        if self._blank_key in data:
            self.special_tokens._blank = tuple(data[self._blank_key])
        if self._sos_key in data:
            self.special_tokens._sos = tuple(data[self._sos_key])
        if self._eos_key in data:
            self.special_tokens._eos = tuple(data[self._eos_key])

    def __get_special_tokens_dict(self) -> dict:
        data = {}
        if self.special_tokens.pad_id is not None:
            data[self._pad_key] = list(self.special_tokens._pad)
        if self.special_tokens.blank_id is not None:
            data[self._blank_key] = list(self.special_tokens._blank)
        if self.special_tokens.sos_id is not None:
            data[self._sos_key] = list(self.special_tokens._sos)
        if self.special_tokens.eos_id is not None:
            data[self._eos_key] = list(self.special_tokens._eos)
        return data

    def load_tokenizer(
            self,
            tokenizer_path: Union[str, PathLike],
            *args,
            **kwargs
            ) -> ITokenizer:
        data = JSONLoader(tokenizer_path).load()
        self._token_to_id = data[self._token_to_id_key]
        self.__set_special_tokens_dict(data[self._special_tokens_key])
        self._reset_id_to_token()
        return self

    def set_tokenizer(self, data: List[str], *args, **kwargs) -> ITokenizer:
        all_tokens = self.get_tokens(data)
        _ = list(map(self.add_token, all_tokens))
        self._reset_id_to_token()
        return self

    def save_tokenizer(
            self,
            save_path: Union[str, PathLike],
            *args,
            **kwargs
            ) -> None:
        data = {
            self._token_to_id_key: self._token_to_id,
            self._special_tokens_key: self.__get_special_tokens_dict()
        }
        save_json(save_path, data)

    def ids2tokens(self, ids: List[str]) -> List[str]:
        return list(map(lambda x: self._id_to_token[x], ids))

    def tokens2ids(self, sentence: str) -> List[int]:
        sentence = self.preprocess_tokens(sentence)
        return list(map(
            lambda x: self._token_to_id.get(x, self.special_tokens.pad_id),
            sentence)
            )

    def batch_tokenizer(self, data: List[str]) -> list:
        return list(map(self.tokens2ids, data))

    def batch_detokenizer(self, data: List[int]) -> list:
        return list(map(self.ids2tokens, data))

"""
class CharTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()

    def get_tokens(self, data: List[str]):
        return set(''.join(data))

    def preprocess_tokens(self, sentence: str) -> List[str]:
        return list(sentence)
"""

class BPETokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()




from transformers import AutoTokenizer

class AutoTokenizerWrapper:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id  # fallback
        self.blank_token_id = tokenizer.pad_token_id  # 대부분의 ASR에서는 BLANK == PAD

    def tokens2ids(self, sentence: str) -> List[int]:
        return self.tokenizer.encode(sentence, add_special_tokens=False)

    def batch_tokenizer(self, data: List[str]) -> List[List[int]]:
        return [self.tokens2ids(x) for x in data]

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def get_vocab(self):
        """
        내부 Hugging Face 토크나이저의 vocab 딕셔너리(토큰: ID 매핑)를 반환합니다.
        """
        # 대부분의 Hugging Face 토크나이저는 'vocab' 속성을 가집니다.
        # 이 딕셔너리는 '토큰 문자열'을 '정수 ID'로 매핑합니다.
        return self.tokenizer.vocab

    # ✅ ID를 토큰 문자열로 변환하는 메서드 (WER/CER 디코딩에 유용)
    def ids2tokens(self, ids: List[int]) -> List[str]:
        """
        토큰 ID 리스트를 토큰 문자열 리스트로 변환합니다.
        """
        # convert_ids_to_tokens는 단일 ID 또는 ID 리스트를 받을 수 있습니다.
        return self.tokenizer.convert_ids_to_tokens(ids)

    # ✅ ID 리스트를 전체 문장으로 디코딩하는 메서드 (WER/CER 디코딩에 유용)
    def decode(self, ids: List[int]) -> str:
        """
        토큰 ID 리스트를 전체 문자열로 디코딩합니다.
        특수 토큰을 건너뛰도록 설정하여 깔끔한 결과물을 얻습니다.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=True)


    @property
    def special_tokens(self):
        class Special:
            pad_id = self.pad_token_id
            blank_id = self.blank_token_id
        return Special()




class CharTokenizer:
    def __init__(self, vocab_path: str):
        self.idx_to_token = []
        self.token_to_idx = {}

        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.idx_to_token.append(token)
                self.token_to_idx[token] = idx

        self.special_tokens = SpecialTokens(
            _pad=("<PAD>", self.token_to_idx.get("<PAD>", 0)),
            _blank=("<BLANK>", self.token_to_idx.get("<BLANK>", 0)),
            _sos=("<SOS>", self.token_to_idx.get("<SOS>", 1)),
            _eos=("<EOS>", self.token_to_idx.get("<EOS>", 2))
        )


    def encode(self, text: str):
        return [self.token_to_idx[ch] for ch in text if ch in self.token_to_idx]

    def tokens2ids(self,text: str):
        return self.encode(text)

    def decode(self, ids):
        return "".join([self.idx_to_token[i] for i in ids if i < len(self.idx_to_token)])

    def ids2tokens(self,ids):
        return self.decode(ids)

    @property
    def vocab_size(self):
        return len(self.idx_to_token)

    def get_vocab(self):
        return self.token_to_idx
