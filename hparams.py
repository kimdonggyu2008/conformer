# hparams.py

from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
from sys import argv
from hydra.core.global_hydra import GlobalHydra
import os

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Hydra config 로드 (명시적 config_path는 'config' 디렉토리 기준)
config_path = "conformer/config"
config_abs_path = os.path.abspath(config_path)  # ← 절대 경로로 변환
# 📍 3. Hydra 초기화
initialize_config_dir(config_dir=config_abs_path, version_base=None)

# 📍 4. 구성 불러오기
hparams: DictConfig = compose(config_name="small")  # 확장자 없이 이름만

def get_hparams() -> DictConfig:
    """
    전체 하이퍼파라미터 설정 반환 (raw config)
    """
    return hparams

def get_audpipe_params():
    return {
        'sampling_rate': hparams.data.sampling_rate,
        'n_mel_channels': hparams.data.n_mels,
        'win_length': hparams.data.win_length,
        'hop_length': hparams.data.hop_length,
        'n_time_masks': hparams.data.time_mask.number_of_masks,
        'ps': hparams.data.time_mask.ps,
        'max_freq_mask': hparams.data.freq_mask.max_freq
    }


def get_model_params(vocab_size: int):
    return {
        'enc_params': hparams.model.enc,
        'dec_params': dict(
            **hparams.model.dec,
            vocab_size=vocab_size
            )
    }


def get_optim_params():
    betas = [
        hparams.training.optim.beta1,
        hparams.training.optim.beta2,
        ]
    return dict(
        **hparams.training.optim,
        betas=betas
        )

def get_csv_paths() -> dict:
    return {
        'train_csv': hparams.data.training_file,
        'test_csv': hparams.data.testing_file,
        'csv_file_keys': {
            'duration': hparams.data.csv_file_keys.duration,
            'path': hparams.data.csv_file_keys.path,
            'text': hparams.data.csv_file_keys.text
        }
    }
