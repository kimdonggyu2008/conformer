import os
import torch
from tqdm import tqdm
from torchmetrics.text import WordErrorRate, CharErrorRate
from omegaconf import OmegaConf
import argparse
import pandas as pd
import gc

from model import Model
from tokenizer import AutoTokenizerWrapper
from transformers import AutoTokenizer
from pipelines import get_pipelines
from data import DataLoader
from hparams import get_model_params, get_audpipe_params

MODEL_KEY = 'model'


def clear_memory():
    """GPU 및 CPU 메모리 캐시 정리"""
    torch.cuda.empty_cache()
    gc.collect()


# ✅ 모델 로드
def load_model(model_params: dict, checkpoint_path: str) -> torch.nn.Module:
    model = Model(**model_params)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state[MODEL_KEY])
    return model


# ✅ 토크나이저 로드
def get_tokenizer(pretrained_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)
    return AutoTokenizerWrapper(tokenizer)


# ✅ Inference 실행
def run_inference(cfg):
    device = f"{cfg.device}:{cfg.gpu_id}" if cfg.device == "cuda" else cfg.device

    # Tokenizer
    tokenizer = get_tokenizer(cfg.tokenizer.pretrained_name_or_path)
    blank_id = tokenizer.special_tokens.blank_id
    vocab_size = tokenizer.vocab_size

    # Model
    text_pipeline, audio_pipeline = get_pipelines(get_audpipe_params())
    model = load_model(get_model_params(vocab_size), checkpoint_path=cfg.checkpoint)
    model.to(device)
    model.eval()

    # DataLoader
    test_loader = DataLoader(
        cfg.data.testing_file,
        text_pipeline,
        audio_pipeline,
        tokenizer,
        cfg.inference.batch_size,
        cfg.data.sampling_rate,
        cfg.data.hop_length,
        cfg.data.files_sep,
        cfg.data.csv_file_keys,
        num_workers=0
    )

    # Metrics
    wer_metric = WordErrorRate()
    cer_metric = CharErrorRate()
    results = []
    total_loss = 0

    criterion = torch.nn.CTCLoss(blank_id)

    print("✅ Start Inference...")
    with torch.no_grad():
        for x, (y, target_lengths, texts,_) in tqdm(test_loader):
            #print(f"batch x shape: {x.shape}, y shape: {y.shape},target lengths: {target_lengths}")
            x = x.to(device)
            y = y.to(device)
            target_lengths = target_lengths.to(device)

            # Forward
            result = model(x).log_softmax(dim=-1).permute(1, 0, 2)

            # Loss 계산 (Optional)
            input_lengths = torch.full((y.size(0),), result.size(0), dtype=torch.long).to(device)
            flattened_targets = torch.cat([y[i, :target_lengths[i]] for i in range(y.size(0))])
            loss = criterion(result, flattened_targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # Decode predictions
            pred_ids_batch = result.argmax(dim=-1).transpose(0, 1).cpu().numpy()
            print(pred_ids_batch[0])
            for i in range(pred_ids_batch.shape[0]):
                raw_pred_ids = pred_ids_batch[i]

                # ✅ CTC post-processing: blank & 중복 제거
                filtered_ids = []
                prev_id = None
                for id_val in raw_pred_ids:
                    if id_val != blank_id and id_val != prev_id:
                        filtered_ids.append(id_val)
                    prev_id = id_val

                pred_text = tokenizer.decode(filtered_ids)
                ref_text = texts[i] if len(texts) > i else ""

                # ✅ Metrics 업데이트
                wer_metric.update(pred_text, ref_text)
                cer_metric.update(pred_text, ref_text)

                # ✅ 결과 저장 (텍스트 + ID)
                ref_ids = y[i, :target_lengths[i]].cpu().tolist()

                print(f"Pred : {pred_text}")
                print(f"Ref : {ref_text}")
                results.append({
                    "ref_text": ref_text,
                    "pred_text": pred_text,
                    "ref_ids": ref_ids,
                    "pred_ids": filtered_ids
                })

            del result, loss, x, y
            clear_memory()

    # ✅ Metrics 결과
    avg_loss = total_loss / len(test_loader)
    avg_wer = wer_metric.compute().item()
    avg_cer = cer_metric.compute().item()

    print("\n✅ Inference Completed")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"WER: {avg_wer:.4f}, CER: {avg_cer:.4f}")

    # ✅ CSV 저장
    if cfg.inference.save_csv:
        os.makedirs(cfg.inference.output_dir, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(cfg.inference.output_dir, "test_predictions.csv"), index=False)
        print(f"✅ Results saved to {cfg.inference.output_dir}/test_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="infer.yaml", help="Path to infer.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    run_inference(cfg)

