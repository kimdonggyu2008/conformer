import gc
import os
from pathlib import Path
import torch
from hparams import (
    get_audpipe_params,
    get_model_params,
    get_optim_params,
    hparams
    )
from functools import wraps
from torch.nn import Module
from data import DataLoader
from torch.optim import Optimizer
from typing import Callable, Union
from tqdm import tqdm
from torchinfo import summary
from model import Model

from optim import AdamWarmup
from pipelines import get_pipelines
from tokenizer import CharTokenizer, ITokenizer
from utils import IPipeline
from transformers import AutoTokenizer
#from tokenizer import AutoTokenizerWrapper
from tokenizer import CharTokenizer
from omegaconf import OmegaConf
# wandb 추가
import wandb
import psutil
from torchmetrics.text import WordErrorRate, CharErrorRate
from data import DataLoader
from statistics import mean

wandb.login(key="")

wandb.init(
    project="conformer",  # 원하는 프로젝트명
    name="conformer_small_scale01_8000",  # 예: run_conformer_01
    config=OmegaConf.to_container(hparams,resolve=True)  # 모든 하이퍼파라미터 로깅
)



MODEL_KEY = 'model'
OPTIMIZER_KEY = 'optimizer'


def log_memory(tag=""):
    process=psutil.Process(os.getpid())
    cpu_mem=process.memory_info().rss/1024**2
    gpu_mem_alloc=torch.cuda.memory_allocated()/1024**2
    gpu_mem_reserved=torch.cuda.memory_reserved()/1024**2
    print(f"[{tag}] CPU:{cpu_mem:.2f}MB | GPU Alloc: {gpu_mem_alloc:.2f} MB |  GPU reserved:{gpu_mem_reserved:.2f}MB")

def check_tensor_growth():
    tensors=[obj for obj in gc.get_objects() if torch.is_tensor(obj)]
    total_elements=sum(t.numel() for t in tensors)
    print(f"Tensor count: {len(tensors)} | Total Elements:{total_elements}")
    return len(tensors)

def print_gpu_status(step):
    allocated=torch.cuda.memory_allocated()/(1024**2)
    reserved=torch.cuda.memory_reserved()/(1024**2)
    print(f"[Step {step}] GPU Memory Allocated: {allocated:.2f} MB | Reserved:{reserved:.2f} MB")

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(func) -> Callable:
    """Save a checkpoint after each iteration
    """
    @wraps(func)
    def wrapper(obj, *args, _counter=[0], **kwargs):
        _counter[0] += 1
        result = func(obj, *args, **kwargs)
        if not os.path.exists(hparams.training.checkpoints_dir):
            os.mkdir(hparams.training.checkpoints_dir)
        checkpoint_path = os.path.join(
            hparams.training.checkpoints_dir,
            'checkpoint_' + str(_counter[0]) + '.pt'
            )
        torch.save(
            {
                MODEL_KEY: obj.model.state_dict(),
                OPTIMIZER_KEY: obj.optimizer.state_dict(),
            },
            checkpoint_path
            )

        print(f'checkpoint saved to {checkpoint_path}')
        return result
    return wrapper


class Trainer:
    __train_loss_key = 'train_loss'
    __test_loss_key = 'test_loss'

    def __init__(
            self,
            criterion: Module,
            optimizer: Optimizer,
            model: Module,
            device: str,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            tokenizer: CharTokenizer,
            blank_id: int
            ) -> None:
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.step_history = dict()
        self.history = dict()

        self.tokenizer=tokenizer
        self.blank_id=blank_id
        self.id_to_char={v:k for k, v in self.tokenizer.get_vocab().items()}
        self.wer_metric=WordErrorRate()
        self.cer_metric=CharErrorRate()


    def fit(self):
        """The main training loop that train the model on the training
        data then test it on the test set and then log the results
        """
        for epoch in range(self.epochs):
            self.train(epoch)
            self.test(epoch)
            self.print_results()

    def set_train_mode(self) -> None:
        """Set the models on the training mood
        """
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        """Set the models on the testing mood
        """
        self.model = self.model.eval()

    def print_results(self):
        """Prints the results after each epoch
        """
        result = ''
        for key, value in self.history.items():
            result += f'{key}: {str(value[-1])}, '
        print(result[:-2])

    def test(self,epoch:int):
        """Iterate over the whole test data and test the models
        for a single epoch
        """
        total_loss = 0
        self.set_test_mode()
        self.wer_metric.reset()
        self.cer_metric.reset()
        with torch.no_grad():
            for x, (y, target_lengths,texts,_) in tqdm(self.test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                target_lengths=target_lengths.to(self.device)
                result = self.model(x)
                result = result.log_softmax(axis=-1)
                
                result=result.permute(1,0,2)
                #y=y[..., :result.shape[0]]
                input_lengths = torch.full(
                    size=(y.shape[0],),
                    fill_value=result.shape[0],
                    dtype=torch.long
                    ).to(self.device)
                flattened_targets = torch.cat([y[i, :target_lengths[i]] for i in range(result.size(1))])
                loss = self.criterion(
                    result,
                    flattened_targets,
                    input_lengths,
                    target_lengths
                )
                total_loss += loss.item()

                pred_ids=result.argmax(dim=-1).transpose(0,1).cpu().numpy()
                for i in range(pred_ids.shape[0]):
                    pred_text_ids=pred_ids[i]
                    decoded_pred_text=[]
                    prev_id= -1
                    for id_val in pred_text_ids:
                        if id_val == self.blank_id:
                            prev_id=id_val
                            continue
                        if id_val != prev_id:
                            decoded_pred_text.append(self.id_to_char.get(id_val, ''))
                        prev_id=id_val
                    pred_text="".join(decoded_pred_text)
                    ref_text=texts[i]

                    self.wer_metric.update(pred_text,ref_text)
                    self.cer_metric.update(pred_text,ref_text)

                del loss,result,x,y,input_lengths,target_lengths
                clear_memory()

        total_loss /= len(self.test_loader)
        wer_score=self.wer_metric.compute().item()
        cer_score=self.cer_metric.compute().item()
        wandb.log({"val_loss": total_loss, "WER":wer_score, "CER":cer_score,"epoch": epoch})

    @save_checkpoint
    def train(self,epoch:int):
        """Iterates over the whole training data and train the models
        for a single epoch
        """
        torch.autograd.set_detect_anomaly(True)
        total_loss = 0
        self.set_train_mode()

        prev_info=None


        progress_bar=tqdm(
            enumerate(self.train_loader,start=1),
            total=len(self.train_loader),
            desc=f"Epoch {epoch} [Training]",
            leave=False,
            dynamic_ncols=True
        )

        #for step, (x, (y, target_lengths)) in enumerate(self.train_loader,start=1):
        for step, (x, (y, target_lengths, texts,audio_paths)) in progress_bar:
            try:

                current_info = {
                    "step": step,
                    "x_shape": tuple(x.shape),  # (batch, time, feature)
                    "y_shape": tuple(y.shape),
                    "max_input_len": x.shape[1],
                    "max_target_len": y.shape[1],
                    "target_lengths": target_lengths.tolist()
                }




                x = x.to(self.device)
                y = y.to(self.device)
                target_lenths=target_lengths.to(self.device)

                if step==1:
                    print(f"[CHECK] model device: {next(self.model.parameters()).device}")
                    print(f"[CHECK] batch x device: {x.device}")

                self.optimizer.zero_grad()
                result = self.model(x)
                result = result.log_softmax(axis=-1)
                result = result.permute(1,0,2)
                #y = y[..., :result.shape[0]]
                """
                input_lengths = torch.full(
                    size=(y.shape[0],),
                    fill_value=y.shape[1],
                    dtype=torch.long
                    ).to(self.device)
                flattened_targets = torch.cat([y[i, :target_lengths[i]] for i in range(result.size(1))])
                """

                # 모델 출력 길이 (T)
                T = result.size(0)
                batch_size = y.size(0)

                # CTC input_lengths: 모델 출력 타임스텝 길이
                input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=T,
                    dtype=torch.long
                ).to(self.device)

                # Flattened targets
                flattened_targets = torch.cat([y[i, :target_lengths[i]] for i in range(batch_size)])


                loss = self.criterion(
                    result,
                    flattened_targets,
                    input_lengths,
                    target_lengths
                )
                loss_value=loss.item()
                loss.backward()
                self.optimizer.step()
                total_loss+=loss_value
            
                progress_bar.set_postfix({
                    "Loss": f"{loss_value:.4f}",
                    "Avg Loss": f"{(total_loss/step):.4f}"
                })

                if step%3000==0:
                    print(f"\n[DEBUG] Epoch {epoch} | Step {step}")
                    log_memory("Train Loop")
                    check_tensor_growth()
                    print_gpu_status(step)

                if step%10==0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train_step_loss": loss_value,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": (epoch * len(self.train_loader)) + step
                    })
                
                
                pred_ids=result.argmax(dim=-1).transpose(0,1).cpu().numpy()
                batch_preds=[]
                for i in range(pred_ids.shape[0]):
                    pred_text_ids=pred_ids[i]
                    decoded_pred_text=[]
                    prev_id= -1
                    for id_val in pred_text_ids:
                        if id_val == self.blank_id:
                            prev_id=id_val
                            continue
                        if id_val != prev_id:
                            decoded_pred_text.append(self.id_to_char.get(id_val, ''))
                        prev_id=id_val
                    pred_text="".join(decoded_pred_text)
                    ref_text=texts[i]
                    self.wer_metric.update(pred_text, ref_text)
                    self.cer_metric.update(pred_text, ref_text)

                    if i<3:
                        batch_preds.append((pred_text,ref_text))

                prev_info=current_info
                

                self.audio_table = wandb.Table(columns=["audio","ref_text","pred_text"])
                if step % 100 == 0:
                    wer_score = self.wer_metric.compute().item()
                    cer_score = self.cer_metric.compute().item()
                    val_losses = []
                    self.set_test_mode()
                    pred=batch_preds[0][0] if batch_preds else ""
                    ref=batch_preds[0][1] if batch_preds else ""
                    audio_path=audio_paths[0] if len(audio_paths)>0 else None
                    raw_pred_ids=pred_ids[0]
                    #print(f"[DEBUG] Raw pred_ids (first 10): {raw_pred_ids[:10].tolist()}")
                    #print(f" Pred: {pred}")
                    #print(f" Ref: {ref}")
                    # ✅ Ref 토크나이징 확인
                    #ref_encoded = self.tokenizer.encode(ref)  # 문자열 → id 리스트
                    #print(f"[DEBUG] Ref encoded (first 10): {ref_encoded[:10]}")

                    #decoded_check = self.tokenizer.decode(ref_encoded)  # id 리스트 → 문자열
                    #print(f"[DEBUG] Decoded from IDs: {decoded_check}")

                    
                    if audio_path:
                        self.audio_table.add_data(
                            wandb.Audio(audio_path,caption="fStep {Step}"),
                            ref,
                            pred
                        )

                    with torch.no_grad():
                        for i, (vx, (vy, v_target_lengths, _, _)) in enumerate(self.test_loader):
                            if i >= 3:  # ✅ 첫 3개 배치만 검증 (속도)
                                break
                            vx = vx.to(self.device)
                            vy = vy.to(self.device)
                            v_target_lengths = v_target_lengths.to(self.device)

                            v_result = self.model(vx).log_softmax(axis=-1).permute(1, 0, 2)
                            V_T = v_result.size(0)
                            v_input_lengths = torch.full((vy.size(0),), V_T, dtype=torch.long).to(self.device)
                            v_flattened_targets = torch.cat([vy[j, :v_target_lengths[j]] for j in range(vy.size(0))])

                            v_loss = self.criterion(v_result, v_flattened_targets, v_input_lengths, v_target_lengths)
                            val_losses.append(v_loss.item())

                    avg_val_loss = mean(val_losses)
                    wandb.log({
                        "train_step_loss": loss_value,
                        "train_step_WER": wer_score,
                        "train_step_CER": cer_score,
                        "epoch": epoch,
                        "sample_audio_and_text": self.audio_table,
                        "step": (epoch * len(self.train_loader)) + step
                    },commit=True)
                    print(f"\n[DEBUG] Step {step} | Val Loss: {avg_val_loss:.4f} | WER:{wer_score:.4f} | CER: {cer_score:.4f}")
                    self.set_train_mode()
                    self.wer_metric.reset()
                    self.cer_metric.reset()
                    
                del loss, result,x,y,input_lengths, target_lengths
                clear_memory()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("\n======================")
                    print("🚨 [OOM ERROR] CUDA Out Of Memory!")
                    print("🔹 Previous Batch Info:")
                    if prev_info:
                        print(f"  - Step: {prev_info['step']}")
                        print(f"  - x_shape: {prev_info['x_shape']}, y_shape: {prev_info['y_shape']}")
                        print(f"  - max_input_len: {prev_info['max_input_len']}, max_target_len: {prev_info['max_target_len']}")
                        print(f"  - target_lengths: {prev_info['target_lengths']}")
                    else:
                        print("  - No previous batch info (first batch)")

                    print("\n🔹 Current Batch Info:")
                    print(f"  - Step: {current_info['step']}")
                    print(f"  - x_shape: {current_info['x_shape']}, y_shape: {current_info['y_shape']}")
                    print(f"  - max_input_len: {current_info['max_input_len']}, max_target_len: {current_info['max_target_len']}")
                    print(f"  - target_lengths: {current_info['target_lengths']}")
                    print(f"\n🔹 GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
                    print("======================\n")

                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e



        total_loss /= len(self.train_loader)


def get_criterion(blank_id: int) -> Module:
    return torch.nn.CTCLoss(blank_id)


def get_optimizer(model: Module, params: dict) -> object:
    return AdamWarmup(
        model.parameters(),
        **params
        )


def load_model(
        model_params: dict,
        checkpoint_path=None
        ) -> Module:
    model = Model(**model_params)

    if checkpoint_path is not None:
        model.load_state_dict(
            torch.load(checkpoint_path)[MODEL_KEY]
            )

    return model


def get_data_loader(
        file_path: Union[str, Path],
        tokenizer: CharTokenizer,
        text_pipeline: IPipeline,
        audio_pipeline: IPipeline,
        ):
    return DataLoader(
        file_path,
        text_pipeline,
        audio_pipeline,
        tokenizer,
        hparams.training.batch_size,
        hparams.data.sampling_rate,
        hparams.data.hop_length,
        hparams.data.files_sep,
        hparams.data.csv_file_keys,
        num_workers=hparams.training.num_workers
        )

def get_tokenizer():
    vocab_path = hparams.tokenizer.get("vocab_path")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"[ERROR] Char vocab file not found: {vocab_path}")
    print(f"[INFO] Using CharTokenizer with vocab: {vocab_path}")
    return CharTokenizer(vocab_path)

def get_train_test_loaders(
        tokenizer: CharTokenizer,
        audio_pipeline: IPipeline,
        text_pipeline: IPipeline
        ) -> tuple:
    return (
        get_data_loader(
            hparams.data.training_file,
            tokenizer,
            text_pipeline,
            audio_pipeline
            ),
        get_data_loader(
            hparams.data.testing_file,
            tokenizer,
            text_pipeline,
            audio_pipeline
            )
    )



def get_trainer() -> Trainer:
    gpu_id = hparams.get("gpu_id", 0)  # 기본값 0
    device = f"{hparams.device}:{gpu_id}" if hparams.device == 'cuda' else hparams.device
    #device = hparams.device
    tokenizer = get_tokenizer()
    blank_id = tokenizer.special_tokens.blank_id
    vocab_size = tokenizer.vocab_size
    text_pipeline, audio_pipeline = get_pipelines(
        get_audpipe_params()
    )
    model = load_model(
        get_model_params(vocab_size),
        checkpoint_path=hparams.checkpoint
        ).to(device)
    train_loader, test_loader = get_train_test_loaders(
        tokenizer,
        audio_pipeline,
        text_pipeline
    )

    #모델 확인
    print("📦 Model summary:")
    summary(
        model,
        input_size=(1, 128, 80),
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
        verbose=1,
        device=device  # 'cpu' or 'cuda'
    )

    return Trainer(
        criterion=get_criterion(blank_id),
        optimizer=get_optimizer(
            model, get_optim_params()
            ),
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=hparams.training.epochs,
        tokenizer=tokenizer,
        blank_id=blank_id
    )


if __name__ == '__main__':
    trainer = get_trainer()
    trainer.fit()
    wandb.finish() 
