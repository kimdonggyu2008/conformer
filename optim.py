from torch.optim import Adam
from typing import Tuple, Optional
import math


class AdamWarmup:
    def __init__(
            self,
            parameters,
            betas: Tuple[float, float],
            eps: float,
            weight_decay: float,
            warmup_steps: int,
            model_dim: int,
            scaler: float = 1.0,
            step_size: int = 1,
            lr: Optional[float] = None,
            *args,
            **kwargs
            ):
        
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        self.scaler = scaler
        self.peak = self.scaler / math.sqrt(self.model_dim)
        #self.inv_warmup_steps = 1 / math.sqrt(self.warmup_steps ** 3)
        self.step_size = step_size
        self.counter = 0
        self.lr_override= lr

        if lr is None:
            self.lr_override = False
            self.peak = self.scaler / math.sqrt(self.model_dim)
        else:
            self.lr_override = True
            self.peak = float(lr)  # 문자열 들어와도 안전하게 변환

        # ✅ Adam 옵티마이저 생성
        self.optimizer = Adam(
            parameters,
            lr=self.peak,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_lr(self, step: int) -> float:
        if self.lr_override:
            return self.peak
        return self.peak * min(step**(-0.5), step * (self.warmup_steps ** -1.5))

    def step(self) -> None:
        self.counter += self.step_size
        lr = self.get_lr(self.counter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()
