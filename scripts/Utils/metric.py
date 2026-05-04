import numpy as np
import torch
import torch.nn as nn
from torch.distributions import NegativeBinomial

# ================= 3. 损失函数 (Negative Binomial) =================
class NBLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_params, target):
        mu = pred_params[:, :, 0]
        alpha = pred_params[:, :, 1]
        eps = 1e-6
        # PyTorch NB 参数: total_count=alpha, probs=alpha/(alpha+mu)
        probs = alpha / (alpha + mu + eps)
        dist = NegativeBinomial(total_count=alpha, probs=probs)
        nll = -dist.log_prob(target)
        return torch.mean(nll)

# ================= 1. 修正后的非对称损失函数 =================
class TimeAwareAsymmetricLoss(nn.Module):
    def __init__(self, night_penalty=10.0, morning_penalty=4.0):
        super().__init__()
        # 显式创建为 Tensor
        self.register_buffer('night_p', torch.tensor(night_penalty, dtype=torch.float32))
        self.register_buffer('morning_p', torch.tensor(morning_penalty, dtype=torch.float32))

    def forward(self, pred, target, hours):
        # 容错处理：强制确保 hours 与 pred 设备对齐
        if hours.device != pred.device:
            hours = hours.to(pred.device)

        residual = target - pred
        abs_loss = torch.abs(residual)

        # 掩码计算（现在 hours 已在 GPU，mask 也会在 GPU）
        night_mask = (hours >= 21) & (hours <= 23)
        over_mask = residual < 0
        morning_mask = (hours >= 0) & (hours <= 5)
        under_mask = residual > 0

        # 创建权重矩阵
        weights = torch.ones_like(abs_loss, device=pred.device)

        # 应用罚分
        weights[night_mask & over_mask] = self.night_p
        weights[morning_mask & under_mask] = self.morning_p

        return (abs_loss * weights).mean()

