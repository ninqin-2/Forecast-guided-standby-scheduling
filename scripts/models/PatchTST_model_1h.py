import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 4. 模型组件 (含可学习时间嵌入) =================
class InstantNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)


class TimeRandomEffectEmbedding(nn.Module):
    """
    正统随机效应（mixed-effects）版本的时间嵌入：
    - 每个时间类别（hour/weekday/day）对应一个随机效应向量 b_g
    - 近似后验 q(b_g)=N(mu_g, diag(sigma_g^2))
    - 先验 p(b_g)=N(0, prior_std^2 I)
    - 训练：重参数化采样；推断：默认用后验均值（也可 MC 采样）

    输出：
      emb: [B, L, D]
      kl : 标量（对一个 batch 的 KL 平均）
    """

    def __init__(self, d_model: int, prior_std: float = 1.0, min_std: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.prior_std = prior_std
        self.min_std = min_std

        # 固定效应（你原来的可学习 embedding，作为 fixed effects）
        self.hour_fixed = nn.Embedding(24, d_model)
        self.weekday_fixed = nn.Embedding(7, d_model)
        self.day_fixed = nn.Embedding(4, d_model)

        # 随机效应：每个类别一个 (mu, log_std)
        self.hour_mu = nn.Embedding(24, d_model)
        self.hour_logstd = nn.Embedding(24, d_model)

        self.weekday_mu = nn.Embedding(7, d_model)
        self.weekday_logstd = nn.Embedding(7, d_model)

        self.day_mu = nn.Embedding(4, d_model)
        self.day_logstd = nn.Embedding(4, d_model)

        # 初始化：让随机效应初始接近 0（更稳）
        nn.init.zeros_(self.hour_mu.weight)
        nn.init.zeros_(self.weekday_mu.weight)
        nn.init.zeros_(self.day_mu.weight)

        # logstd 初始小一点，避免一开始噪声太大
        nn.init.constant_(self.hour_logstd.weight, -3.0)
        nn.init.constant_(self.weekday_logstd.weight, -3.0)
        nn.init.constant_(self.day_logstd.weight, -3.0)

    def _sample_re(self, mu_emb, logstd_emb, training: bool, mc_samples: int = 0):
        """
        mu_emb/logstd_emb: [B,L,D]
        training=True: 采样一次
        training=False:
          - mc_samples<=0: 用均值（正统推断常用）
          - mc_samples>0 : MC 采样取平均（用于不确定性）
        """
        std = torch.exp(logstd_emb).clamp_min(self.min_std)

        if training:
            eps = torch.randn_like(std)
            return mu_emb + std * eps
        else:
            if mc_samples and mc_samples > 0:
                acc = 0.0
                for _ in range(mc_samples):
                    eps = torch.randn_like(std)
                    acc = acc + (mu_emb + std * eps)
                return acc / float(mc_samples)
            else:
                return mu_emb  # posterior mean

    def _kl_gaussian_diag_to_prior(self, mu, logstd):
        """
        KL( N(mu, sigma^2) || N(0, prior_std^2) ) for diagonal Gaussian.
        mu/logstd: [..., D]
        返回：[...,] 逐元素 KL（未求和 D 之前）
        """
        sigma2 = torch.exp(2.0 * logstd).clamp_min(self.min_std ** 2)
        prior2 = (self.prior_std ** 2)

        # KL = 0.5 * [ (sigma^2 + mu^2)/prior^2 - 1 + log(prior^2/sigma^2) ]
        kl = 0.5 * ((sigma2 + mu * mu) / prior2 - 1.0 + torch.log(prior2 / sigma2))
        return kl  # [..., D]

    def forward(self, x_mark, mc_samples: int = 0):
        """
        x_mark: [B, L, F], 其中：
          x_mark[:,:,1]=hour (0-23)
          x_mark[:,:,0]=weekday (0-6)
          x_mark[:,:,3]=day_bucket (0-3)
        """
        x_mark = x_mark.long()
        hour = x_mark[:, :, 1]
        weekday = x_mark[:, :, 0]
        day = x_mark[:, :, 3]

        # fixed effects
        hour_fx = self.hour_fixed(hour)
        weekday_fx = self.weekday_fixed(weekday)
        day_fx = self.day_fixed(day)

        # random effects posterior params
        hour_mu = self.hour_mu(hour)
        hour_logstd = self.hour_logstd(hour)

        weekday_mu = self.weekday_mu(weekday)
        weekday_logstd = self.weekday_logstd(weekday)

        day_mu = self.day_mu(day)
        day_logstd = self.day_logstd(day)

        # sample random effects
        hour_re = self._sample_re(hour_mu, hour_logstd, self.training, mc_samples)
        weekday_re = self._sample_re(weekday_mu, weekday_logstd, self.training, mc_samples)
        day_re = self._sample_re(day_mu, day_logstd, self.training, mc_samples)

        emb = (hour_fx + weekday_fx + day_fx) + (hour_re + weekday_re + day_re)

        # KL：对 batch&time 维求平均，对 D 求和（常用）
        kl_hour = self._kl_gaussian_diag_to_prior(hour_mu, hour_logstd).sum(dim=-1)      # [B,L]
        kl_week = self._kl_gaussian_diag_to_prior(weekday_mu, weekday_logstd).sum(dim=-1)# [B,L]
        kl_day  = self._kl_gaussian_diag_to_prior(day_mu, day_logstd).sum(dim=-1)        # [B,L]

        kl = (kl_hour + kl_week + kl_day).mean()  # 标量
        return emb, kl

class PatchTST_WithSideInfo(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, side_vars_indices,
                 patch_len=12, stride=6, d_model=128, n_heads=4,
                 n_layers=2, ff_dim=128, dropout=0.3):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.side_vars_indices = side_vars_indices
        self.n_side_vars = len(side_vars_indices)

        # === 核心修改：定义总 Side 维度 = 原始索引(5) + 电价(1) = 6 ===
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 2
        self.pad_len = max(0, ((self.n_patches - 1) * stride + patch_len) - seq_len)

        self.instant_norm = InstantNorm()
        self.patch_embed = nn.Linear(patch_len * n_vars, d_model)
        self.time_embedding = TimeRandomEffectEmbedding(d_model, prior_std=1.0)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                                               dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # === 知识对齐组件：输入维度设为 6 ===
        self.n_total_side = len(side_vars_indices)

        # 增大投影层的表达能力
        self.exogenous_proj = nn.Sequential(
            nn.Linear(self.n_total_side, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        self.stochastic_gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保噪声标准差永远为正
        )
        self.alignment_layer = ExogenousAlignment(d_model, n_heads, dropout)
        # Head 层增加外部知识直连路径
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * d_model + self.n_total_side, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, pred_len * 2)
        )

    def _pad_last_value(self, x):
        if self.pad_len <= 0: return x
        last = x[:, -1:, :]
        return torch.cat([x, last.repeat(1, self.pad_len, 1)], dim=1)

    def forward(self, x, x_mark, y_indice):
        B = x.shape[0]

        # 1. 构建 Side Info (含电价)
        info1 = x[:, -1, self.side_vars_indices[1:]]
        info2 = y_indice[:, :, self.side_vars_indices[0]]
        side_info = torch.cat((info1, info2), dim=1)

        # 2. Patching & Transformer 主干 (高性能逻辑保留)
        x_norm = self.instant_norm(x)
        x_pad = self._pad_last_value(x_norm)
        patches = x_pad.permute(0, 2, 1).unfold(2, self.patch_len, self.stride)
        patches = patches.permute(0, 2, 3, 1).contiguous().view(B, self.n_patches, -1)
        z = self.patch_embed(patches)
        time_emb, kl_time = self.time_embedding(x_mark)


        z = z + self.pos_embed + time_emb
        z = self.encoder(self.drop(z))  # 提取序列趋势

        # ============================================================
        # === 核心增强：强制知识对齐 ===
        # ============================================================
        exo_token = self.exogenous_proj(side_info).unsqueeze(1)  # [B, 1, D]

        # 强制交互：使用带门控的 Cross-Attention
        z = self.alignment_layer(z, exo_token)
        # ============================================================

        # 3. 输出层
        rep = z.reshape(B, -1)
        # 将交互后的特征再次与 side_info 拼接，确保“物理常数”在输出端有强存在感
        combined = torch.cat([rep, side_info], dim=1)
        output = self.head(combined).reshape(B, self.pred_len, 2)

        mu = F.softplus(output[:, :, 0])
        alpha = F.softplus(output[:, :, 1]) + 1e-4
        return torch.stack([mu, alpha], dim=-1), kl_time


class ExogenousAlignment(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # 增大注意力权重的作用，使用多个 heads 捕捉电价的不同影响
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 门控机制：控制外部知识对序列的干预程度
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x, exo_token):
        # 1. 交互
        attn_out, _ = self.cross_attn(query=x, key=exo_token, value=exo_token)

        # 2. 增强门控：决定每个时间 patch 应该在多大程度上被电价/流量修正
        # 将原始特征与对齐特征拼接，计算门控系数
        g = self.gate(torch.cat([x, attn_out], dim=-1))
        x = self.norm1(x + g * attn_out)  # 门控引导的残差连接

        # 3. 前馈增强
        x = self.norm2(x + self.ffn(x))
        return x