"""
P3: MDN-LSTM 궤적 생성 모델 (Graves 2013 handwriting synthesis 스타일).

매 8ms 스텝마다 '다음 canonical 오프셋'을 2D bivariate Gaussian 혼합분포로 예측하고,
end-of-move(EOM) 확률을 함께 출력한다. 결정론적 회귀(MSE)와 달리 분포를 샘플링하므로
사람처럼 매번 다른 곡선이 나온다.

출력(스텝당): M개 혼합성분 × (π, μx, μy, σx, σy, ρ) + EOM logit = 6M + 1
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 9   # 입력 feature 차원 (학습 시 dataset 과 동일). 추론은 ckpt config의 input_dim 사용.

NUM_MIXTURES = 20
HIDDEN = 256
NUM_LAYERS = 2

_LOG_2PI = math.log(2.0 * math.pi)


class MDNLSTM(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden=HIDDEN,
                 num_layers=NUM_LAYERS, num_mixtures=NUM_MIXTURES, dropout=0.1):
        super().__init__()
        self.M = num_mixtures
        self.hidden = hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 6 * num_mixtures + 1)

    def forward(self, x, hidden=None):
        """x: (B,T,FEATURE_DIM) → raw params (B,T,6M+1), hidden."""
        out, hidden = self.lstm(x, hidden)
        return self.head(out), hidden

    # ───────── 파라미터 파싱 ─────────
    def parse(self, raw):
        """raw(...,6M+1) → dict(log_pi, mu, sigma, rho, eom_logit)."""
        M = self.M
        pi_logits = raw[..., 0:M]
        mu = raw[..., M:3 * M].reshape(*raw.shape[:-1], M, 2)
        log_sigma = raw[..., 3 * M:5 * M].reshape(*raw.shape[:-1], M, 2)
        rho = torch.tanh(raw[..., 5 * M:6 * M]) * 0.999
        eom_logit = raw[..., 6 * M]

        log_sigma = torch.clamp(log_sigma, -7.0, 5.0)
        sigma = torch.exp(log_sigma)
        log_pi = F.log_softmax(pi_logits, dim=-1)
        return {"log_pi": log_pi, "mu": mu, "sigma": sigma, "rho": rho,
                "eom_logit": eom_logit}


def _log_bivariate(target, mu, sigma, rho):
    """
    target:(B,T,2)  mu:(B,T,M,2)  sigma:(B,T,M,2)  rho:(B,T,M)
    반환: 성분별 로그밀도 (B,T,M)
    """
    x = target.unsqueeze(-2)                       # (B,T,1,2)
    sx, sy = sigma[..., 0], sigma[..., 1]          # (B,T,M)
    mux, muy = mu[..., 0], mu[..., 1]
    dx = (x[..., 0] - mux) / sx
    dy = (x[..., 1] - muy) / sy
    one_m_rho2 = 1.0 - rho * rho
    z = dx * dx + dy * dy - 2.0 * rho * dx * dy
    log_n = (
        -_LOG_2PI
        - torch.log(sx) - torch.log(sy)
        - 0.5 * torch.log(one_m_rho2)
        - z / (2.0 * one_m_rho2)
    )
    return log_n                                   # (B,T,M)


def mdn_loss(raw, model, target, eom_target, mask, eom_weight=1.0):
    """MDN NLL + EOM BCE (mask로 패딩 제외). 스텝 단위 평균."""
    p = model.parse(raw)
    log_n = _log_bivariate(target, p["mu"], p["sigma"], p["rho"])     # (B,T,M)
    log_prob = torch.logsumexp(p["log_pi"] + log_n, dim=-1)           # (B,T)
    nll = -log_prob

    eom_bce = F.binary_cross_entropy_with_logits(
        p["eom_logit"], eom_target, reduction="none")                 # (B,T)

    denom = mask.sum().clamp(min=1.0)
    nll_mean = (nll * mask).sum() / denom
    eom_mean = (eom_bce * mask).sum() / denom
    total = nll_mean + eom_weight * eom_mean
    return total, nll_mean.detach(), eom_mean.detach()


@torch.no_grad()
def sample_step(raw_step, model, bias=0.0, temperature=1.0, rng=None):
    """
    한 스텝의 raw 파라미터(..,6M+1)에서 오프셋 1개 샘플 + EOM 확률.
    bias>0: 혼합 가중치 sharpen & σ 축소 → 더 또렷/덜 떨리는 궤적(Graves bias sampling).
    반환: offset(2,) tensor, eom_prob(float)
    """
    M = model.M
    pi_logits = raw_step[..., 0:M] * (1.0 + bias)
    mu = raw_step[..., M:3 * M].reshape(M, 2)
    log_sigma = torch.clamp(raw_step[..., 3 * M:5 * M].reshape(M, 2), -7.0, 5.0)
    sigma = torch.exp(log_sigma - bias) * temperature
    rho = torch.tanh(raw_step[..., 5 * M:6 * M]) * 0.999
    eom_prob = torch.sigmoid(raw_step[..., 6 * M]).item()

    g = rng if rng is not None else torch
    k = torch.distributions.Categorical(logits=pi_logits).sample().item()

    sx, sy = sigma[k, 0], sigma[k, 1]
    r = rho[k]
    z1 = torch.randn((), device=raw_step.device)
    z2 = torch.randn((), device=raw_step.device)
    dx = mu[k, 0] + sx * z1
    dy = mu[k, 1] + sy * (r * z1 + torch.sqrt(torch.clamp(1 - r * r, min=1e-6)) * z2)
    return torch.stack([dx, dy]), eom_prob
