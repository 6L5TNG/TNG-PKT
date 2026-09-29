"""
3단계 신경망 — 연속 스트림 송신 + 문맥 창 수신.

송신 (StreamTransmitter)
    비트 (B, N, k)  ──► [비트 + 위치임베딩(프레임 안 위치)] ──► MLP ──► 심볼당 sps 샘플
                    ──► 이어붙여 길이 N*sps 스트림 ──► 고정 LPF **한 번** ──► 전력 정규화

    2단계와 결정적으로 다른 점: 프레임을 따로따로 필터링하지 않는다.
    필터가 프레임 경계를 가로질러 걸리므로 경계 불연속이 애초에 생기지 않고,
    송신 NN 은 경계까지 포함해 학습한다.

수신 (ContextReceiver)
    프레임 앞뒤로 ctx_symbols 만큼 더 본 창 (B, win_len) ──► 고정 LPF ──► RMS 정규화
    ──► 샘플률 CNN ──► 심볼률 다운샘플 ──► 확장 잔차 블록 ──► 전역 문맥(FiLM)
    ──► 가운데 S 개 위치의 비트 로짓 (B, S, k)

    여유 구간을 함께 보는 이유: LPF 가 프레임 경계 너머로 번지므로,
    프레임 끝 심볼을 제대로 읽으려면 이웃 프레임의 앞부분을 봐야 한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from waveform import FixedLPF


def _mlp(i, hidden, o):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers += [nn.Linear(d, o)]
    return nn.Sequential(*layers)


class StreamTransmitter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c = cfg
        S, k, E = cfg.symbols_per_frame, cfg.bits_per_symbol, cfg.pos_embed_dim
        self.pos = nn.Parameter(torch.randn(S, E) * 0.5)
        self.net = _mlp(k + E, tuple(cfg.tx_hidden), 2 * cfg.sps)
        self.lpf = FixedLPF(cfg.lpf_cutoff_hz, cfg.fs_base, cfg.lpf_taps, cfg.lpf_trans_hz)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        bits: (B, N, k) — N 은 심볼 총 개수 (프레임 수 × S, 프레임 경계로 나뉨)
        반환: (B, N*sps) complex, 샘플당 평균전력 1
        """
        c = self.c
        B, N, k = bits.shape
        idx = torch.arange(N, device=bits.device) % c.symbols_per_frame
        pos = self.pos[idx].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([2.0 * bits - 1.0, pos], dim=-1)
        y = self.net(x).view(B, N, c.sps, 2).reshape(B, N * c.sps, 2)
        w = torch.complex(y[..., 0], y[..., 1])
        w = self.lpf(w)                                   # 스트림 전체에 한 번만
        p = w.abs().pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12)
        return w * torch.rsqrt(p)


class _ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return x + h


class ContextReceiver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c = cfg
        C = cfg.rx_channels
        self.lpf = FixedLPF(cfg.lpf_cutoff_hz, cfg.fs_base, cfg.lpf_taps, cfg.lpf_trans_hz)
        self.stem = nn.Sequential(
            nn.Conv1d(2, C, 9, padding=4), nn.GELU(),
            nn.Conv1d(C, C, 9, padding=4), nn.GELU(),
        )
        self.down = nn.Conv1d(C, C, cfg.sps * 2, stride=cfg.sps, padding=cfg.sps // 2)
        self.blocks = nn.ModuleList([_ResBlock(C, d) for d in cfg.rx_dilations])
        self.glob = nn.Sequential(nn.Linear(2 * C, 2 * C), nn.GELU(), nn.Linear(2 * C, 2 * C))
        self.out_norm = nn.GroupNorm(8, C)
        self.head = nn.Conv1d(C, cfg.bits_per_symbol, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, win_len) complex → (B, S, k) 로짓"""
        c = self.c
        y = self.lpf(y)
        rms = y.abs().pow(2).mean(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        y = y / rms
        x = torch.stack([y.real, y.imag], dim=1)

        h = self.stem(x)
        h = self.down(h)                                   # (B, C, S + 2*ctx_symbols)
        h = h[:, :, :c.symbols_per_frame + 2 * c.ctx_symbols]
        for b in self.blocks:
            h = b(h)
        g = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        gs, gb = self.glob(g).chunk(2, dim=-1)
        h = h * (1 + gs).unsqueeze(-1) + gb.unsqueeze(-1)
        h = F.gelu(self.out_norm(h))
        h = h[:, :, c.ctx_symbols:c.ctx_symbols + c.symbols_per_frame]   # 가운데 S개만
        return self.head(h).transpose(1, 2)


class StreamModem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tx = StreamTransmitter(cfg)
        self.rx = ContextReceiver(cfg)
