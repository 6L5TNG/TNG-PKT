"""
파형 처리 도구 — 모두 미분 가능하고 배치 단위로 동작한다.

  * design_lpf      : 고정 저역통과 필터 계수 (점유대역폭을 구조적으로 보장)
  * FixedLPF        : 학습되지 않는 FIR 필터 모듈 (복소 신호용)
  * frac_delay      : 분수(소수점) 샘플 지연 — 타이밍 오차와 다중경로에 사용
  * papr_db         : 최대전력 대비 평균전력비
  * occupied_bw_hz  : 99% 에너지 점유대역폭
"""

import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 저역통과 필터
@functools.lru_cache(maxsize=64)
def design_lpf(cutoff_hz: float, fs: float, num_taps: int, trans_hz: float = 40.0) -> np.ndarray:
    """
    카이저 창 저역통과 FIR.
    신경망이 무엇을 내놓든 이 필터를 통과하므로 점유대역폭이 구조적으로 제한된다.
    """
    if num_taps % 2 == 0:
        num_taps += 1
    # 카이저 베타: 저지대역 감쇠 약 80dB 목표
    atten_db = 80.0
    beta = 0.1102 * (atten_db - 8.7)
    n = np.arange(num_taps) - (num_taps - 1) / 2
    fc = cutoff_hz / fs                      # 정규화 차단주파수 (0~0.5)
    h = 2 * fc * np.sinc(2 * fc * n)
    h *= np.kaiser(num_taps, beta)
    h /= h.sum()                             # DC 이득 1
    h = h.astype(np.float32)
    h.setflags(write=False)                  # 캐시된 배열 — 호출하는 쪽이 고치지 못하게
    return h


class FixedLPF(nn.Module):
    """복소 신호 (B, L) 에 실수 FIR 을 적용. 계수는 학습되지 않는다."""

    def __init__(self, cutoff_hz: float, fs: float, num_taps: int, trans_hz: float = 40.0):
        super().__init__()
        h = design_lpf(cutoff_hz, fs, num_taps, trans_hz)
        self.num_taps = len(h)
        self.pad = (self.num_taps - 1) // 2
        # conv1d 는 상관(correlation)이므로 계수를 뒤집어 둔다 (선형위상이라 대칭이지만 명시)
        self.register_buffer("h", torch.tensor(h[::-1].copy()).view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) complex → (B, L) complex"""
        xr = torch.stack([x.real, x.imag], dim=1)              # (B, 2, L)
        B, _, L = xr.shape
        y = F.conv1d(xr.reshape(B * 2, 1, L), self.h, padding=self.pad)
        y = y.reshape(B, 2, L)
        return torch.complex(y[:, 0], y[:, 1])


# ---------------------------------------------------------------- 분수 샘플 지연
def frac_delay(x: torch.Tensor, d: torch.Tensor, half: int = 8) -> torch.Tensor:
    """
    y(t) = x(t - d)  — d 는 샘플 단위 실수(배치마다 다름), 미분 가능.
    윈도잉된 sinc 보간 커널을 배치별로 만들어 grouped conv 로 한 번에 적용한다.

    x: (B, L) complex,  d: (B,) float
    """
    B, L = x.shape
    dev, dt = x.device, x.real.dtype
    # 커널 폭은 요청된 지연량까지 덮도록 넓힌다 (정수 지연 부분에서 탭이 잘리지 않게)
    span = half + int(torch.ceil(d.detach().abs().max()).item()) + 1
    n = torch.arange(-span, span + 1, device=dev, dtype=dt).view(1, -1)   # (1, K)
    arg = n - d.view(-1, 1).to(dt)                                        # (B, K)
    # sinc 보간 + 해밍 창 (커널 양끝을 부드럽게 잘라 링잉을 줄인다)
    sinc = torch.sinc(arg)
    win = 0.54 + 0.46 * torch.cos(math.pi * arg / (half + 1))
    win = torch.where(arg.abs() <= half + 1, win, torch.zeros_like(win))
    k = sinc * win
    k = k / k.sum(dim=1, keepdim=True).clamp(min=1e-8)                    # (B, K)

    xr = torch.stack([x.real, x.imag], dim=1)                             # (B, 2, L)
    xr = xr.reshape(1, B * 2, L)
    # 같은 커널을 실수부/허수부에 → 채널 2개씩 복사
    w = k.unsqueeze(1).repeat_interleave(2, dim=0)                        # (B*2, 1, K)
    # conv1d 는 상관이므로 지연을 얻으려면 커널을 뒤집어야 한다
    w = torch.flip(w, dims=[-1])
    y = F.conv1d(xr, w, padding=span, groups=B * 2).reshape(B, 2, L)
    return torch.complex(y[:, 0], y[:, 1])


# ---------------------------------------------------------------- 측정 도구
def papr_db(x: torch.Tensor) -> torch.Tensor:
    """x: (B, L) complex → 프레임별 PAPR [dB]"""
    p = x.abs().pow(2)
    return 10.0 * torch.log10(p.amax(dim=-1) / p.mean(dim=-1).clamp(min=1e-12))


def psd_db(x: np.ndarray, fs: float, nfft: int = 4096):
    """평균 주기도(periodogram). x: (B, L) complex → (freq, dB)"""
    w = np.hanning(x.shape[-1])
    X = np.fft.fftshift(np.fft.fft(x * w, n=nfft, axis=-1), axes=-1)
    p = (np.abs(X) ** 2).mean(axis=0)
    p /= p.max()
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    return f, 10 * np.log10(np.maximum(p, 1e-16))


def occupied_bw_hz(x: np.ndarray, fs: float, frac: float = 0.99, nfft: int = 4096) -> float:
    """전체 에너지의 frac(기본 99%) 가 들어가는 최소 대역폭 [Hz]"""
    w = np.hanning(x.shape[-1])
    X = np.fft.fftshift(np.fft.fft(x * w, n=nfft, axis=-1), axes=-1)
    p = (np.abs(X) ** 2).mean(axis=0)
    total = p.sum()
    # 중심(무게중심)에서 좌우로 넓혀가며 frac 을 채운다
    c = int(np.round((p * np.arange(nfft)).sum() / total))
    lo = hi = c
    acc = p[c]
    while acc < frac * total and (lo > 0 or hi < nfft - 1):
        left = p[lo - 1] if lo > 0 else -1
        right = p[hi + 1] if hi < nfft - 1 else -1
        if right >= left:
            hi += 1; acc += p[hi]
        else:
            lo -= 1; acc += p[lo]
    return (hi - lo + 1) * fs / nfft


def emission_bw(x, fs, level_db=-60.0, pad_s=1.0, smooth_hz=5.0, return_spectrum=False):
    """
    송신 한 번 전체의 점유대역폭 [Hz] — 5단계부터 공통 측정 기준.

    앞뒤에 무음을 붙이고, 테이퍼가 **무음 구간에만** 걸리는 창(Tukey)으로 전체를 한 번에
    FFT 한다. 그래서
      · 송신 시작/끝·구간 경계의 과도 구간을 가리지 않는다 (신호 전체가 창의 평평한 부분에 있다)
      · 짧은 창을 쓸 때 생기는 창 누설로 스펙트럼 가장자리가 부풀지 않는다
    단일 실현의 주기도는 요동이 커서, 선형 전력을 smooth_hz 폭으로 평균한 뒤 문턱을 잰다.

    (이전 bw_at_level 은 신호 구간에만 Hann 창을 씌워 시작·끝 램프를 가렸다.
     audio_bw_at_level 은 FFT 길이보다 긴 오디오의 앞부분만 봤다.)
    """
    x = np.asarray(x)
    npad = int(pad_s * fs)
    y = np.concatenate([np.zeros(npad, dtype=x.dtype), x, np.zeros(npad, dtype=x.dtype)])
    from scipy.signal.windows import tukey
    w = tukey(len(y), alpha=min(1.0, 2.0 * npad / len(y)))
    nfft = 1 << int(np.ceil(np.log2(len(y) * 4)))
    X = np.fft.fft(y * w, n=nfft)
    P = np.abs(X) ** 2
    f = np.fft.fftfreq(nfft, 1.0 / fs)
    if np.isrealobj(x):                       # 실수 신호는 양의 주파수만
        keep = f >= 0
        f, P = f[keep], P[keep]
    order = np.argsort(f)
    f, P = f[order], P[order]
    k = max(1, int(round(smooth_hz / (fs / nfft))))
    Ps = np.convolve(P, np.ones(k) / k, mode="same")
    Pd = 10 * np.log10(np.maximum(Ps / Ps.max(), 1e-30))
    idx = np.where(Pd > level_db)[0]
    bw = float(f[idx[-1]] - f[idx[0]]) if len(idx) else 0.0
    return (bw, f, Pd) if return_spectrum else bw
