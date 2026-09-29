"""
오디오 중심 주파수 옮기기 (16부). 모델 · 프로토콜 · 수신기는 그대로 1500 Hz 기준으로 두고, 소리만 옮긴다.
  송신: 만든 송신 오디오 (1500 Hz 중심) → 해석 신호 (Hilbert, FFT) × e^{j2π·Δf·t} 의 실수부 → 선택 중심
  수신: 입력 → 해석 신호 (FIR Hilbert, 흐름 상태 유지) × e^{-j2π·Δf·t} 의 실수부 → 1500 Hz 중심으로 되돌려 수신기에
실수 신호를 그냥 곱하면 거울상이 생겨 (f ± Δf) 다른 주파수 신호가 수신 대역에 겹치므로 해석 신호로 한 방향만 옮긴다.
"""
import numpy as np
from scipy.signal import hilbert, oaconvolve

BASE_HZ = 1500.0               # 모델 · 프로토콜 기준 중심
PASS_LO, PASS_HI = 300.0, 2700.0   # 무전기 통과 대역 (중심 선택 범위 = 이 안에 모드 대역 절반씩)


def center_range(bw_hz):
    """모드 점유 대역 bw 가 통과 대역 안에 드는 중심 범위 (Hz, 10 Hz 단위로 안쪽)"""
    lo = PASS_LO + bw_hz / 2.0
    hi = PASS_HI - bw_hz / 2.0
    return float(np.ceil(lo / 10) * 10), float(np.floor(hi / 10) * 10)


def shift_audio(a, fs, df):
    """송신 오디오 전체를 df Hz 옮김 (df = 0 이면 그대로)"""
    a = np.asarray(a)
    if abs(df) < 1e-9 or len(a) == 0:
        return a
    from scipy.fft import next_fast_len
    z = hilbert(np.asarray(a, np.float64), N=next_fast_len(len(a) + 4096))[:len(a)]    # 빠른 FFT 길이 (16부: 2.1M 샘플 230 → 수십 ms)
    fsi, dfi = int(round(fs)), int(round(df))
    if abs(fs - fsi) < 1e-9 and abs(df - dfi) < 1e-9 and dfi != 0:
        P = fsi // np.gcd(fsi, abs(dfi))
        car = np.resize(np.exp(2j * np.pi * dfi * np.arange(P) / fs), len(a))
    else:
        car = np.exp(2j * np.pi * df * np.arange(len(a)) / fs)
    return np.real(z * car).astype(a.dtype, copy=False)


class StreamShift:
    """수신 입력 흐름을 df Hz 옮김 (조각마다, 상태 유지). FIR Hilbert (홀수 탭, Blackman 창) + 같은 지연의 실수 경로.
    지연 (ntap-1)/2 샘플 (48 kHz 511 탭 = 5.3 ms). df = 0 이면 만들지 않는다 (앱이 그대로 넘김)"""

    def __init__(self, fs, df, ntap=511):
        self.fs, self.df = float(fs), float(df)
        m = np.arange(ntap) - (ntap - 1) // 2
        h = np.zeros(ntap)
        odd = m % 2 != 0
        h[odd] = 2.0 / (np.pi * m[odd])
        self.h = h * np.blackman(ntap)
        self.d = (ntap - 1) // 2
        self.tail = np.zeros(ntap - 1)
        self.n = 0                                  # 출력 샘플 번호 (믹서 위상)

    def process(self, x):
        x = np.asarray(x, np.float64)
        buf = np.concatenate([self.tail, x])
        im = oaconvolve(buf, self.h, mode="valid")          # len(x)
        re = buf[len(self.tail) - self.d:len(buf) - self.d]
        self.tail = buf[-len(self.tail):]
        k = self.n + np.arange(len(x))
        self.n += len(x)
        out = np.real((re + 1j * im) * np.exp(-2j * np.pi * self.df * k / self.fs))
        return out.astype(np.float32)
