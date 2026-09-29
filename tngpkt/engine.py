"""
NeuroMod 엔진 — GUI 가 쓰는 얇은 래퍼.

변조/복조 로직은 새로 만들지 않는다. 5단계 modem5.py (코스타스 프리앰블/포스트앰블 +
4단계 AI 모뎀·FEC) 의 함수를 그대로 호출한다. 프리앰블이 없는 옛 형식 WAV 도 modem5 가
4단계 방식으로 자동 복조한다.
여기서 하는 일은 GUI 에 필요한 주변 처리뿐이다.
  * 모델을 한 번만 불러오고, 여러 스레드가 동시에 쓰지 않도록 잠금
  * 오디오 장치 샘플레이트(예: 44100Hz) ↔ 모뎀이 받는 샘플레이트(2000Hz 의 정수배) 변환
  * WAV 입출력 (16/24/32비트, 스테레오 등 일반 WAV 를 받도록)
  * 수신 SNR 추정 (표시용 — 복조에는 쓰지 않는다)
"""

import threading
import time
from dataclasses import dataclass, field
from math import gcd

import numpy as np
import torch
from scipy import signal as sg
from scipy.io import wavfile

from tngpkt.modem import load_model, baseband_to_audio, audio_to_baseband, write_wav
from tngpkt.modem4 import detect_span4
from tngpkt.modem5 import text_to_baseband5, decode_audio5, first_text
from tngpkt.config import PreambleConfig
from tngpkt.preamble import CostasDetector

MODEM_FS = 48000.0          # 모뎀 내부 처리용 오디오 샘플레이트 (2000Hz 의 정수배)
SAVE_FS = 8000.0            # WAV 저장 샘플레이트 (3단계 규격)


@dataclass
class DecodeResult:
    ok: bool
    text: str = ""
    snr_db: float = float("nan")
    reason: str = ""
    elapsed: float = 0.0
    df_hz: float = float("nan")     # 추정 주파수 오차 (코스타스로 찾았을 때)
    mode: str = ""                  # 복조 경로 (코스타스 A+B / A 만 / 옛 형식 호환)
    viz: dict = field(default_factory=dict)    # 시각화용 원자료 (오디오 참조·검출 정보·LLR) — 계산은 viz.py
    info: dict = field(default_factory=dict)


TORCH_THREADS = 1
APP_DEVICE = "cpu"          # 28부: 앱 · 진단은 GPU 가 있어도 CPU 만 (저사양 PC 원칙, CPU 가 바쁠 때 GPU 경로가 더 느렸음 1.3 vs 4.1배)


def app_threads():
    """27부: 앱 · 진단의 torch CPU 스레드 수. 실시간 수신은 0.25 s 마다 작은 연산이라 스레드를 늘려도 빨라지지 않고
    (진단: TNG5 1 스레드 9.5배 · 2 스레드 9.7배), CPU 가 바쁠 때 여러 스레드가 서로 기다려 크게 느려진다 (6 스레드 0.1배 · 1 스레드 1.4배)"""
    torch.set_num_threads(TORCH_THREADS)


def resample(x, fs_from, fs_to):
    """정수비 폴리페이즈 리샘플링"""
    fs_from, fs_to = int(round(fs_from)), int(round(fs_to))
    if fs_from == fs_to:
        return np.asarray(x, dtype=np.float64)
    g = gcd(fs_from, fs_to)
    return sg.resample_poly(np.asarray(x, dtype=np.float64), fs_to // g, fs_from // g)


def read_any_wav(path):
    """일반 WAV 를 읽어 (float64 모노 -1~1, fs) 로 돌려준다."""
    fs, d = wavfile.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    if d.dtype == np.int16:
        x = d / 32768.0
    elif d.dtype == np.int32:
        x = d / 2147483648.0
    elif d.dtype == np.uint8:
        x = (d.astype(np.float64) - 128) / 128.0
    else:
        x = d.astype(np.float64)
    return np.asarray(x, dtype=np.float64), float(fs)


def estimate_snr_2500(audio, fs, cfg, df_hz=0.0):
    """
    2500Hz 대역 기준 SNR 추정 (표시용).

    신호 대역(1500±250Hz) 밖, 그러나 SSB 통과대역 안인 두 구간의 잡음 밀도를 재서
    신호 대역의 잡음을 빼고, 2500Hz 폭으로 환산한다. 버스트 구간만 쓴다.
    """
    try:
        x = resample(audio, fs, 8000)
        f8 = 8000.0
        bb = audio_to_baseband(x, f8, cfg)
        s0, s1, _ = detect_span4(bb, cfg)
        k = int(f8 / cfg.fs_base)
        seg = x[s0 * k:s1 * k]
        if len(seg) < 2048:
            seg = x
        f, P = sg.welch(seg, fs=f8, nperseg=2048)
        df = f[1] - f[0]
        c = cfg.audio_center_hz + (df_hz if np.isfinite(df_hz) else 0.0)   # 주파수 오차만큼 옮겨 잰다
        noise_bins = ((f > 400) & (f < c - 400)) | ((f > c + 400) & (f < 2600))
        sig_bins = (f >= c - 250) & (f <= c + 250)
        n0 = np.median(P[noise_bins])
        ps = np.sum(P[sig_bins]) * df - n0 * 500.0
        if n0 <= 0 or ps <= 0:
            return float("inf") if ps > 0 else float("nan")
        return 10 * np.log10(ps / (n0 * 2500.0))
    except Exception:
        return float("nan")


class ModemEngine:
    def __init__(self, model_path="models/stage3.pt", device=None):
        dev = device or APP_DEVICE
        self.device = torch.device(dev)
        self.model, self.cfg = load_model(model_path, self.device)
        self.pc = PreambleConfig()
        self.det = CostasDetector(self.pc, self.cfg.fs_base)
        self._lock = threading.Lock()
        # 첫 호출 지연을 없애기 위해 한 번 데워 둔다
        self.encode("warmup", MODEM_FS)

    # ---------------------------------------------------------------- 송신
    def encode(self, text, fs):
        """텍스트 → 실수 오디오 (fs), 최대 -3dBFS. 앞뒤 0.2초 무음 포함."""
        with self._lock:
            bb, _ = text_to_baseband5(text, self.model, self.cfg, self.device, self.pc)
            fs_m = fs if (fs % self.cfg.fs_base == 0) else MODEM_FS
            a = baseband_to_audio(bb, self.cfg, fs_m)
        a = resample(a, fs_m, fs)
        a = a / max(np.max(np.abs(a)), 1e-12) * 10 ** (-3 / 20)
        return a

    def save_wav(self, text, path, fs=SAVE_FS):
        a = self.encode(text, fs)
        return write_wav(path, a, fs, headroom_db=3.0)

    # ---------------------------------------------------------------- 수신
    def decode(self, audio, fs):
        t0 = time.time()
        x = np.asarray(audio, dtype=np.float64)
        if fs % self.cfg.fs_base != 0:
            x = resample(x, fs, MODEM_FS)
            fs = MODEM_FS
        if np.max(np.abs(x)) < 1e-6:
            return DecodeResult(False, reason="무음", elapsed=time.time() - t0)
        with self._lock:
            msgs = decode_audio5(x, fs, self.model, self.cfg, self.device, self.pc, self.det)
        text, m = first_text(msgs)
        df = float(m.get("df", float("nan"))) if m.get("a") is not None else float("nan")
        snr = estimate_snr_2500(x, fs, self.cfg, df)
        el = time.time() - t0
        info = m.get("info", {}) if m else {}
        mode = m.get("mode", "") if m else ""
        viz = {"audio": x, "fs": fs, "msg": m}
        if text is None:
            if m.get("a") is None:
                mode = "코스타스 미검출"          # 옛 형식 복조까지 시도했지만 실패
            return DecodeResult(False, snr_db=snr, reason=info.get("reason", "복호 실패"),
                                elapsed=el, info=info, df_hz=df, mode=mode, viz=viz)
        return DecodeResult(True, text=text, snr_db=snr, elapsed=el, info=info, df_hz=df, mode=mode,
                            viz=viz)
