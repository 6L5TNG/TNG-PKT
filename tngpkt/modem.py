"""
3단계 오디오 모뎀 CLI — 텍스트 ↔ WAV.

    python modem.py tx       --text "CQ CQ DE HL1ABC" --out out.wav
    python modem.py rx       --wav out.wav
    python modem.py loopback --text "..." --snr -3 --freq-off 4

핵심 규칙
  * 송신은 메시지 전체를 **하나의 연속 스트림**으로 만들고 고정 LPF 를 1회 통과시킨다.
    프레임을 따로 만들어 이어붙이지 않으므로 경계 스플래터가 생기지 않는다.
  * 시작/끝에는 램프를 걸어 서서히 켜고 끈다. 램프 구간은 가드 프레임이 덮는다.
  * 수신은 프레임 시작 위치를 **스스로 찾는다**. 파일럿이 없으므로,
    가능한 모든 정렬(0 ~ 프레임길이-1)에 대해 복호해 보고 CRC 통과 수가 가장 많은
    정렬을 고른다. 사람이 만든 상관기나 동기 워드는 쓰지 않는다.
  * 실제 무선 송신 기능은 없다. WAV 파일로만 주고받는다.
"""

from tngpkt import _io_utf8  # noqa: F401
import argparse
import functools
import os
import sys
import wave

import numpy as np
import torch
from scipy import signal as sg

from tngpkt.config import Stage3Config
from tngpkt.models3 import StreamModem
from tngpkt.framing import (FRAME_BITS, FRAME_BYTES, text_to_frames, frames_to_bits,
                     parse_frame, frames_to_text)
from tngpkt.waveform import design_lpf, psd_db


# ------------------------------------------------------------------ 모델/WAV
def load_model(path, device):
    # Resolve bundled models from the source/package root, independent of CWD.
    from tngpkt import app_paths
    path = app_paths.resource(path)
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Stage3Config(**ck["config"])
    cfg.device = str(device)
    m = StreamModem(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "16비트 WAV 만 지원한다"
        fs = w.getframerate()
        d = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            d = d.reshape(-1, w.getnchannels())[:, 0]
    return d.astype(np.float64) / 32768.0, float(fs)


def write_wav(path, x, fs, headroom_db=3.0):
    peak = max(np.max(np.abs(x)), 1e-12)
    y = x / peak * (10 ** (-headroom_db / 20.0))
    d = np.clip(np.round(y * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(d.tobytes())
    return len(x) / fs


# ------------------------------------------------------------------ 대역폭 측정
def bw_at_level(x_complex, fs, level_db=-60.0, nfft=32768):
    """연속 스트림의 기저대역 스펙트럼이 level_db 위로 올라오는 폭 [Hz]"""
    _, P = psd_db(x_complex[None, :], fs, nfft)
    i = np.where(P > level_db)[0]
    return (i[-1] - i[0] + 1) * fs / nfft if len(i) else 0.0


def audio_bw_at_level(audio, fs, level_db=-60.0, nfft=32768):
    """오디오(실수) 스펙트럼에서 level_db 위 구간의 폭 [Hz]"""
    w = np.hanning(len(audio)) if len(audio) < nfft else np.hanning(nfft)
    seg = audio[:len(w)] * w
    P = np.abs(np.fft.rfft(seg, nfft)) ** 2
    P = 10 * np.log10(np.maximum(P / P.max(), 1e-18))
    i = np.where(P > level_db)[0]
    return (i[-1] - i[0] + 1) * fs / nfft if len(i) else 0.0


# ------------------------------------------------------------------ 송신
def ramp_envelope(n, ramp_samples):
    """앞뒤에 레이즈드 코사인 램프가 붙은 포락선"""
    e = np.ones(n)
    r = min(int(ramp_samples), n // 2)
    if r > 0:
        t = np.arange(r) / r
        up = 0.5 * (1 - np.cos(np.pi * t))
        e[:r] = up
        e[-r:] = up[::-1]
    return e


@torch.no_grad()
def text_to_baseband(text, model, cfg, dev, safety_lpf=False):
    """텍스트 → 복소 기저대역 연속 스트림 (+ 프레임 목록)"""
    frames = text_to_frames(text)
    bits = frames_to_bits(frames)                                  # (F, 96)
    F = len(frames)
    b = torch.tensor(bits.reshape(1, F * cfg.symbols_per_frame, cfg.bits_per_symbol),
                     dtype=torch.float32, device=dev)
    x = model.tx(b)[0].cpu().numpy()                               # 스트림 전체, LPF 1회

    x = x * ramp_envelope(len(x), cfg.ramp_ms * cfg.fs_base / 1000.0)

    if safety_lpf:      # 안전장치 — 기본은 끈다
        h = design_lpf(cfg.lpf_cutoff_hz, cfg.fs_base, cfg.lpf_taps)
        p = (len(h) - 1) // 2
        x = sg.lfilter(h, 1.0, np.concatenate([x, np.zeros(len(h))]))[p:p + len(x)]

    return x / np.sqrt(np.mean(np.abs(x) ** 2)), frames


def baseband_to_audio(bb, cfg, fs_audio, lead_sec=0.2):
    """복소 기저대역 → 실수 오디오 (중심 1500Hz), 앞뒤 무음 포함"""
    up = int(round(fs_audio / cfg.fs_base))
    assert abs(fs_audio / cfg.fs_base - up) < 1e-9, "오디오 샘플레이트는 기저대역의 정수배여야 한다"
    x = sg.resample_poly(bb, up, 1)
    t = np.arange(len(x)) / fs_audio
    a = np.real(x * np.exp(2j * np.pi * cfg.audio_center_hz * t))
    z = np.zeros(int(lead_sec * fs_audio))
    return np.concatenate([z, a, z])


# ------------------------------------------------------------------ 수신
def audio_to_baseband(audio, fs, cfg):
    """실수 오디오 → 복소 기저대역 (1500Hz 하향변환 후 fs_base 로 데시메이션)"""
    t = np.arange(len(audio)) / fs
    x = 2.0 * audio * np.exp(-2j * np.pi * cfg.audio_center_hz * t)
    down = int(round(fs / cfg.fs_base))
    assert abs(fs / cfg.fs_base - down) < 1e-9, "WAV 샘플레이트가 기저대역의 정수배가 아니다"
    return sg.resample_poly(x, 1, down)      # 데시메이션 필터가 상측 이미지를 제거한다


def detect_signal_span(bb, cfg, rel_db=-20.0):
    """에너지로 신호 구간의 대략적인 시작/끝을 찾는다 (정밀 정렬은 다음 단계가 한다)."""
    p = np.convolve(np.abs(bb) ** 2, np.ones(cfg.sps) / cfg.sps, mode="same")
    thr = p.max() * (10 ** (rel_db / 10.0))
    idx = np.where(p > thr)[0]
    if len(idx) == 0:
        return 0, len(bb)
    return int(idx[0]), int(idx[-1]) + 1


# CRC16/CCITT-FALSE 테이블 (많은 후보를 빠르게 검사하기 위함)
_CRC_TABLE = np.zeros(256, dtype=np.uint16)
for _i in range(256):
    _c = _i << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) & 0xFFFF if (_c & 0x8000) else (_c << 1) & 0xFFFF
    _CRC_TABLE[_i] = _c


def crc16_rows(data: np.ndarray) -> np.ndarray:
    """data: (N, 10) uint8 → (N,) uint16 CRC. 행 전체를 한꺼번에 계산."""
    crc = np.full(len(data), 0xFFFF, dtype=np.uint16)
    for j in range(data.shape[1]):
        idx = ((crc >> 8) ^ data[:, j]).astype(np.uint8)
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[idx]
    return crc


@torch.no_grad()
def decode_baseband(bb, model, cfg, dev, max_extra_frames=2, chunk=1024, verbose=False):
    """
    프레임 시작 위치를 스스로 찾아 복호한다.

    가능한 정렬 0 ~ frame_len-1 을 전부 시도하고, 각 정렬마다 전체 프레임을 복호해
    CRC 통과 개수가 가장 많은 정렬을 고른다. CRC 가 정답을 알려주므로
    사람이 만든 동기 워드나 상관기가 필요 없다.
    """
    L, W, C = cfg.frame_len, cfg.win_len, cfg.ctx_len
    s0, s1 = detect_signal_span(bb, cfg)
    base = s0 - L                                        # 한 프레임 앞에서 시작
    n_frames = int(np.ceil((s1 - base) / L)) + max_extra_frames

    pad = 2 * L + W
    x = np.concatenate([np.zeros(pad, dtype=complex), bb, np.zeros(pad + n_frames * L, dtype=complex)])
    origin = pad + base

    # 모든 (정렬, 프레임) 조합의 창 시작 위치
    offs = np.arange(L)
    fidx = np.arange(n_frames)
    starts = (origin + offs[:, None] + fidx[None, :] * L - C).reshape(-1)     # (L*n_frames,)
    win = x[starts[:, None] + np.arange(W)[None, :]]                          # (N, W)

    # 수신 NN 통과 → 비트
    bits = np.empty((len(win), cfg.bits_per_frame), dtype=np.uint8)
    for i in range(0, len(win), chunk):
        w = torch.tensor(win[i:i + chunk], dtype=torch.complex64, device=dev)
        lg = model.rx(w)                                   # (n, S, k)
        bits[i:i + chunk] = (lg > 0).to(torch.uint8).reshape(len(w), -1).cpu().numpy()

    # 비트 → 바이트 → CRC
    by = np.packbits(bits, axis=1)                         # (N, 12)
    ok = (crc16_rows(by[:, :FRAME_BYTES - 2]) ==
          ((by[:, -2].astype(np.uint16) << 8) | by[:, -1])).reshape(L, n_frames)

    score = ok.sum(axis=1)
    best = int(np.argmax(score))
    if verbose:
        top = np.argsort(score)[::-1][:3]
        print("    정렬 탐색: 최고 {}샘플({}/{}프레임 CRC통과), 다음 후보 {}".format(
            best, int(score[best]), n_frames,
            ", ".join("{}({})".format(int(o), int(score[o])) for o in top[1:])))

    parsed = [parse_frame(by[best * n_frames + i].tobytes()) for i in range(n_frames)]
    text, st = frames_to_text(parsed)
    st.update({"offset": best, "n_windows": int(n_frames), "crc_pass": int(score[best])})
    return text, st


def decode_audio(audio, fs, model, cfg, dev, verbose=False):
    return decode_baseband(audio_to_baseband(audio, fs, cfg), model, cfg, dev, verbose=verbose)


# ------------------------------------------------------------------ 손상 주입 (loopback)
@functools.lru_cache(maxsize=16)
def _band_taps(fs, lo, hi):
    return sg.firwin(511, [lo, hi], fs=fs, pass_zero=False)


def add_band_noise(x, fs, snr_db, band, rng, sig_power=None):
    """폭 2500Hz 대역 기준 SNR 이 되도록 대역제한 잡음을 더한다."""
    taps = _band_taps(float(fs), float(band[0]), float(band[1]))
    n = sg.lfilter(taps, 1.0, rng.standard_normal(len(x) + 1024))[1024:]
    ps = np.mean(x ** 2) if sig_power is None else sig_power
    n *= np.sqrt((ps / (10 ** (snr_db / 10.0))) / max(np.mean(n ** 2), 1e-20))
    return x + n


def shift_frequency(x, fs, df_hz):
    """실수 오디오를 df_hz 만큼 주파수 이동 (해석 신호를 거쳐 처리)"""
    if abs(df_hz) < 1e-9:
        return x
    a = sg.hilbert(x)
    t = np.arange(len(x)) / fs
    return np.real(a * np.exp(2j * np.pi * df_hz * t))


# ------------------------------------------------------------------ 명령들
def cmd_tx(args, model, cfg, dev):
    bb, frames = text_to_baseband(args.text, model, cfg, dev, safety_lpf=args.safety_lpf)
    audio = baseband_to_audio(bb, cfg, args.fs)
    dur = write_wav(args.out, audio, args.fs)

    bw_bb = bw_at_level(bb, cfg.fs_base, cfg.bw_criterion_db)
    bw_au = audio_bw_at_level(audio[int(0.2 * args.fs):], args.fs, cfg.bw_criterion_db)
    verdict = "통과" if max(bw_bb, bw_au) <= cfg.bw_limit_hz else "초과!"

    print("  송신 완료 → {}".format(args.out))
    print("    원문        : {} ({}바이트)".format(
        args.text if len(args.text) < 60 else args.text[:57] + "...",
        len(args.text.encode())))
    print("    프레임      : {}개 (가드 2 + 데이터 {})".format(len(frames), len(frames) - 2))
    print("    길이        : {:.2f}초  ({:.0f}Hz, 16비트)".format(dur, args.fs))
    print("    안전 LPF    : {}".format("적용" if args.safety_lpf else "미적용"))
    print("    -60dB 대역폭: 기저대역 {:.0f}Hz / 오디오 {:.0f}Hz  (한계 {:.0f}Hz) → {}".format(
        bw_bb, bw_au, cfg.bw_limit_hz, verdict))
    return 0


def cmd_rx(args, model, cfg, dev):
    audio, fs = read_wav(args.wav)
    text, st = decode_audio(audio, fs, model, cfg, dev, verbose=True)
    print("  수신 완료 ← {} ({:.2f}초, {:.0f}Hz)".format(args.wav, len(audio) / fs, fs))
    print("    프레임 정렬 : {} 샘플에서 발견".format(st["offset"]))
    print("    CRC         : 통과 {} / 가드 {} / 실패 {}".format(
        st["ok"], st["guard"], st["bad"]))
    if st["missing"]:
        print("    빠진 순번   : {}".format(st["missing"]))
    print("    완전 복원   : {}".format("예" if st["complete"] else "아니오"))
    print("    복원 텍스트 : {}".format(text))
    return 0 if st["complete"] else 1


def cmd_loopback(args, model, cfg, dev):
    rng = np.random.default_rng(args.seed)
    bb, frames = text_to_baseband(args.text, model, cfg, dev, safety_lpf=args.safety_lpf)
    audio = baseband_to_audio(bb, cfg, args.fs)
    ps = float(np.mean(audio[audio != 0] ** 2)) if np.any(audio != 0) else 1.0

    y = shift_frequency(audio, args.fs, args.freq_off)
    y = add_band_noise(y, args.fs, args.snr, cfg.noise_band_hz, rng, sig_power=ps)

    if args.save:
        write_wav(args.save, y, args.fs)
        print("  손상된 오디오 저장 → {}".format(args.save))

    text, st = decode_audio(y, args.fs, model, cfg, dev, verbose=True)
    same = (text == args.text)
    print("  루프백  SNR {:+.1f}dB (2500Hz 기준), 주파수오차 {:+.1f}Hz".format(args.snr, args.freq_off))
    print("    CRC 통과 {} / 가드 {} / 실패 {}   정렬 {}샘플".format(
        st["ok"], st["guard"], st["bad"], st["offset"]))
    print("    원문 : {}".format(args.text))
    print("    복원 : {}".format(text))
    print("    완전 복원: {}".format("예" if same else "아니오"))
    return 0 if same else 1


def main():
    ap = argparse.ArgumentParser(description="3단계 오디오 모뎀 (텍스트 ↔ WAV)")
    ap.add_argument("--model", default="models/stage3.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tx", help="텍스트 → WAV")
    p.add_argument("--text", required=True)
    p.add_argument("--out", default="tx.wav")
    p.add_argument("--fs", type=float, default=8000.0)
    p.add_argument("--safety-lpf", action="store_true", help="스트림 LPF 를 한 번 더 건다(안전장치)")

    p = sub.add_parser("rx", help="WAV → 텍스트")
    p.add_argument("--wav", required=True)

    p = sub.add_parser("loopback", help="WAV 에 잡음/주파수오차를 넣고 복조")
    p.add_argument("--text", required=True)
    p.add_argument("--snr", type=float, default=0.0, help="2500Hz 대역 기준 SNR [dB]")
    p.add_argument("--freq-off", type=float, default=0.0, help="주파수 오차 [Hz]")
    p.add_argument("--fs", type=float, default=8000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="손상된 오디오를 이 경로에 저장")
    p.add_argument("--safety-lpf", action="store_true")

    args = ap.parse_args()
    dev = torch.device(args.device)
    if not os.path.exists(args.model):
        print("모델 파일이 없다: {}  (먼저 python train_stage3.py)".format(args.model))
        return 2
    model, cfg = load_model(args.model, dev)

    return {"tx": cmd_tx, "rx": cmd_rx, "loopback": cmd_loopback}[args.cmd](args, model, cfg, dev)


if __name__ == "__main__":
    sys.exit(main())
