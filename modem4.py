"""
4단계 AI 모뎀 — 3단계 신경망 + FEC/인터리빙 링크 계층.

신경망은 **3단계 모델(models/stage3.pt)을 그대로 쓴다.** 재학습이 필요 없다:
수신 NN 은 BCE 로 학습했으므로 출력 로짓이 곧 LLR 이고, 그것을 soft 비터비에
그대로 넣으면 된다. FEC 는 순수하게 링크 계층에서 얹히는 것이다.

    python modem4.py tx       --text "..." --out wav/f4.wav
    python modem4.py rx       --wav wav/f4.wav
    python modem4.py loopback --text "..." --snr -3 --freq-off 7
"""

import _io_utf8  # noqa: F401
import argparse
import os
import sys

import numpy as np
import torch

from modem import (load_model, baseband_to_audio, audio_to_baseband, detect_signal_span,
                   ramp_envelope, write_wav, read_wav, add_band_noise, shift_frequency,
                   bw_at_level, audio_bw_at_level)
from link4 import encode_text, search_and_decode, GUARD_BITS, BITS_PER_FRAME
from waveform import design_lpf
from scipy import signal as sg


# ------------------------------------------------------------------ 송신
@torch.no_grad()
def text_to_baseband4(text, model, cfg, dev, safety_lpf=False):
    data, n_data = encode_text(text)                       # (n_data, 96)
    frames = np.concatenate([GUARD_BITS[None, :], data, GUARD_BITS[None, :]], axis=0)
    n_total = len(frames)

    b = torch.tensor(frames.reshape(1, n_total * cfg.symbols_per_frame, cfg.bits_per_symbol),
                     dtype=torch.float32, device=dev)
    x = model.tx(b)[0].cpu().numpy()
    x = x * ramp_envelope(len(x), cfg.ramp_ms * cfg.fs_base / 1000.0)
    if safety_lpf:
        h = design_lpf(cfg.lpf_cutoff_hz, cfg.fs_base, cfg.lpf_taps)
        p = (len(h) - 1) // 2
        x = sg.lfilter(h, 1.0, np.concatenate([x, np.zeros(len(h))]))[p:p + len(x)]
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), n_data, n_total


# ------------------------------------------------------------------ 수신
def detect_span4(bb, cfg):
    """
    잡음 속에서도 버스트의 시작/끝을 튼튼하게 찾는다.

    3단계의 검출기는 최대치 대비 -20dB 로 잘랐는데, 잡음이 세면 잡음까지 신호로 잡혀
    시작점이 앞으로 밀렸다. 여기서는 앞뒤 무음 구간에서 잡음 바닥을 따로 추정하고
    (잡음바닥 + 최고치)의 중간 지점을 문턱으로 쓴다.
    """
    # 1) 고정 LPF 로 대역 밖 잡음을 먼저 버린다 (2000Hz → 400Hz, 약 7dB 이득)
    h = design_lpf(cfg.lpf_cutoff_hz, cfg.fs_base, cfg.lpf_taps)
    d = (len(h) - 1) // 2
    z = sg.lfilter(h, 1.0, np.concatenate([bb, np.zeros(len(h))]))[d:d + len(bb)]
    # 2) 프레임 길이만큼 평활해 잡음 분산을 더 줄인다
    p = np.convolve(np.abs(z) ** 2, np.ones(cfg.frame_len) / cfg.frame_len, mode="same")
    s = np.sort(p)
    floor = np.median(s[:max(1, len(s) // 5)])            # 하위 20% = 잡음 바닥
    peak = np.median(s[-max(1, len(s) // 5):])            # 상위 20% = 신호 구간
    thr = floor + 0.5 * (peak - floor)
    idx = np.where(p > thr)[0]
    if len(idx) < cfg.frame_len:
        return 0, len(bb), 3
    s0, s1 = int(idx[0]), int(idx[-1]) + 1
    return s0, s1, max(3, int(round((s1 - s0) / cfg.frame_len)))


@torch.no_grad()
def _rx_windows(win, model, dev, chunk=1024):
    """수신 창들을 수신 NN 에 통과시켜 비트 LLR 을 얻는다."""
    out = np.empty((len(win), BITS_PER_FRAME), dtype=np.float32)
    for i in range(0, len(win), chunk):
        w = torch.tensor(win[i:i + chunk], dtype=torch.complex64, device=dev)
        out[i:i + chunk] = model.rx(w).reshape(len(w), -1).cpu().numpy()
    return out


def llr_grid_at(bb, model, cfg, dev, frame0, radius, n_frames):
    """
    좁은 탐색용 격자: 첫 데이터 프레임이 frame0 근처(±radius 샘플)에 있다는 것을 알 때.
    (코스타스 A 로 시작 위치를 이미 찾은 경우 — 넓은 탐색의 384×(n+4) 창 대신 (2r+1)×n 창)
    반환: (2r+1, n_frames, 96)
    """
    L, W, C = cfg.frame_len, cfg.win_len, cfg.ctx_len
    pad = W + L + radius
    x = np.concatenate([np.zeros(pad, dtype=complex), bb,
                        np.zeros(pad + (n_frames + 1) * L, dtype=complex)])
    offs = np.arange(-radius, radius + 1)
    starts = (pad + frame0 + offs[:, None] + np.arange(n_frames)[None, :] * L - C).reshape(-1)
    win = x[starts[:, None] + np.arange(W)[None, :]]
    return _rx_windows(win, model, dev).reshape(len(offs), n_frames, BITS_PER_FRAME)


@torch.no_grad()
def llr_grid(bb, model, cfg, dev, chunk=1024):
    grid, n_total, _ = _llr_grid_full(bb, model, cfg, dev, chunk)
    return grid, n_total


@torch.no_grad()
def _llr_grid_full(bb, model, cfg, dev, chunk=1024):
    """
    (정렬 후보, 프레임 자리, 96) LLR 격자를 만든다.
    수신 NN 로짓을 스케일 없이 그대로 쓴다 — BCE 학습 덕분에 곧 LLR 이다.

    탐색은 검출된 버스트 시작점에 고정하고, 그 둘레로 ±프레임길이/2 만큼만 민다.
    이렇게 하면 '몇 번째 자리가 첫 프레임인가' 하는 모호함이 사라져 후보가 L 개로 준다.
    """
    L, W, C = cfg.frame_len, cfg.win_len, cfg.ctx_len
    s0, s1, n_total = detect_span4(bb, cfg)
    n_slots = n_total + 4

    pad = 2 * L + W
    x = np.concatenate([np.zeros(pad, dtype=complex), bb,
                        np.zeros(pad + (n_slots + 2) * L, dtype=complex)])
    offs = np.arange(-(L // 2), L - (L // 2))                  # [-L/2, L/2)
    starts = (pad + s0 + offs[:, None] + np.arange(n_slots)[None, :] * L - C).reshape(-1)
    win = x[starts[:, None] + np.arange(W)[None, :]]

    out = _rx_windows(win, model, dev, chunk)
    return out.reshape(L, n_slots, BITS_PER_FRAME), n_total, s0


def decode_baseband4(bb, model, cfg, dev, n_data_hint=None, data_start_hint=None, radius=8,
                     wide_fallback=True):
    """
    data_start_hint(첫 데이터 프레임의 샘플 위치)를 주면 먼저 그 둘레 ±radius 만 좁게 탐색한다.
    좁은 탐색이 실패하면 원래의 넓은 탐색으로 넘어간다 (결과가 나빠지지 않도록).
    info 에 'frame0' (첫 데이터 프레임 위치) 와 'path' (좁은/넓은 탐색) 를 남긴다.
    """
    L = cfg.frame_len
    if data_start_hint is not None:
        if n_data_hint is not None:
            hyps_n = [n_data_hint]
        else:
            _, _, n_est = detect_span4(bb, cfg)
            hyps_n = [h for h in (n_est - 1, n_est - 2, n_est, n_est - 3) if h >= 1]
        if hyps_n:
            g = llr_grid_at(bb, model, cfg, dev, data_start_hint, radius, max(hyps_n))
            for nd in hyps_n:
                text, info = search_and_decode(g[:, :nd, :], nd, start_slots=(0,))
                if text is not None:
                    info.update({"n_data": nd, "n_total_est": nd + 2, "path": "좁은 탐색",
                                 "frame0": data_start_hint + info["offset"] - radius})
                    return text, info
        if not wide_fallback:
            return None, {"crc_ok": False, "reason": "좁은 탐색 실패 (코스타스 기준 구간)",
                          "path": "좁은 탐색"}

    grid, n_total, s0 = _llr_grid_full(bb, model, cfg, dev)
    n_slots = grid.shape[1]
    # 첫 자리는 가드 프레임이므로 데이터는 자리 1 부터 시작한다.
    # 검출기는 평활 창 때문에 버스트를 한 프레임쯤 짧게 잡는 편향이 있다.
    # 그래서 n_total-1 을 첫 가설로 두고 그 둘레를 훑는다.
    hyps = ([n_data_hint] if n_data_hint is not None
            else [n_total - 1, n_total - 2, n_total, n_total - 3])
    for nd in hyps:
        if nd is None or nd < 1 or 3 + nd > n_slots + 2:
            continue
        text, info = search_and_decode(grid, nd, start_slots=(0, 1, 2))
        if text is not None:
            info.update({"n_data": nd, "n_total_est": n_total, "path": "넓은 탐색",
                         "frame0": s0 + (info["offset"] - L // 2) + info["start_slot"] * L})
            return text, info
    return None, {"crc_ok": False, "reason": "모든 프레임 수 가설 실패", "n_total_est": n_total}


def decode_audio4(audio, fs, model, cfg, dev, n_data_hint=None):
    return decode_baseband4(audio_to_baseband(audio, fs, cfg), model, cfg, dev, n_data_hint)


# ------------------------------------------------------------------ 명령들
def cmd_tx(args, model, cfg, dev):
    bb, n_data, n_total = text_to_baseband4(args.text, model, cfg, dev, args.safety_lpf)
    audio = baseband_to_audio(bb, cfg, args.fs)
    dur = write_wav(args.out, audio, args.fs)
    b1 = bw_at_level(bb, cfg.fs_base, cfg.bw_criterion_db)
    b2 = audio_bw_at_level(audio[int(0.2 * args.fs):], args.fs, cfg.bw_criterion_db)
    print("  송신 완료 → {}".format(args.out))
    print("    원문         : {} ({}바이트)".format(args.text[:50], len(args.text.encode())))
    print("    프레임       : 가드 2 + 데이터 {} = {}개".format(n_data, n_total))
    print("    FEC          : K=7 부호율 1/2 + 전체 인터리빙")
    print("    길이         : {:.2f}초 → 순 처리율 {:.0f} bps".format(
        dur, len(args.text.encode()) * 8 / (n_total * cfg.frame_ms / 1000)))
    print("    -60dB 대역폭 : 기저 {:.0f}Hz / 오디오 {:.0f}Hz → {}".format(
        b1, b2, "통과" if max(b1, b2) <= cfg.bw_limit_hz else "초과!"))
    return 0


def cmd_rx(args, model, cfg, dev):
    audio, fs = read_wav(args.wav)
    text, info = decode_audio4(audio, fs, model, cfg, dev)
    print("  수신 완료 ← {} ({:.2f}초)".format(args.wav, len(audio) / fs))
    print("    추정 프레임 수: {}".format(info.get("n_total_est")))
    if text is None:
        print("    복호 실패: {}".format(info.get("reason")))
        return 1
    print("    정렬 {}샘플 / 시작자리 {} / 후보순위 {} / metric 여유 {:.0f}".format(
        info["offset"], info["start_slot"], info["rank"], info.get("metric_margin", 0)))
    print("    복원 텍스트 : {}".format(text))
    return 0


def cmd_loopback(args, model, cfg, dev):
    rng = np.random.default_rng(args.seed)
    bb, n_data, n_total = text_to_baseband4(args.text, model, cfg, dev, args.safety_lpf)
    audio = baseband_to_audio(bb, cfg, args.fs)
    ps = float(np.mean(audio[audio != 0] ** 2))
    y = shift_frequency(audio, args.fs, args.freq_off)
    y = add_band_noise(y, args.fs, args.snr, cfg.noise_band_hz, rng, sig_power=ps)
    if args.save:
        write_wav(args.save, y, args.fs)
        print("  손상된 오디오 저장 → {}".format(args.save))
    text, info = decode_audio4(y, args.fs, model, cfg, dev)
    ok = (text == args.text)
    print("  루프백(FEC)  SNR {:+.1f}dB, 주파수오차 {:+.1f}Hz".format(args.snr, args.freq_off))
    print("    원문 : {}".format(args.text))
    print("    복원 : {}".format(text if text is not None else "(복호 실패: {})".format(
        info.get("reason"))))
    print("    완전 복원: {}".format("예" if ok else "아니오"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="4단계 AI 모뎀 (FEC + 인터리빙)")
    ap.add_argument("--model", default="models/stage3.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tx"); p.add_argument("--text", required=True)
    p.add_argument("--out", default="wav/f4_tx.wav"); p.add_argument("--fs", type=float, default=8000.0)
    p.add_argument("--safety-lpf", action="store_true")
    p = sub.add_parser("rx"); p.add_argument("--wav", required=True)
    p = sub.add_parser("loopback"); p.add_argument("--text", required=True)
    p.add_argument("--snr", type=float, default=0.0); p.add_argument("--freq-off", type=float, default=0.0)
    p.add_argument("--fs", type=float, default=8000.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None); p.add_argument("--safety-lpf", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if not os.path.exists(args.model):
        print("모델이 없다: {}".format(args.model))
        return 2
    model, cfg = load_model(args.model, dev)
    return {"tx": cmd_tx, "rx": cmd_rx, "loopback": cmd_loopback}[args.cmd](args, model, cfg, dev)


if __name__ == "__main__":
    sys.exit(main())
