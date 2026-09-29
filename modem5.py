"""
5단계 엔진 — 4단계 AI 모뎀(modem4) 앞뒤에 코스타스 프리앰블/포스트앰블을 붙인다.

AI 변조/복조와 학습된 모델은 건드리지 않는다. 데이터 구간은 modem4 가 만든 그대로다.

    송신  [희생 톤 300ms][코스타스 A 224ms][AI 데이터 + FEC][코스타스 B 224ms]
          데이터 구간 형식 (7단계부터 기본 = 스트리밍):
            스트리밍 v2 link7 — 첫 구간에 길이 2바이트 + 32바이트 구간 + CRC16 (9단계, 기본)
            스트리밍 v1 link7 — 길이 필드 없음 (7 · 8단계, 수신 호환)
            단일 블록   link4 — 메시지 전체 한 블록 (4~6단계 형식, 수신 호환)
          세 형식은 가드 프레임 패턴으로 구별한다 (v1 = 단일 블록 반전, v2 = 절반 반전 → 서로 상관 0 또는 -1).
    수신  코스타스 A 검출 → 시작 위치 + 주파수 오차(±100Hz) 추정
          → 주파수 보정 → modem4 복조(FEC)
          코스타스 B 로 메시지 끝을 알면 프레임 수를 정확히 안다.
          B 를 못 찾으면 A 위치부터 복조를 시도한다.
          A 도 없으면 (옛날 형식 WAV) modem4 방식 그대로 복조한다.

    python modem5.py tx       --text "..." --out wav/m5.wav
    python modem5.py rx       --wav wav/m5.wav
    python modem5.py loopback --text "..." --snr -3 --freq-off 60
"""

import _io_utf8  # noqa: F401
import argparse
import sys

import numpy as np
import torch

from config import PreambleConfig
from modem import (load_model, baseband_to_audio, audio_to_baseband, write_wav, read_wav,
                   add_band_noise, shift_frequency, bw_at_level, audio_bw_at_level, ramp_envelope)
from modem4 import text_to_baseband4, decode_baseband4, detect_span4, _rx_windows
from link4 import GUARD_BITS, BITS_PER_FRAME
from link7 import encode_stream, StreamDecoder, GUARD_STREAM, GUARDS, OFFSETS, frames_for, LEN_BYTES
from preamble import assemble, CostasDetector
from waveform import emission_bw

PAD_S = 0.2        # 데이터 구간을 잘라 modem4 에 넘길 때 앞뒤로 붙이는 무음


# ------------------------------------------------------------------ 송신
@torch.no_grad()
def _frames_to_baseband(frames, guard, model, cfg, dev):
    """[가드] + 데이터 프레임 + [가드] → 송신 NN → 램프 → 전력 정규화 (modem4 와 같은 처리)"""
    fr = np.concatenate([guard[None, :], frames, guard[None, :]], axis=0)
    b = torch.tensor(fr.reshape(1, -1, cfg.bits_per_symbol), dtype=torch.float32, device=dev)
    x = model.tx(b)[0].cpu().numpy()
    x = x * ramp_envelope(len(x), cfg.ramp_ms * cfg.fs_base / 1000.0)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def text_to_baseband5(text, model, cfg, dev, pc=None, fmt="stream"):
    """
    반환: (기저대역, 정보 dict). 구간 경계를 정보에 담는다.
    fmt: 'stream' (스트리밍 v2, 기본) / 'stream1' (7단계 v1, 호환 시험용) / 'block' (4단계 단일 블록)
    """
    pc = pc or PreambleConfig()
    if fmt == "block":
        data, n_data, n_total = text_to_baseband4(text, model, cfg, dev)   # 4단계 그대로
        extra = {"format": "단일 블록"}
    else:
        ver = 1 if fmt == "stream1" else 2
        frames, si = encode_stream(text, version=ver)
        data = _frames_to_baseband(frames, GUARDS["stream2" if ver == 2 else "stream1"], model, cfg, dev)
        n_data, n_total = si["n_data"], si["n_data"] + 2
        extra = {"format": "스트리밍", "version": ver, "n_segments": si["n_segments"], "payload": si["payload"]}
    x, info = assemble(pc, cfg.fs_base, data)
    info.update({"n_data": n_data, "n_total": n_total}, **extra)
    return x, info


# ------------------------------------------------------------------ 수신
def _shift(bb, df, fs):
    t = np.arange(len(bb)) / fs
    return bb * np.exp(-2j * np.pi * df * t)


def _decode_block(bb, a_end, b_start, df, model, cfg, dev):
    """단일 블록 형식 (4단계): A 끝 ~ B 시작(또는 버퍼 끝) 을 주파수 보정해 modem4 로 복조한다."""
    L = cfg.frame_len
    y = _shift(bb, df, cfg.fs_base)
    seg = y[a_end:b_start] if b_start is not None else y[a_end:]
    z = np.zeros(int(PAD_S * cfg.fs_base), dtype=complex)
    bbd = np.concatenate([z, seg, z])
    hint = None
    if b_start is not None:
        n_total = int(round(len(seg) / L))       # B 덕분에 프레임 수를 정확히 안다
        hint = n_total - 2
    # 코스타스 A 끝 = 데이터(가드 프레임) 시작 → 첫 데이터 프레임은 그 한 프레임 뒤.
    # 위치를 이미 아니 정렬 탐색은 그 둘레만 좁게 한다 (실패하면 넓은 탐색으로 넘어감).
    first = len(z) + L
    # A 가 위치를 알려줬으니 넓은 탐색 재시도는 하지 않는다. A 자체가 틀린 경우는
    # decode_baseband5 의 마지막 호환 경로(버퍼 전체 넓은 탐색)가 다시 확인한다.
    text, info = decode_baseband4(bbd, model, cfg, dev, n_data_hint=hint, data_start_hint=first,
                                  wide_fallback=False)
    if text is None and hint is not None:        # B 를 잘못 짚었을 수도 있다 → 프레임 수 가설 탐색
        text, info = decode_baseband4(bbd, model, cfg, dev, data_start_hint=first, wide_fallback=False)
    info["pad"] = len(z)                         # 시각화가 같은 구간을 다시 만들 수 있게
    return text, info


def _llr_at(bbd, first, frames, offs, eps, model, cfg, dev):
    """
    프레임 번호 frames (첫 데이터 프레임 = 0, 가드 = -1) × 정렬 가설 offs 의 LLR (H, k, 96).
    eps: 클럭 오차 (A~B 간격으로 추정) — 프레임 f 의 위치를 f*L*(1+eps) 로 늘인다.
    """
    L, W, C = cfg.frame_len, cfg.win_len, cfg.ctx_len
    offs = np.asarray(offs)
    f = np.asarray(list(frames))
    pad = W + 2 * L + int(np.abs(offs).max()) + 8
    x = np.concatenate([np.zeros(pad, dtype=complex), bbd, np.zeros(pad + 2 * L, dtype=complex)])
    base = first + np.round(f * L * (1.0 + eps)).astype(np.int64)
    starts = (pad + base[None, :] + offs[:, None] - C).reshape(-1)
    starts = np.clip(starts, 0, len(x) - W)
    win = x[starts[:, None] + np.arange(W)[None, :]]
    return _rx_windows(win, model, dev).reshape(len(offs), len(f), BITS_PER_FRAME)


def _detect_format(bbd, first, n_data, eps, model, cfg, dev):
    """가드 프레임 LLR 과 세 가드 패턴의 상관 → ('stream2' | 'stream1' | 'block', 상관 dict)"""
    idx = [-1] + ([n_data] if n_data is not None else [])
    g = _llr_at(bbd, first, idx, OFFSETS, eps, model, cfg, dev)
    corr = {k: float((g * (2.0 * v - 1.0)).sum(axis=(1, 2)).max()) for k, v in GUARDS.items()}
    return max(corr, key=corr.get), corr


def _decode_stream(bbd, first, n_hyps, eps, model, cfg, dev, pad, version=2, b_known=True):
    """
    스트리밍 형식 복조. 반환: (완전 텍스트 또는 None, 정보 dict)
    v2 · B 없음: 첫 구간의 길이로 프레임 수를 정해 다시 복조한다 (B 가 있으면 B 가 정한 프레임 수가 우선).
    """
    best = None
    n_hyps = list(n_hyps)
    tried = set()
    nd_len = None                                        # v2 길이 필드가 알려 준 프레임 수
    i = 0
    while i < len(n_hyps):
        nd = n_hyps[i]
        i += 1
        if nd in tried:
            continue
        tried.add(nd)
        try:
            dec = StreamDecoder(nd)
        except ValueError:
            continue
        llr = _llr_at(bbd, first, range(nd), OFFSETS, eps, model, cfg, dev)
        dec.feed(llr)
        dec.finish()
        text, partial, si = dec.result(version=version)
        cand = (si["segments_ok"], text, partial, si, llr, nd)
        if version == 2 and not b_known and si["length"] is not None:
            nd_len = frames_for(LEN_BYTES + si["length"])
            # 신호 소실 뒤 복조: 에너지로 잡은 길이는 끊긴 곳까지라 짧다 → 길이 필드의 프레임 수로 다시 (상한 = 최대 메시지)
            if nd_len != nd and nd_len not in tried and nd_len <= 700:
                n_hyps.insert(i, nd_len)                 # 길이가 알려 준 프레임 수를 바로 다음에 시도
        # B 없음: 같은 통과 수면 길이 필드가 알려 준 프레임 수 쪽 (끝까지 본 결과) 을 고른다
        if best is None or text is not None or cand[0] > best[0] or                 (cand[0] == best[0] and nd_len is not None and nd == nd_len):
            best = cand
            dec_best = dec
        if text is not None:
            break
    if best is None:
        return None, {"crc_ok": False, "reason": "프레임 수 가설 없음", "path": "스트리밍", "format": "스트리밍"}
    nok, text, partial, si, llr, nd = best
    # 시각화용: 가장 많이 고른 정렬의 LLR (통과 구간 기준)
    offs_ok = [o for o, k in zip(si["offsets"], si["seg_ok"]) if k] or [0]
    o = max(set(offs_ok), key=offs_ok.count)
    h = int(np.argmin(np.abs(np.asarray(OFFSETS) - o)))
    info = {"crc_ok": text is not None, "format": "스트리밍", "version": version, "path": "스트리밍 추적",
            "length": si.get("length"), "declared_segments": si.get("declared_segments"),
            "stopped": bool(si.get("stopped")) and b_known,       # 송신 중지 판정은 B 로 끝을 알 때만
            "reason": "" if text is not None else "구간 {}/{} 통과".format(si["segments_ok"], si["segments_used"]),
            "partial": partial, "segments_ok": si["segments_ok"], "n_segments": si["n_segments"],
            "segments_used": si["segments_used"], "seg_ok": si["seg_ok"], "seg_offsets": si["offsets"],
            "relocks": si["relocks"], "drift_ppm": eps * 1e6, "n_data": nd, "n_total_est": nd + 2,
            "offset": int(OFFSETS[h]), "llr": llr[h].copy(), "frame0": first + int(OFFSETS[h]), "pad": pad,
            "segments": [dict(r) for r in dec_best.segments]}          # UI 표시용 구간별 상세 (복조 결과 그대로)
    return text, info


def _decode_segment(bb, a_end, b_start, df, model, cfg, dev):
    """A 끝 ~ B 시작(또는 버퍼 끝): 가드 패턴으로 형식을 가려 복조한다. 실패하면 다른 형식도 시도한다."""
    L = cfg.frame_len
    y = _shift(bb, df, cfg.fs_base)
    seg = y[a_end:b_start] if b_start is not None else y[a_end:]
    z = np.zeros(int(PAD_S * cfg.fs_base), dtype=complex)
    bbd = np.concatenate([z, seg, z])
    first = len(z) + L
    eps, n_data = 0.0, None
    if b_start is not None:
        n_total = int(round(len(seg) / L))
        n_data = n_total - 2
        eps = (len(seg) - n_total * L) / float(n_total * L)     # 클럭 오차 (A~B 간격)
        n_hyps = [n_data]
    else:
        _, _, n_est = detect_span4(bbd, cfg)
        n_hyps = [h for h in (n_est - 1, n_est - 2, n_est, n_est - 3) if h >= 1]
    fmt, corr = _detect_format(bbd, first, n_data, eps, model, cfg, dev)
    tried = []
    order = [fmt] + [f for f in ("stream2", "stream1", "block") if f != fmt]
    for f in order:
        if f.startswith("stream"):
            text, info = _decode_stream(bbd, first, n_hyps, eps, model, cfg, dev, len(z),
                                        version=2 if f == "stream2" else 1, b_known=b_start is not None)
            got = text is not None or info.get("segments_ok", 0) > 0
        else:
            text, info = _decode_block(bb, a_end, b_start, df, model, cfg, dev)
            info["format"] = "단일 블록"
            got = text is not None
        info["guard_corr"] = corr
        info["format_detected"] = {"stream2": "스트리밍 v2", "stream1": "스트리밍 v1", "block": "단일 블록"}[fmt]
        tried.append((text, info))
        if got:
            return text, info
    return tried[0]


BACK_THR = 0.08          # 역방향 동기: B 가 잡힌 뒤 A 를 다시 찾는 낮은 문턱 (프레임 격자 · 주파수 일치로 가짜를 거른다)
BACK_GRID = 16           # A 끝 ~ B 시작 간격이 프레임 길이의 정수배에서 벗어나도 되는 샘플 (클럭 오차 300 ppm · 120 s ≈ 72 → 짧게 제한)
BACK_DF = 3.0            # A · B 주파수 오차 차이 허용 [Hz]


def find_a_backward(det, bb, b, cfg, pc):
    """
    역방향 동기 (송신 비용 0): A 를 놓친 메시지를 B 에서 거꾸로 찾는다.
    B 앞 최대 길이 창에서 낮은 문턱으로 A 후보를 다시 찾고, (1) B 와 주파수 오차가 같고 (2) A 끝 ~ B 시작 간격이
    프레임 길이의 정수배(가드 2 + 데이터 ≥ 1)인 후보 중 ρ² 가 가장 큰 것을 고른다. 없으면 None.
    """
    L = cfg.frame_len
    lo = max(0, b["start"] - int(pc.max_message_s * cfg.fs_base) - det.n)
    seg = bb[lo:b["start"] + det.n]
    best = None
    for d in det.detect(seg, threshold=BACK_THR):
        if d["kind"] != "A" or abs(d["df"] - b["df"]) > BACK_DF:
            continue
        gap = b["start"] - (lo + d["start"] + det.n)
        n = int(round(gap / L))
        if n < 3 or abs(gap - n * L) > BACK_GRID:
            continue
        if best is None or d["rho"] > best["rho"]:
            best = dict(d, start=lo + d["start"], back=True)
    return best


def decode_baseband5(bb, model, cfg, dev, pc=None, det=None):
    """
    반환: 메시지 목록 [{'text','ok','mode','df','a','b', ...}]
    한 버퍼에 메시지가 여러 개 있어도 된다.
    """
    pc = pc or PreambleConfig()
    det = det or CostasDetector(pc, cfg.fs_base)
    dets = det.detect(bb)
    As = [d for d in dets if d["kind"] == "A"]
    Bs = [d for d in dets if d["kind"] == "B"]
    out = []
    max_len = int(pc.max_message_s * cfg.fs_base)
    min_len = 3 * cfg.frame_len                   # 가드 2 + 데이터 1 프레임

    for a in As:
        a_end = a["start"] + det.n
        b = next((d for d in Bs if a_end + min_len <= d["start"] <= a_end + max_len), None)
        df = a["df"]
        if b is not None and abs(b["df"] - a["df"]) < 3.0:
            df = 0.5 * (a["df"] + b["df"])         # 두 추정의 평균
        text, info = _decode_segment(bb, a_end, None if b is None else b["start"],
                                     df, model, cfg, dev)
        out.append({"text": text, "ok": text is not None,
                    "format": info.get("format"), "partial": info.get("partial"),
                    "mode": "코스타스 A+B" if b is not None else "코스타스 A (B 없음)",
                    "df": df, "a": a, "b": b, "info": info,
                    "seg": (a_end, None if b is None else b["start"])})

    # 역방향 동기: 어느 A 와도 짝이 안 된 B → B 에서 거꾸로 A 를 찾아 복조
    used = [m["b"]["start"] for m in out if m.get("b") is not None]
    for b in Bs:
        if any(abs(b["start"] - u) < det.n for u in used):
            continue
        a = find_a_backward(det, bb, b, cfg, pc)
        if a is None:
            continue
        a_end = a["start"] + det.n
        df = 0.5 * (a["df"] + b["df"])
        text, info = _decode_segment(bb, a_end, b["start"], df, model, cfg, dev)
        out.append({"text": text, "ok": text is not None, "format": info.get("format"), "partial": info.get("partial"),
                    "mode": "코스타스 B 역방향 (A ρ² {:.2f})".format(a["rho"]), "df": df, "a": a, "b": b,
                    "info": info, "seg": (a_end, b["start"])})

    if not any(m["ok"] for m in out):
        # 옛날 형식 WAV (프리앰블 없음) — 또는 A 를 잘못 짚은 경우: 4단계 방식 그대로
        text, info = decode_baseband4(bb, model, cfg, dev)
        if text is not None or not out:
            out = [{"text": text, "ok": text is not None, "mode": "호환 (프리앰블 없음)",
                    "df": 0.0, "a": None, "b": None, "info": info}] + \
                  ([] if text is not None else out)
    return out


def decode_audio5(audio, fs, model, cfg, dev, pc=None, det=None):
    return decode_baseband5(audio_to_baseband(audio, fs, cfg), model, cfg, dev, pc, det)


def first_text(msgs):
    for m in msgs:
        if m["ok"]:
            return m["text"], m
    return None, (msgs[0] if msgs else {})


# ------------------------------------------------------------------ 명령들
def cmd_tx(args, model, cfg, dev):
    bb, info = text_to_baseband5(args.text, model, cfg, dev, fmt=args.fmt)
    audio = baseband_to_audio(bb, cfg, args.fs)
    dur = write_wav(args.out, audio, args.fs)
    fs = cfg.fs_base
    b1 = emission_bw(bb, fs, cfg.bw_criterion_db)
    b2 = emission_bw(audio, args.fs, cfg.bw_criterion_db)
    oh = (len(bb) - info["data"]) / fs
    print("  송신 완료 → {}".format(args.out))
    print("    구간         : 톤+코스타스A {:.0f}ms | 데이터 {:.0f}ms ({}프레임) | 코스타스B {:.0f}ms".format(
        1000 * info["pre"] / fs, 1000 * info["data"] / fs, info["n_total"], 1000 * info["post"] / fs))
    print("    길이         : {:.2f}초 (추가 오버헤드 {:.0f}ms)".format(dur, 1000 * oh))
    print("    -60dB 대역폭 : 기저 {:.0f}Hz / 오디오 {:.0f}Hz → {}".format(
        b1, b2, "통과" if max(b1, b2) <= cfg.bw_limit_hz else "초과!"))
    return 0


def _print_msgs(msgs):
    for m in msgs:
        a = m["a"]
        extra = "" if a is None else " · A rho²={:.2f} @ {:.3f}s".format(a["rho"], a["start"] / 2000.0)
        inf = m["info"]
        fmt = " · {}".format(m.get("format") or inf.get("format", "단일 블록"))
        if inf.get("n_segments"):
            fmt += " · 구간 {}/{} · 드리프트 {:+.0f}ppm · 재탐색 {}회".format(
                inf["segments_ok"], inf["segments_used"], inf.get("drift_ppm", 0.0), inf.get("relocks", 0))
        print("    [{}{}] 주파수오차 {:+.1f}Hz{} → {}".format(
            m["mode"], fmt, m["df"], extra, m["text"] if m["ok"] else "(실패: {}){}".format(
                inf.get("reason"), " 부분: " + m["partial"] if m.get("partial") else "")))


def cmd_rx(args, model, cfg, dev):
    audio, fs = read_wav(args.wav)
    msgs = decode_audio5(audio, fs, model, cfg, dev)
    print("  수신 ← {} ({:.2f}초)".format(args.wav, len(audio) / fs))
    _print_msgs(msgs)
    return 0 if any(m["ok"] for m in msgs) else 1


def cmd_loopback(args, model, cfg, dev):
    rng = np.random.default_rng(args.seed)
    bb, info = text_to_baseband5(args.text, model, cfg, dev, fmt=args.fmt)
    audio = baseband_to_audio(bb, cfg, args.fs)
    ps = float(np.mean(audio[audio != 0] ** 2))
    y = shift_frequency(audio, args.fs, args.freq_off)
    if args.ppm:                                  # 송수신 사운드카드 클럭 오차 모사
        from scipy import signal as sg
        y = sg.resample(y, int(round(len(y) * (1.0 + args.ppm * 1e-6))))
    y = add_band_noise(y, args.fs, args.snr, cfg.noise_band_hz, rng, sig_power=ps)
    print("  송신 형식 {} · 데이터 {}프레임 · 송신 {:.2f}초".format(
        info["format"], info["n_data"], len(bb) / cfg.fs_base))
    if args.save:
        write_wav(args.save, y, args.fs)
    msgs = decode_audio5(y, args.fs, model, cfg, dev)
    text, _ = first_text(msgs)
    print("  루프백  SNR {:+.1f}dB, 주파수오차 {:+.1f}Hz".format(args.snr, args.freq_off))
    _print_msgs(msgs)
    print("    완전 복원: {}".format("예" if text == args.text else "아니오"))
    return 0 if text == args.text else 1


def main():
    ap = argparse.ArgumentParser(description="5단계 엔진 (코스타스 프리앰블 + AI 모뎀 + FEC)")
    ap.add_argument("--model", default="models/stage3.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tx"); p.add_argument("--text", required=True)
    p.add_argument("--out", default="wav/m5_tx.wav"); p.add_argument("--fs", type=float, default=8000.0)
    p.add_argument("--fmt", choices=["stream", "stream1", "block"], default="stream")
    p = sub.add_parser("rx"); p.add_argument("--wav", required=True)
    p = sub.add_parser("loopback"); p.add_argument("--text", required=True)
    p.add_argument("--snr", type=float, default=0.0); p.add_argument("--freq-off", type=float, default=0.0)
    p.add_argument("--fs", type=float, default=8000.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None)
    p.add_argument("--fmt", choices=["stream", "stream1", "block"], default="stream")
    p.add_argument("--ppm", type=float, default=0.0, help="클럭 오차 [ppm]")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model, cfg = load_model(args.model, dev)
    return {"tx": cmd_tx, "rx": cmd_rx, "loopback": cmd_loopback}[args.cmd](args, model, cfg, dev)


if __name__ == "__main__":
    sys.exit(main())
