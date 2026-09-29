"""
[v1 보관본 — WAV 열기에서 v1 파일 복조 전용 (2026-09-25 v2 적용 때 복사, 실시간 수신에는 안 씀)]
TNG5 앱 연결 — 송신 오디오 · 실시간 수신 · 복조 (엔진은 tng5.py 그대로, TNG44 코드는 건드리지 않는다).

    Tng5Engine     모델 stage3_strong.pt, 동기 템플릿, 송신 오디오, 구간 복호
    Tx5Audio       stream_ui.TxAudio 와 같은 모양 (송신 패널 · 송신 모니터 · 송신 해부가 그대로 쓴다)
    Tng5Receiver   realtime_rx.RealtimeReceiver 와 같은 콜백 · 사건 (on_status, on_marker, on_decoded, on_live)
    MultiRx        TNG44 · TNG5 수신기 묶음 (수신 모드: 전체 동시 / 선택 모드만)

실시간 수신 (Tng5Receiver)
  · 탐색: A 묶음 (톤 + A×4 + 가드) · B 묶음 (가드 + B×4) 템플릿 — tng5.detect (문턱 · 봉우리 · SNR 일치 검사 그대로)
  · A 뒤: 도착하는 프레임마다 LLR (정렬 가설 7개) → 구간 부호 비트가 다 도착하면 그 구간 복호 → 첫 CRC 통과 때 표시
    (인터리버 최대 약 9.6 s 때문에 첫 구간 표시는 A 뒤 약 10 s — 프로토콜 특성)
  · 끝내는 순서: 코스타스 B > 길이 필드 (첫 구간 글자 수 → 프레임 수) > 신호 소실 (|LLR| · 대역 에너지 · 구간 실패)
  · 가짜 B: 길이를 알면 예상 끝 ±0.5 s 밖 B 는 (이르면 |LLR| 확인 뒤) 무시, 모르면 B 뒤 0.6 s 평균 |LLR| 이 잡음 수준일 때만 인정
  · 역방향 동기: A 를 놓치고 B 만 잡히면 링버퍼를 되짚어 A 를 찾아 복조
"""
import math
import threading
import time

import numpy as np
import torch

from tngpkt import tng5_v1 as tng5
from tngpkt.engine import DecodeResult, load_model, resample, MODEM_FS, estimate_snr_2500
from tngpkt.modem import audio_to_baseband, baseband_to_audio
from tngpkt.realtime_rx import RingBuffer

NAME = tng5.NAME
BB_FS = tng5.FS
L = 384                                  # 프레임 (기저대역 샘플) = 96비트 = 192 ms
SC = tng5.SEG_CODED
OFFS = (-6, -4, -2, 0, 2, 4, 6)          # 정렬 가설 (기저대역 샘플)
NSEG_MAX = 40                            # 최대 구간 (첫 구간 24자 + 39 × 27자 = 1077자, 송신 약 3.6분)
N_TONE = int(round(tng5.PC.tone_ms * 1e-3 * BB_FS))
N_SYNC = tng5.SYNC_REP * 7 * int(round(tng5.PC.symbol_ms * 1e-3 * BB_FS))     # A×4 · B×4 길이 (2688)
# 신호 소실: TNG5 는 -12 dB 이하에서 |LLR| 이 잡음과 겹친다 (측정: 잡음 2.5 s 평균 최대 0.40,
# 신호 -12 dB Moderate 2.5 s 평균 최소 0.28) → TNG44 (2.5 s) 보다 길게 5 s, 대역 에너지 · 구간 실패 조건 함께
LOSS_T = 26
LOSS_LLR = 0.45
LOSS_E_DB = 1.5
B_LLR_S = 0.6
B_DF_HZ = 3.0                             # A · B 주파수 일치 허용 (추정 오차 RMS 약 0.6 Hz, 측정)
RIVAL_S = 2 * 7 * 0.048 + 0.1            # 경쟁 후보 대기: 후보 뒤 2벌 (0.672 s) + 0.1 s (전: 3벌 1.108 s. 1벌은 반향 · 클럭에서 가짜 시작 1건 → 2벌, 원격 세션 4번)


def seg_count(nchars):
    return len(tng5.segments("A" * nchars))


def frames_for_segs(nseg):
    return int(math.ceil(nseg * SC / tng5.BITS_PER_FRAME))


class Plan5:
    """stream_ui.StreamPlan 과 같은 속성 (타임라인 · 워터폴 구간 · 송신 모니터가 읽는 것)"""

    def __init__(self, n_frames, nseg=None):
        self.n = int(n_frames)
        self.n_seg = nseg if nseg is not None else max(1, self.n * tng5.BITS_PER_FRAME // SC)
        self.n_coded = self.n_seg * SC
        self.q = tng5.interleave_map(self.n_coded)
        self.seg_frames = [np.unique(self.q[k * SC:(k + 1) * SC] // 96) for k in range(self.n_seg)]
        self.seg_first = np.array([f.min() for f in self.seg_frames])
        self.seg_last = np.array([f.max() for f in self.seg_frames])
        self.seg_nominal = [(k * SC / 96.0, (k + 1) * SC / 96.0) for k in range(self.n_seg)]
        # (정보비트 시작, 끝, 바이트 시작, 바이트 수) — 바이트는 구간 18바이트 기준
        self.seg_bits = [(k * tng5.SEG_INFO, (k + 1) * tng5.SEG_INFO, k * tng5.SEG_BYTES, tng5.SEG_BYTES)
                         for k in range(self.n_seg)]

    def frame_segments(self, f):
        return [k for k, fr in enumerate(self.seg_frames) if f in fr]

    def seg_state_tx(self, frames_done, frames_started):
        st = np.zeros(self.n_seg, dtype=int)
        st[self.seg_first < frames_started] = 1
        st[self.seg_last < frames_done] = 2
        return st


def timeline_pc():
    """packet_timeline 용: 코스타스 길이 = symbol_ms × n_tones → A×4 (1344 ms)"""
    import dataclasses
    return dataclasses.replace(tng5.PC, symbol_ms=tng5.PC.symbol_ms * tng5.SYNC_REP)


def seg_text(segs_b, ok, k):
    b = segs_b[k]
    body = b[tng5.LEN_BYTES:] if k == 0 else b
    return tng5.unpack(body) if ok[k] else "□"


def assemble(segs, nchars=None):
    """구간 결과 [{'index','ok','data'}] → (텍스트, 글자별 구간 번호). 순서대로, 실패 구간은 □ 하나"""
    segs = sorted(segs, key=lambda r: r["index"])
    out, owner, expect = [], [], 0
    for r in segs:
        if r["index"] != expect:
            break
        expect += 1
        if r["ok"]:
            body = r["data"][tng5.LEN_BYTES:] if r["index"] == 0 else r["data"]
            s = tng5.unpack(body)
        else:
            s = "□"
        out.append(s)
        owner += [r["index"]] * len(s)
    t = "".join(out)
    if nchars is not None and "□" not in t:
        t, owner = t[:nchars], owner[:nchars]
    else:
        t2 = t.rstrip()
        owner = owner[:len(t2)]
        t = t2
    return t, owner


# ====================================================================== 엔진
class Tng5Engine:
    def __init__(self, device=None):
        dev = device or __import__("tngpkt.engine", fromlist=["*"]).APP_DEVICE           # 28부: CPU 만
        self.device = torch.device(dev)
        self.model, self.cfg = load_model(tng5.MODEL, self.device)
        self._lock = threading.Lock()
        _, info, data = tng5.tx_waveform("CQ", self.model, self.cfg, self.device)
        self.A, self.B = tng5.templates(data)
        self.pre = info["pre"]                               # 톤 + A×4 (기저대역 샘플)
        self.data0 = info["data0"]                           # 송신 시작 → 첫 데이터 프레임
        self.spanA = max(c[0] + len(c[1]) for c in self.A)
        self.spanB = max(c[0] + len(c[1]) for c in self.B)
        self.preB = -min(c[0] for c in self.B)
        # 구간 수 n 마다 인터리버가 다르다 (끝 2구간). 첫 구간은 후보 n 별로 부호 비트가 다 오면 시험한다
        self._q = {}
        self.need0 = {n: int(self.qmap(n)[:SC].max() // 96 + 1) for n in range(1, NSEG_MAX + 1)}
        self.pc_tl = timeline_pc()

    def qmap(self, nseg):
        q = self._q.get(nseg)
        if q is None:
            q = self._q[nseg] = tng5.interleave_map(nseg * SC)
        return q

    def need(self, nseg, k):
        """구간 수 nseg 메시지에서 구간 k 의 부호 비트가 다 도착하는 프레임 수"""
        return int(self.qmap(nseg)[k * SC:(k + 1) * SC].max() // 96 + 1)

    # ---------------------------------------------------------------- 송신
    def tx_baseband(self, text):
        with self._lock:
            return tng5.tx_waveform(text, self.model, self.cfg, self.device)

    def encode(self, text, fs):
        x, info, _ = self.tx_baseband(text)
        fs_m = fs if (fs % self.cfg.fs_base == 0) else MODEM_FS
        a = resample(baseband_to_audio(x, self.cfg, fs_m), fs_m, fs)
        return a / max(np.max(np.abs(a)), 1e-12) * 10 ** (-3 / 20)

    # ---------------------------------------------------------------- 수신 공통
    def to_bb(self, x, fs):
        """→ (기저대역, 입력 샘플 / 기저대역 샘플). 비율은 항상 실제 입력 레이트 기준 (44.1 kHz → 22.05)"""
        x = np.asarray(x, dtype=np.float64)
        fs_in = float(fs)
        if fs % self.cfg.fs_base != 0:
            x = resample(x, fs, MODEM_FS)
            fs = MODEM_FS
        return audio_to_baseband(x, fs, self.cfg), fs_in / self.cfg.fs_base

    def llr_frames(self, bb, d0, df, f_from, f_to):
        """bb (기저대역, 주파수 보정 전), 첫 데이터 프레임 위치 d0 → (가설, 프레임, 96) LLR (양수 = 1)"""
        from tngpkt.modem4 import _rx_windows
        cfg = self.cfg
        W, C = cfg.win_len, cfg.ctx_len
        t = np.arange(len(bb)) / cfg.fs_base
        y = bb * np.exp(-2j * np.pi * df * t)
        f = np.arange(f_from, f_to)
        st = (int(round(d0)) + f[None, :] * L + np.asarray(OFFS)[:, None] - C).reshape(-1)
        st = np.clip(st, 0, max(0, len(y) - W))
        win = y[st[:, None] + np.arange(W)[None, :]]
        with self._lock:
            llr = _rx_windows(win, self.model, self.device)
        fr = y[np.clip(int(round(d0)) + f * L, 0, max(0, len(y) - L))[:, None] + np.arange(L)[None, :]]
        from tngpkt.stream_ui import band_ratio_db
        return llr.reshape(len(OFFS), len(f), 96), band_ratio_db(fr)

    def decode_segments(self, llr_tx, nseg, q=None):
        """송신 순서 LLR (n,) → 구간 결과 [{'index','ok','data'}]"""
        n = nseg * SC
        if len(llr_tx) < n:
            llr_tx = np.concatenate([llr_tx, np.zeros(n - len(llr_tx), np.float32)])
        if q is None:
            sb, ok = tng5.decode_llr(llr_tx[None, :n], nseg)
            return [{"index": k, "ok": bool(ok[0, k]), "data": sb[0][k]} for k in range(nseg)]
        return [self.decode_one(llr_tx, k, q) for k in range(nseg)]

    def decode_one(self, llr_tx, k, q):
        from tngpkt.conv_sim import viterbi
        from tngpkt.framing import crc16_ccitt
        c = llr_tx[q[k * SC:(k + 1) * SC]].reshape(tng5.REP, -1).sum(0)[None]
        bits = viterbi(c, tng5.RATE_N, tng5.SEG_INFO + tng5.TAIL)[:, :tng5.SEG_INFO]
        by = np.packbits(bits.astype(np.uint8), axis=1)[0]
        data = bytes(by[:tng5.SEG_BYTES])
        return {"index": k, "ok": crc16_ccitt(data) == int.from_bytes(bytes(by[tng5.SEG_BYTES:]), "big"), "data": data}

    def result(self, x, fs, a_bb, df, llr, segs, nchars, why, b_bb=None, rho=None):
        """복조 결과 → engine.DecodeResult (TNG44 와 같은 모양, info['format'] = 'TNG5')"""
        segs = sorted(segs, key=lambda r: r["index"])
        n_ok = sum(1 for r in segs if r["ok"])
        text, _ = assemble(segs, nchars)
        ok = bool(segs) and n_ok == len(segs) and nchars is not None
        try:
            snr = estimate_snr_2500(x, fs, self.cfg, df)
        except Exception:
            snr = float("nan")
        info = {"format": NAME, "mode_name": NAME, "segments": segs, "segments_ok": n_ok, "segments_used": len(segs),
                "declared_segments": seg_count(nchars) if nchars is not None else None, "chars": nchars,
                "partial": text, "n_data": None if llr is None else llr.shape[0],
                "llr": llr, "reason": "" if ok else "CRC16 실패 구간 {}".format(len(segs) - n_ok)}
        m = {"a": {"start": a_bb, "rho": rho}, "b": None if b_bb is None else {"start": b_bb}, "df": df,
             "info": info, "mode": NAME}
        return DecodeResult(ok, text=text if ok else "", snr_db=snr, reason=info["reason"] if not ok else "",
                            df_hz=df, mode="{} · {}".format(NAME, why), info=info,
                            viz={"audio": x, "fs": fs, "msg": m, "tng5": True})

    def decode_at(self, x, fs, a_bb, df, nF=None, nseg=None, b_bb=None, why="", rho=None):
        """오디오 x 안의 알려진 A 위치 (기저대역 번호 = 송신 시작) 로 복조"""
        bb, k = self.to_bb(x, fs)
        d0 = a_bb + self.data0
        if b_bb is not None and nF is None:
            nF = int(round((b_bb - d0) / L)) - 1
        if nF is None:
            nF = max(1, int((len(bb) - d0) // L) - 1)
        nF = int(max(1, min(nF, frames_for_segs(NSEG_MAX))))
        if nF < frames_for_segs(1) and nseg is None:
            # 구간 하나 (부호어 SC 비트) 도 못 채우는 프레임 수 → 안전하게 실패 (이전: decode_one IndexError)
            return DecodeResult(False, reason="TNG5 프레임 부족 ({} < {})".format(nF, frames_for_segs(1)),
                                mode="{} · {}".format(NAME, why), info={"format": NAME, "segments": [], "segments_ok": 0},
                                df_hz=df, viz={})
        g, _ = self.llr_frames(bb, d0, df, 0, nF)
        h = int(np.argmax(np.abs(g).mean(axis=(1, 2))))
        llr = g[h]
        self._h_off = OFFS[h]
        flat = llr.reshape(-1)
        if nseg is None:
            nseg = max(1, (nF * 96) // SC)
        s0 = self.decode_one(flat, 0, tng5.interleave_map(nseg * SC))
        nchars = int.from_bytes(s0["data"][:tng5.LEN_BYTES], "big") if s0["ok"] else None
        if nchars is not None and 0 < seg_count(nchars) <= NSEG_MAX and b_bb is None:
            nseg = seg_count(nchars)
            llr = llr[:frames_for_segs(nseg)]
            flat = llr.reshape(-1)
        segs = self.decode_segments(flat, nseg)
        if segs and segs[0]["ok"]:
            nchars = int.from_bytes(segs[0]["data"][:tng5.LEN_BYTES], "big")
        r = self.result(x, fs, a_bb, df, llr, segs, nchars, why, b_bb=b_bb, rho=rho)
        r.info["h_off"] = self._h_off                            # 정렬 가설 (전문가 모듈: 프레임 시작)
        if b_bb is not None and nchars is not None and seg_count(nchars) > nseg:
            r.info["stopped"] = True                             # B 로 정한 구간 수 < 길이 필드 → 송신 중지
            r.ok, r.text = False, ""
        return r

    def find(self, bb, which="A", floor=None):
        y = torch.as_tensor(bb.astype(np.complex64), device=self.device)[None]
        with self._lock:
            return tng5.detect(y, self.A if which == "A" else self.B, self.device, floor=floor)

    def decode(self, audio, fs):
        """파일 · 역방향용: 버퍼 전체에서 A (· B) 를 찾아 복조"""
        bb, k = self.to_bb(audio, fs)
        dA, dB = tng5.resolve_ab(self.find(bb, "A"), self.find(bb, "B"))
        if not dA:
            return DecodeResult(False, reason="TNG5 코스타스 A 미검출", mode=NAME + " · A 미검출",
                                info={"format": NAME}, viz={})
        a = dA[-1] if dB else dA[0]
        if dB:
            before = [d for d in dA if d["start"] < dB[-1]["start"]]
            a = before[-1] if before else dA[0]
        # B: A 와 주파수가 맞고 (±B_DF_HZ) 구간 하나 이상 뒤에 있는 것만 (데이터 안 다른 주파수 가짜 B 거름)
        b = next((d for d in dB if d["start"] > a["start"] + self.data0 + (frames_for_segs(1) + 1) * L
                  and abs(d["df"] - a["df"]) <= B_DF_HZ), None)
        return self.decode_at(audio, fs, a["start"], a["df"], b_bb=None if b is None else b["start"],
                              why="코스타스 A+B" if b else "A 만")


# ====================================================================== 송신 오디오
def frames_from_segs(segs):
    """구간 바이트 목록 → 데이터 프레임 (tng5.encode_bits 와 같은 규칙, 텍스트 대신 구간을 받는다)"""
    c = np.concatenate([tng5.seg_codeword(sb) for sb in segs])
    q = tng5.interleave_map(len(c))
    tx = np.empty_like(c)
    tx[q] = c
    nF = int(math.ceil(len(tx) / tng5.BITS_PER_FRAME))
    tx = np.concatenate([tx, np.random.default_rng(len(tx)).integers(0, 2, nF * tng5.BITS_PER_FRAME - len(tx))])
    return tx.reshape(nF, tng5.BITS_PER_FRAME)


class Tx5Audio:
    """
    stream_ui.TxAudio 와 같은 모양.
    송신 중지 (stopped): 이미 비트가 나간 구간까지만 남긴 메시지 (앞 구간 바이트 그대로 → 길이 필드 = 원래 글자 수)
    로 다음 프레임 경계에서 갈아타고 끝 가드 · B×4 로 정상 종료. 이어 붙이기 전 나간 프레임 비트가 같은지 확인한다.
    """
    mode = NAME

    def __init__(self, eng, text, fs=48000.0, segs=None, gain=None):
        from tngpkt.modem5 import _frames_to_baseband
        self._eng = eng
        if segs is None:
            x, info, data = eng.tx_baseband(text)
        else:                                                     # 송신 중지: 원래 메시지 앞 구간만
            fr = frames_from_segs(segs)
            with eng._lock:
                data = _frames_to_baseband(fr.astype(np.float32), tng5.GUARD5, eng.model, eng.cfg, eng.device)
            x, info = tng5.build_tx(data)
            info.update({"text": text, "bad": [], "segments": list(segs), "n_frames": fr.shape[0]})
            self.frames = fr
        cfg = eng.cfg
        fs_m = fs if (fs % cfg.fs_base == 0) else MODEM_FS
        a = resample(baseband_to_audio(x, cfg, fs_m), fs_m, fs)
        self.gain = gain if gain is not None else 10 ** (-3 / 20) / max(np.max(np.abs(a)), 1e-12)
        self.audio = a * self.gain
        self.fs = float(fs)
        self.text = info["text"]
        self.bad = info["bad"]
        self.segs = info["segments"]
        self.S = b"".join(self.segs)
        self.P = self.text.encode("ascii")
        self.plan = Plan5(info["n_frames"], len(self.segs))
        self.si = {"Pp": self.S, "n_data": info["n_frames"]}
        self.lead = 0.2
        fsb = cfg.fs_base
        self.t_tone = self.lead
        self.t_a = self.lead + N_TONE / fsb
        self.t_data = self.lead + info["pre"] / fsb
        self.frame_s = L / fsb
        self.frame_t0 = self.t_data + self.frame_s
        self.t_b = self.lead + info["b_start"] / fsb
        self.t_b_len = N_SYNC / fsb
        self.dur = len(self.audio) / self.fs
        self.data_bb = data

    def progress(self, t):
        u = (t - self.frame_t0) / self.frame_s
        done = int(np.clip(np.floor(u), 0, self.plan.n))
        started = int(np.clip(np.ceil(u), 0, self.plan.n))
        return done, started, self.plan.seg_state_tx(done, started)

    def char_segments(self):
        """글자마다 (위치, 구간, 구간) — 첫 구간 24자, 이후 27자"""
        out = []
        for i in range(len(self.text)):
            k = 0 if i < 24 else 1 + (i - 24) // 27
            out.append((i, k, k))
        return out

    def stopped(self, engine, t_now, margin_s=0.35):
        """중지: t_now + margin 뒤 첫 프레임 경계에서 잘린 메시지로. 반환 (새 Tx5Audio, 이어 붙일 샘플) 또는 (None, None)"""
        f_sw = int(np.ceil((t_now + margin_s - self.frame_t0) / self.frame_s))
        n = self.plan.n_seg
        if f_sw >= self.plan.n - 1:
            return None, None
        f_sw = max(f_sw, 1)
        started = np.nonzero(self.plan.seg_first < f_sw)[0]
        keep = int(started.max()) + 1 if len(started) else 1
        full = frames_from_segs(self.segs)
        while keep < n:                                            # 나간 프레임 비트가 같아지는 가장 짧은 구간 수
            if np.array_equal(frames_from_segs(self.segs[:keep])[:f_sw], full[:f_sw]):
                break
            keep += 1
        if keep >= n:
            return None, None
        new = Tx5Audio(self._eng, self.text, self.fs, segs=self.segs[:keep], gain=self.gain)
        t_sw = self.frame_t0 + f_sw * self.frame_s - 0.08
        i_sw = int(round(t_sw * self.fs))
        i0 = max(0, i_sw - int(0.5 * self.fs))
        x, y = self.audio[i0:i_sw], new.audio[i0:i_sw]
        c = float(np.dot(x, y) / max(np.dot(y, y), 1e-20))
        new.audio = new.audio * c
        new.gain *= c
        new.splice_err = float(np.max(np.abs(x - c * y)) / max(np.max(np.abs(x)), 1e-12))
        new.f_switch = f_sw
        new.keep = keep
        new.P = self.P                                             # 원래 길이 (표시용)
        return new, i_sw

    # 송신 해부 (widgets_qso.TxAnalysisPanel) 가 부르는 구간 보기
    def seg_view(self, k):
        from tngpkt.framing import crc16_ccitt
        from tngpkt.conv_sim import encode as conv_encode
        data = self.segs[k]
        full = data + crc16_ccitt(data).to_bytes(2, "big")
        kind = ["len" if (k == 0 and i < 2) else ("crc" if i >= len(data) else "body") for i in range(len(full))]
        bits = np.unpackbits(np.frombuffer(full, np.uint8)).astype(np.int64)
        coded = conv_encode(np.concatenate([bits, np.zeros(tng5.TAIL, np.int64)]), tng5.RATE_N)
        pos = self.plan.q[k * SC:(k + 1) * SC]
        return full, kind, bits, coded, pos

    def seg_chars(self, k):
        body = self.segs[k][tng5.LEN_BYTES:] if k == 0 else self.segs[k]
        n0 = 0 if k == 0 else 24 + 27 * (k - 1)
        return tng5.unpack(body)[:max(0, len(self.text) - n0)]

    def chars_done(self, sent):
        return min(len(self.text), 0 if sent <= 0 else 24 + 27 * (sent - 1))


# ====================================================================== 실시간 수신
class Live5:
    """A 검출 뒤 프레임을 차례로 받아 구간을 결정한다 (길이 모를 때는 최대 길이 인터리버 기준)"""

    def __init__(self, eng, a_abs, df, fs, k):
        self.eng, self.fs, self.k = eng, float(fs), k
        self.a_abs, self.df = a_abs, df
        self.d0_abs = a_abs + eng.data0 * k                      # 첫 데이터 프레임 (입력 절대 번호)
        self.g = []                                              # 프레임별 (가설, 96)
        self.band, self.llr_lv = [], []
        self.fed = 0
        self.segments = {}
        self.nchars = None
        self.nseg = None
        self.n_true = None
        self.tried = set()
        self.done = False
        self.end_reason = None
        self.format = NAME

    def _windows(self, ring, f_from, f_to):
        eng, k = self.eng, self.k
        W, C = eng.cfg.win_len, eng.cfg.ctx_len
        need_end = self.d0_abs + (f_to * L + (W - C) + 8 + 64) * k
        if ring.count < need_end:
            return None
        x0 = int(self.d0_abs + (f_from * L - C - 8 - 160) * k)
        s0, raw = ring.read(x0, int(need_end))
        if len(raw) < int((need_end - x0) * 0.98):
            return None
        bb, kk = eng.to_bb(raw, self.fs)
        d0 = (self.d0_abs - s0) / kk
        g, ratio = eng.llr_frames(bb, d0 + f_from * L, self.df, 0, f_to - f_from)
        return g, ratio

    def best(self):
        G = np.stack(self.g, axis=1)                             # (H, n, 96)
        h = int(np.argmax(np.abs(G).mean(axis=(1, 2))))
        return G[h]

    def end_abs(self):
        """길이로 끝을 알 때 코스타스 B 예상 시작 (입력 절대 번호)"""
        return self.d0_abs + (self.n_true + 1) * L * self.k

    def step(self, ring, max_frames=8):
        if self.done:
            return []
        n_to = self.n_true if self.n_true is not None else frames_for_segs(NSEG_MAX)
        got = None
        if self.fed < n_to:
            f_to = min(n_to, self.fed + max_frames)
            while f_to > self.fed:
                got = self._windows(ring, self.fed, f_to)
                if got is not None:
                    break
                f_to = self.fed + (f_to - self.fed) // 2
        if got is None:
            return []
        g, ratio = got
        for i in range(g.shape[1]):
            self.g.append(g[:, i])
            self.band.append(float(ratio[i]))
        self.fed += g.shape[1]
        G = self.best()
        self.llr_lv = list(np.abs(G).mean(axis=1))
        flat = G.reshape(-1)
        new = []
        eng = self.eng
        if self.nseg is None:                                     # 길이 모름: 후보 구간 수마다 첫 구간 시험
            for n in range(1, NSEG_MAX + 1):
                if n in self.tried or self.fed < eng.need0[n]:
                    continue
                self.tried.add(n)
                r = eng.decode_one(flat, 0, eng.qmap(n))
                if r["ok"]:
                    nc = int.from_bytes(r["data"][:tng5.LEN_BYTES], "big")
                    if seg_count(nc) == n:
                        self.nchars, self.nseg, self.n_true = nc, n, frames_for_segs(n)
                        break
        if self.nseg is not None:
            if self.fed >= self.n_true:
                segs = eng.decode_segments(flat[:self.nseg * SC], self.nseg)
                self.segments = {r["index"]: r for r in segs}
                self.done, self.end_reason = True, "length"
                return segs
            for kk in range(self.nseg):
                if kk in self.segments or self.fed < eng.need(self.nseg, kk):
                    continue
                r = eng.decode_one(flat, kk, eng.qmap(self.nseg))
                self.segments[kk] = r
                new.append(r)
        if self._lost():
            self.done, self.end_reason = True, "loss"
        return new

    def _lost(self):
        from tngpkt.stream_ui import fast_lost
        if fast_lost(self.band):                             # 강신호 갑자기 끊김: 약 1 s
            self.fast_loss = True
            return True
        if len(self.band) < LOSS_T:
            return False
        if max(self.band[-LOSS_T:]) >= LOSS_E_DB or np.mean(self.llr_lv[-LOSS_T:]) >= LOSS_LLR:
            return False
        dec = [self.segments[k] for k in sorted(self.segments)]
        return not dec or not dec[-1]["ok"]


class Tng5Receiver:
    SCAN_S = 3.0
    STEP_S = 0.25
    EDGE_S = 0.15
    PRE_S = 0.6
    POST_S = 0.2                     # (0.4 → 0.2: 원격 세션 3번)
    B_TOL_S = 0.5
    # 가짜 B: B 뒤 0.6 s 평균 |LLR| (가설 최대) 이 이 값 이상이면 신호 계속. 측정: 잡음 최대 0.69 → 진짜 B 는 버리지 않음.
    # 신호 중앙 -8 dB 2.56 · -12 dB 0.86 · -16 dB 0.39 → 약한 신호의 가짜 B 는 못 거른다 (그때는 B 인정, 이전과 같음)
    LLR_T = 1.0
    MAX_S = 225.0             # 최대 메시지 (NSEG_MAX 구간)

    def __init__(self, eng5, fs, on_status, on_marker, on_decoded, sync=False):
        self.eng, self.fs, self.sync = eng5, float(fs), sync
        self.on_status, self.on_marker, self.on_decoded = on_status, on_marker, on_decoded
        self.on_live = None
        self.on_heartbeat = None
        self.on_a_confirm = None
        self.k = self.fs / eng5.cfg.fs_base          # 입력 샘플 / 기저대역 샘플 (실제 입력 레이트 기준, 44.1 kHz → 22.05)
        self.ring = RingBuffer(self.fs * (self.MAX_S + 10.0))
        self.stats = {"A": 0, "B": 0, "decoded": 0, "ok": 0, "scans": 0}
        self.paused = False
        self.ignore_before = 0
        self.stop_req = False
        self.enabled = True
        self._stop = threading.Event()
        self._thread = None
        self._scan_from = 0                      # 다음 탐색 시작 후보 (입력 절대 번호)
        self._done = []                          # 최근 인정한 후보 (다음 묶음의 A/B 정리용)
        self._a_hist = []                        # 실시간으로 이미 받은 A (입력 절대 번호) — 역방향 동기가 다시 복조하지 않게
        self._protect = []                       # 끝낸 메시지 보호 구간 [(시작, 끝, df)]
        self._sets = None                        # TNG5 코스타스 한 벌 템플릿 (공동 판정용)
        self._cid = 0
        self._floor = None
        self._floor_t = -1e9
        self.expect_b_until = 0
        self._reset_state()

    # ---------------------------------------------------------------- 제어 (RealtimeReceiver 와 같음)
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rt-rx5")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def feed(self, x):
        self.ring.write(x)

    def pause(self, on, holdoff_s=0.3):
        self.paused = bool(on)
        self._reset_state()
        self.ignore_before = self.ring.count + (int(holdoff_s * self.fs) if not on else 0)
        self._scan_from = max(self._scan_from, self.ignore_before)

    def _reset_state(self):
        self.state = "idle"
        self.live = None
        self.a = None
        self.pending = None
        self.b_hold = None
        self.seen = {"A": [], "B": []}
        self._cand = []

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                if not self.paused and self.enabled:
                    self._step()
            except Exception as ex:
                self.on_status("TNG5 수신 오류 — {}: {}".format(type(ex).__name__, ex), True)
            self._stop.wait(max(0.0, self.STEP_S - (time.time() - t0)))

    def step(self):
        if not self.paused and self.enabled:
            self._step()

    def _ev(self, ev):
        ev["mode"] = NAME
        if self.on_live is not None:
            try:
                self.on_live(ev)
            except Exception:
                pass

    # ---------------------------------------------------------------- 한 주기
    def _step(self):
        now = self.ring.count
        fs = self.fs
        if self.pending is not None and now >= self.pending[0]:
            p, self.pending = self.pending, None
            self._final(*p[1:])
        if self.b_hold is not None:
            self._check_b_hold()
        lv = self.live
        if lv is not None and not lv.done and self.state == "rx":
            had = lv.n_true
            new = lv.step(self.ring)
            if lv.done and lv.end_reason == "length":
                segs = [lv.segments[k] for k in sorted(lv.segments)]
                self._ev({"type": "final_segments", "segs": segs, "fed": lv.fed, "length": lv.nchars})
                end = int(lv.end_abs() + 0.35 * fs)
                self.pending = (end, "length")
                self.state = "decoding"
                self.expect_b_until = int(lv.end_abs() + N_SYNC * self.k + 2.0 * fs)
                self._protect_until(self.a["abs"] - 0.3 * fs, lv.end_abs() + (N_SYNC + 384) * self.k + 0.5 * fs,
                                    self.a["df"])
                self.on_status("TNG5 길이 필드로 메시지 끝 확인 — 복조 중…", False)
            elif lv.done and lv.end_reason == "loss":
                if new:
                    self._ev({"type": "segments", "segs": new, "fed": lv.fed})
                self._ev({"type": "lost", "fed": lv.fed})
                self.on_status("TNG5 신호 소실 — 받은 부분까지 복조", True)
                self._final("loss")
            elif new:
                self._ev({"type": "segments", "segs": new, "fed": lv.fed})
            if had is None and lv.n_true is not None and not lv.done:
                self._ev({"type": "format", "format": NAME})
        if self.stop_req:
            self.stop_req = False
            if self.state == "rx" and self.a is not None:
                self._ev({"type": "manual_stop"})
                self._final("manual")
        self._scan(now)
        if self.state == "rx" and self.pending is None and now - self.a["abs"] > int((self.MAX_S + 1.0) * fs):
            self._final("maxlen")

    def _noise_floor(self, now):
        if time.time() - self._floor_t < 2.0 and self._floor is not None and not self.sync:
            return self._floor
        s0, raw = self.ring.read(now - int(10.0 * self.fs), now)
        if len(raw) < int(1.0 * self.fs):
            return None
        bb, _ = self.eng.to_bb(raw, self.fs)
        self._floor = tng5.noise_floor(torch.as_tensor(bb.astype(np.complex64))[None])
        self._floor_t = time.time()
        return self._floor

    def _scan(self, now):
        fs, k = self.fs, self.k
        eng = self.eng
        lo = max(self._scan_from, self.ignore_before, now - int(self.MAX_S * fs))
        a0 = max(int(lo - (self.EDGE_S + (eng.preB + 8) / eng.cfg.fs_base) * fs), self.ignore_before, 0)
        if now - a0 < int((eng.spanA / eng.cfg.fs_base + 2 * self.EDGE_S + 0.25) * fs):
            return
        a0 = min(a0, now - int(self.SCAN_S * fs))
        s0, raw = self.ring.read(max(a0, 0), now)
        if len(raw) == 0 or np.max(np.abs(raw)) < 1e-7:
            self._scan_from = now
            return
        self.stats["scans"] += 1
        bb, kk = eng.to_bb(raw, fs)
        floor = self._noise_floor(now)
        edge = int(self.EDGE_S * eng.cfg.fs_base)
        p_lo = max(edge, int(np.ceil((lo - s0) / kk)), eng.preB + 8)
        y = torch.as_tensor(bb.astype(np.complex64), device=eng.device)[None]
        last = {}
        hits = []
        p_hiA = len(bb) - eng.spanA - edge
        for which, chunks, off in (("A", eng.A, 0), ("B", eng.B, N_TONE)):
            # B 후보는 A 후보와 같은 코스타스 시작 범위만 (B 템플릿이 짧아 A×4 에 먼저 걸리는 것 방지)
            p_hi = p_hiA + off
            last[which] = p_hiA
            if p_hi <= p_lo + off:
                continue
            st = np.arange(p_lo + off, p_hi, tng5.STEP)
            with eng._lock:
                ds = tng5.detect(y, chunks, eng.device, starts=st, floor=floor)
            hits.append((which, ds))
        if "A" in last and p_hiA > p_lo:
            self._scan_from = int(s0 + p_hiA * kk)
        # 후보로 모았다가 더 센 경쟁 후보가 더 나올 수 없는 시점에 인정: 같은 송신의 경쟁 후보(반복 배열의 벌 단위
        # 어긋남, A 템플릿이 B×4 에 걸림)는 후보 뒤 3벌 (1.008 s) 안에만 생긴다 → 탐색이 후보 + 3벌 + 0.1 s 를 지나면 확정.
        # 판단 기준 (2 s 안 가장 센 것 · A/B 정리) 은 그대로, 기다리는 시간만 고정 2 s → 경쟁 가능 구간.
        # 후보가 처음 잡히면 바로 워터폴에 임시 표시 (점선), 확정 = 실선, 탈락 = 삭제.
        for which, ds in hits:
            for d in ds:
                pos = int(round(s0 + d["start"] * kk))
                if pos >= self.ignore_before:
                    c = dict(d, kind=which, abs=pos, start=pos / kk, cid=self._cid)
                    self._cid += 1
                    self._cand.append(c)
        covered = self._scan_from
        dd = int(tng5.DEDUP_S * self.fs)
        hold = int(RIVAL_S * self.fs)
        nt = int(N_TONE * kk)                                     # B 후보는 A 기준 좌표 (B − 톤) 로 같은 묶음에
        ca = lambda c: c["abs"] - (nt if c["kind"] == "B" else 0)
        span_ = int(N_SYNC * kk)
        raw_ = lambda c: c.get("raw", c["metric"])

        def beaten(c):
            return any(x is not c and raw_(x) >= raw_(c) and
                       ((x["kind"] == c["kind"] and abs(x["abs"] - c["abs"]) < dd) or
                        (x["kind"] != c["kind"] and abs(ca(x) - ca(c)) < span_)) for x in self._cand + self._done)
        for c in self._cand:                                      # 임시 표시 = 묶음에서 지금까지 가장 센 후보 하나
            lead = not beaten(c)
            if lead and not c.get("shown"):
                c["shown"] = True
                self._mark(c, tentative=True)
            elif not lead and c.get("shown"):
                c["shown"] = False
                self._mark(c, remove=True)
        ready = [c for c in self._cand if ca(c) + hold < covered]
        if not ready:
            return
        self._cand = [c for c in self._cand if ca(c) + hold >= covered]
        span = int(N_SYNC * kk)
        raw = lambda c: c.get("raw", c["metric"])
        keep = []
        for c in sorted(ready, key=lambda c: -raw(c)):
            rivals = keep + self._cand + self._done
            same = [x for x in rivals if x["kind"] == c["kind"] and abs(x["abs"] - c["abs"]) < dd]
            cross = [x for x in rivals if x["kind"] != c["kind"] and abs(ca(x) - ca(c)) < span]
            if any(raw(x) >= raw(c) for x in same + cross) or any(x in keep for x in same):
                if c.get("shown"):
                    self._mark(c, remove=True)                     # 같은 송신의 더 센 후보 (A 템플릿이 B×4 에 걸림 등)
                continue
            keep.append(c)
        self._done = (self._done + keep)[-6:]
        for c in sorted(keep, key=lambda c: c["abs"]):
            if self._dup(c["kind"], c["abs"]):
                if c.get("shown"):
                    self._mark(c, remove=True)
                continue
            self.seen[c["kind"]].append(c["abs"])
            self._on_detect(c["kind"], c["abs"], c)

    def _mark(self, c, tentative=False, remove=False):
        """워터폴 표시: 임시 (점선) · 삭제. 확정 표시는 _on_detect 가 같은 id 로 보낸다"""
        m = {"kind": c["kind"], "abs": c["abs"] if c["kind"] == "B" else c["abs"] + int(N_TONE * self.k),
             "df": c["df"], "rho": c.get("metric", 0.0), "mode": NAME, "dur": N_SYNC / self.eng.cfg.fs_base,
             "id": "{}{}".format(NAME, c["cid"])}
        if remove:
            m["remove"] = True
        elif tentative:
            m["tentative"] = True
        self.on_marker(m)

    def _dup(self, kind, pos):
        win = int(tng5.DEDUP_S * self.fs)
        self.seen[kind] = [p for p in self.seen[kind] if abs(p - pos) < int(60 * self.fs)]
        return any(abs(p - pos) < win for p in self.seen[kind])

    def zones(self):
        """TNG44 검출을 막을 TNG5 동기 구간 (입력 절대 번호): [(시작, 끝, df)]"""
        out = []
        k, m = self.k, int(0.3 * self.fs)
        if self.a is not None:
            out.append((self.a["abs"] - m, self.a["abs"] + (self.eng.data0 + 8) * k + m, self.a["df"]))
            lv = self.live
            end = (lv.end_abs() if (lv is not None and lv.n_true is not None) else None)
            out.append((self.a["abs"], (end + N_SYNC * k + m) if end is not None else self.ring.count + 10 * self.fs,
                        self.a["df"]))
        for p in self.seen["B"][-3:]:
            out.append((p - 384 * k - m, p + N_SYNC * k + m, None))
        out += [z for z in self._protect if z[1] > self.ring.count - 60 * self.fs]
        return out

    def _protect_until(self, lo, hi, df):
        """끝낸 메시지의 보호 구간 (예상 B×4 끝까지) — 상태를 지운 뒤에도 TNG44 검출을 막고, 그 안에서 시작한 TNG44 는 되돌림"""
        self._protect = [z for z in self._protect if z[1] > self.ring.count - 60 * self.fs] + [(int(lo), int(hi), df)]
        if self.on_a_confirm is not None:
            self.on_a_confirm(int(lo), int(hi))

    def set_rho(self, pos, now, back_s=0.75):
        """
        TNG44 검출 위치 근처 (pos − back_s ~ 지금) 에서 TNG5 코스타스 한 벌 (A · B) 정규화 상관 ρ² 최대 (±100 Hz).
        TNG5 A×4 · B×4 는 같은 벌을 4번 반복하므로 TNG44 가 그 안에 걸렸다면 앞뒤 벌이 버퍼에 이미 있다.
        """
        if self._sets is None:
            a, b = tng5.tng5_chunks()
            self._sets = [(0, a, "costas"), (0, b, "costas")]
        n = len(self._sets[0][1])
        s0, raw = self.ring.read(pos - int(back_s * self.fs), now)
        if len(raw) < int((n / BB_FS + 0.2) * self.fs):
            return None
        bb, kk = self.eng.to_bb(raw, self.fs)
        y = torch.as_tensor(bb.astype(np.complex64), device=self.eng.device)[None]
        hi = len(bb) - 2 * n - 1
        if hi <= n:
            return None
        st = np.arange(n, hi, tng5.STEP)
        best = 0.0
        with self.eng._lock:
            for c in self._sets:
                _, r = tng5.metric(y, [c], st, self.eng.device)
                best = max(best, float(r.max()) / n)
        return best

    def _rho(self, d):
        return float(d.get("metric", 0.0)) / max(len(self.eng.A), 1) / 100.0

    def _on_detect(self, kind, pos, d):
        fs = self.fs
        self.stats[kind] += 1
        dur = (N_SYNC if kind == "B" else N_TONE + N_SYNC) / self.eng.cfg.fs_base
        mk = {"kind": kind, "abs": pos if kind == "B" else pos + int(N_TONE * self.k), "df": d["df"],
              "rho": d.get("metric", 0.0), "mode": NAME, "dur": N_SYNC / self.eng.cfg.fs_base,
              "id": "{}{}".format(NAME, d["cid"]) if "cid" in d else None}
        if kind == "B":
            self.on_marker(mk)
        if kind == "A" and self.state == "rx" and self.a is not None:
            # 탐색 범위 ±100 Hz 전체가 한 신호 대역 안이라 겹친 두 송신은 어차피 함께 복조 못 한다 → 주파수 무관
            lv = self.live
            inside = lv is not None and lv.n_true is not None and pos < lv.end_abs()
            weak = d.get("raw", 0.0) < 0.5 * self.a.get("raw", 0.0)
            if inside or weak:                       # 수신 중 메시지 데이터에 걸린 A
                self.on_status("TNG5 코스타스 A 무시 — 수신 중 메시지 안 ({})".format(
                    "길이 필드 범위" if inside else "현재 A 의 절반 미만"), True)
                if mk["id"]:
                    self.on_marker(dict(mk, remove=True))
                return dur
        if kind == "A":
            self.on_marker(mk)
            if self.state == "rx" and self.pending is None:
                self._final("next_a", end=pos - int(0.3 * fs))
            self.state = "rx"
            self._a_hist = [p for p in self._a_hist if p > pos - int((self.MAX_S + 10) * fs)] + [pos]
            self.a = {"abs": pos, "df": d["df"], "rho": d.get("metric", 0.0), "raw": d.get("raw", 0.0)}
            if self.on_a_confirm is not None:
                m_ = int(0.3 * fs)
                self.on_a_confirm(pos - m_, pos + int((self.eng.data0 + 8) * self.k) + m_)
            self.live = Live5(self.eng, pos, d["df"], fs, self.k)
            self._ev({"type": "start", "abs": pos + int(N_TONE * self.k), "tx_abs": pos, "df": d["df"],
                      "rho": d.get("metric", 0.0), "fs": fs})
            self.on_status("TNG5 신호 감지 (주파수 오차 {:+.1f} Hz)".format(d["df"]), False)
        else:
            self._on_b(pos, d)
        return dur

    def _on_b(self, pos, d):
        fs = self.fs
        lv = self.live
        if self.state == "rx" and self.a is not None and abs(d["df"] - self.a["df"]) > B_DF_HZ:
            self.on_status("TNG5 코스타스 B 무시 — A 와 주파수 불일치 ({:+.1f} Hz)".format(d["df"] - self.a["df"]), True)
            self._ev({"type": "b_ignored", "abs": pos, "df_off": d["df"] - self.a["df"]})
            return
        if self.state == "rx" and lv is not None and not d.get("llr_ok"):
            if lv.n_true is not None:
                exp = lv.end_abs()
                if pos < exp - int(self.B_TOL_S * fs) and pos > self.a["abs"] + int((self.eng.data0 + 3 * L) * self.k):
                    self.b_hold = (pos, d)            # 예상보다 이른 B: 송신 중지일 수 있다 → B 뒤 신호 소실이면 인정
                    return
                if abs(pos - exp) > int(self.B_TOL_S * fs):
                    self.on_status("TNG5 코스타스 B 무시 — 길이 필드 예상 끝과 {:+.1f} s 차이".format((pos - exp) / fs), True)
                    self._ev({"type": "b_ignored", "abs": pos, "off_s": (pos - exp) / fs})
                    return
            elif pos > self.a["abs"] + int((self.eng.data0 + 3 * L) * self.k):
                self.b_hold = (pos, d)
                return
        if self.state == "rx" and pos > self.a["abs"] + int((self.eng.data0 + 3 * L) * self.k):
            self.pending = (pos + int(N_SYNC * self.k + self.POST_S * fs), "b", pos)
            self.state = "decoding"
            if lv is not None:
                lv.done = True
            self._ev({"type": "b", "abs": pos, "df": d["df"], "fs": fs})
            self.on_status("TNG5 메시지 끝 감지 — 복조 중…", False)
        elif self.state == "idle" and pos <= self.expect_b_until:
            pass
        elif self.state == "idle" and self.pending is None:
            self._ev({"type": "b_back", "abs": pos, "df": d["df"], "fs": fs})
            self.on_status("TNG5 메시지 끝(B)만 감지 — 시작(A)을 되짚어 찾는 중…", False)
            self._reverse(pos, d)

    def _check_b_hold(self):
        pos, d = self.b_hold
        lv = self.live
        if self.state != "rx" or lv is None:
            self.b_hold = None
            return
        if lv.n_true is not None and abs(pos - lv.end_abs()) <= int(self.B_TOL_S * self.fs):
            self.b_hold = None
            self._on_b(pos, d)
            return
        f0 = int(np.ceil((pos + N_SYNC * self.k + 0.05 * self.fs - lv.d0_abs) / (L * self.k)))
        nf = max(1, int(round(B_LLR_S * self.eng.cfg.fs_base / L)))
        got = lv._windows(self.ring, f0, f0 + nf)
        if got is None:
            return
        self.b_hold = None
        m = float(np.abs(got[0]).mean(axis=(1, 2)).max())
        if m < self.LLR_T:
            self._on_b(pos, dict(d, llr_ok=True))
        else:
            self.on_status("TNG5 코스타스 B 무시 — B 뒤 신호 계속 (|LLR| {:.2f})".format(m), True)
            self._ev({"type": "b_ignored", "abs": pos, "llr": m})

    # ---------------------------------------------------------------- 복조
    def _final(self, why, b_pos=None, end=None):
        lv, a = self.live, self.a
        self._reset_state()
        if a is None:
            return
        fs, k = self.fs, self.k
        # 보호 구간: 길이 필드 예상 끝 (B×4 포함) 또는 B 끝 또는 지금까지 (길이 · B 모두 모름)
        hi = self.ring.count
        if lv is not None and lv.n_true is not None:
            hi = max(hi, lv.end_abs() + (N_SYNC + 384) * k)
        if b_pos is not None:
            hi = max(hi, b_pos + N_SYNC * k)
        self._protect_until(a["abs"] - 0.3 * fs, hi + 0.5 * fs, a["df"])
        start = a["abs"] - int(self.PRE_S * fs)
        end = end if end is not None else (b_pos + int((N_SYNC * k) + self.POST_S * fs) if b_pos else self.ring.count)
        s0, x = self.ring.read(start, end)
        if len(x) < int(1.0 * fs):
            return
        a_bb = (a["abs"] - s0) / k
        b_bb = None if b_pos is None else (b_pos - s0) / k
        label = {"b": "코스타스 B", "length": "길이 필드", "loss": "신호 소실", "manual": "수동 중지",
                 "next_a": "B 미검출 — 다음 신호 시작", "maxlen": "B 미검출 — 최대 길이 초과"}.get(why, why)
        nF = lv.n_true if (lv is not None and lv.n_true is not None and b_pos is None) else None
        if nF is None and b_pos is None and lv is not None:
            nF = max(1, lv.fed)
        self.stats["decoded"] += 1

        def work():
            try:
                r = self.eng.decode_at(x, fs, a_bb, a["df"], nF=nF, b_bb=b_bb, why=label, rho=a.get("rho"))
            except Exception as ex:
                r = DecodeResult(False, reason="{}: {}".format(type(ex).__name__, ex), mode=NAME,
                                 info={"format": NAME})
            if r.ok:
                self.stats["ok"] += 1
            self.on_decoded(r, "실시간 · {} · {}".format(NAME, label))

        if self.sync:
            work()
        else:
            threading.Thread(target=work, daemon=True, name="rt-decode5").start()

    def _reverse(self, pos, d):
        """B 만 검출: 링버퍼를 되짚어 A 를 찾는다"""
        fs = self.fs
        end = pos + int(N_SYNC * self.k + self.POST_S * fs)

        def work():
            s0, x = self.ring.read(pos - int(self.MAX_S * fs), pos + int(0.2 * fs))
            if len(x) < int(3 * fs):
                return
            bb, kk = self.eng.to_bb(x, fs)
            # A: B 앞 2 s 이상 · 주파수 일치 · 실시간으로 이미 받은 A 가 아닌 것 (옛 메시지를 다시 복조하지 않게)
            dA = [a for a in self.eng.find(bb, "A") if a["start"] * kk + s0 < pos - int(2 * fs)
                  and abs(a["df"] - d["df"]) <= B_DF_HZ
                  and not any(abs(a["start"] * kk + s0 - p) < int(0.5 * fs) for p in self._a_hist)]
            if not dA:
                self._ev({"type": "b_only", "abs": pos, "df": d["df"]})
                return
            a = dA[-1]
            s1, x2 = self.ring.read(int(s0 + a["start"] * kk) - int(self.PRE_S * fs), end)
            a_bb = (s0 + a["start"] * kk - s1) / kk
            r = self.eng.decode_at(x2, fs, a_bb, a["df"], b_bb=(pos - s1) / kk, why="코스타스 B 역방향")
            self.on_decoded(r, "실시간 · {} · 코스타스 B 역방향".format(NAME))

        if self.sync:
            work()
        else:
            threading.Thread(target=work, daemon=True, name="rt-back5").start()


# ====================================================================== 두 모드 묶음
class MultiRx:
    """
    TNG44 수신기(RealtimeReceiver) · TNG5 수신기 묶음. 앱은 이전처럼 self._rx 하나로 부른다.
    속성은 TNG44 수신기로 넘긴다 (기존 시험 스크립트 호환). 전체 동시일 때 TNG5 동기 구간의 TNG44 검출은 버린다.
    """

    def __init__(self, rx44, rx5, active):
        self.__dict__["rx44"] = rx44
        self.__dict__["rx5"] = rx5
        self.__dict__["active"] = set(active)
        rx44_detect = rx44._on_detect

        def gate(kind, pos, d, now):
            if "TNG5" in self.active and rx5 is not None:
                for a, b, df in rx5.zones():
                    if a <= pos <= b and (df is None or abs(df - d["df"]) < 60.0):
                        self.__dict__["zone_drop"] = self.__dict__.get("zone_drop", 0) + 1
                        return                        # TNG5 동기 · 데이터 구간에 걸린 TNG44 검출 (교차 오검출)
                # 모드 간 공동 판정: 같은 시간대의 TNG5 코스타스 한 벌 (A · B, 48 ms · 41.67 Hz) ρ² 가 이 TNG44 검출
                # ρ² 이상이면 TNG5 동기에 걸린 것 → 버림 (TNG5 확정 전, 버퍼에 이미 있는 오디오로 바로)
                r5 = rx5.set_rho(pos, now)
                if r5 is not None and r5 >= d["rho"]:
                    rx44.on_status("TNG44 {} 무시 — TNG5 동기 (ρ² TNG5 {:.2f} ≥ TNG44 {:.2f})".format(
                        kind, r5, d["rho"]), False)
                    self.__dict__["joint_drop"] = self.__dict__.get("joint_drop", 0) + 1
                    return
            if "TNG44" not in self.active:
                return
            return rx44_detect(kind, pos, d, now)
        rx44._on_detect = gate
        # TNG44 검출기는 TNG5 A×4 앞부분에 먼저 걸린다 (TNG5 확정은 A×4 끝 뒤). TNG5 A 가 확정되면 그 동기 구간
        # 안에서 시작한 TNG44 메시지를 되돌린다 (tng5.suppress44 와 같은 구간 기준, TNG44 스레드에서 처리)
        rx44_step = rx44._step

        def step44():
            req = self.__dict__.get("cancel44")
            if req is not None:
                self.__dict__["cancel44"] = None
                self._cancel44(*req)
            rx44_step()
        rx44._step = step44
        if rx5 is not None:
            rx5.on_a_confirm = lambda lo, hi: self.__dict__.__setitem__("cancel44", (lo, hi))

    def _cancel44(self, lo, hi):
        r = self.rx44
        for kind in ("A", "B"):
            for p in [p for p in r.seen[kind] if lo <= p <= hi]:
                r.on_marker({"kind": kind, "abs": p, "df": 0.0, "rho": 0.0, "mode": "TNG44", "remove": True})
        if r.a is not None and lo <= r.a["abs"] <= hi:
            self.__dict__["cancel_n"] = self.__dict__.get("cancel_n", 0) + 1
            r._reset_state()
            r._live_event({"type": "cancel"})
            r.on_status("TNG44 검출 취소 — TNG5 동기 구간", False)

    def __getattr__(self, k):
        return getattr(self.rx44, k)

    def __setattr__(self, k, v):
        if k == "stop_req":
            self.rx44.stop_req = v
            if self.rx5 is not None:
                self.rx5.stop_req = v
        else:
            setattr(self.rx44, k, v)

    def set_active(self, active):
        self.__dict__["active"] = set(active)
        if self.rx5 is not None:
            self.rx5.enabled = "TNG5" in self.active
            if not self.rx5.enabled:
                self.rx5._reset_state()

    def start(self):
        self.rx44.start()
        if self.rx5 is not None:
            self.rx5.start()

    def stop(self):
        self.rx44.stop()
        if self.rx5 is not None:
            self.rx5.stop()

    def feed(self, x):
        self.rx44.feed(x)
        if self.rx5 is not None:
            self.rx5.feed(x)

    def step(self):
        self.rx44.step()
        if self.rx5 is not None:
            self.rx5.step()

    def pause(self, on, holdoff_s=0.3):
        self.rx44.pause(on, holdoff_s)
        if self.rx5 is not None:
            self.rx5.pause(on, holdoff_s)

    @property
    def paused(self):
        return self.rx44.paused


# ====================================================================== 패킷 분석 · 해부 (analysis.PacketAnalyzer 가 부른다)
def analyze(res, job, last_tx_text=None):
    """TNG5 수신 결과 → 분석 패널 · 타임라인 · 해부 자료. 정답 비트 = CRC 통과 구간 재부호화 (루프백이면 보낸 텍스트)"""
    from tngpkt.engine import load_model  # noqa: F401  (모델 불필요: 재부호화만)
    from tngpkt.stream_ui import packet_timeline
    from tngpkt.viz import llr_hist
    from tngpkt.framing import crc16_ccitt
    info = res.info or {}
    m = (res.viz or {}).get("msg") or {}
    llr = info.get("llr")
    segs = sorted(info.get("segments") or [], key=lambda r: r["index"])
    out = {"kind": "rx", "id": job.get("id"), "ok": res.ok, "text": res.text, "label": job.get("label", ""),
           "df": res.df_hz, "snr": res.snr_db, "utc": job.get("utc"), "mode": res.mode, "reason": res.reason,
           "format": NAME, "segments_ok": info.get("segments_ok"), "segments_used": info.get("segments_used"),
           "partial": info.get("partial"), "margin": None, "rho_a": None}
    if llr is None or not segs:
        return out
    nF, nseg = llr.shape[0], len(segs)
    q = tng5.interleave_map(nseg * SC)
    flat = llr.reshape(-1)
    truth = np.zeros(nF * 96, np.int64)
    mask = np.zeros(nF * 96, bool)
    if job.get("loopback") and last_tx_text:
        fr, _ = tng5.encode_bits(tng5.normalize(last_tx_text)[0])
        if fr.shape[0] == nF:
            truth, mask = fr.reshape(-1).astype(np.int64), np.ones(nF * 96, bool)
    if not mask.all():
        for r in segs:
            if r["ok"]:
                k = r["index"]
                idx = q[k * SC:(k + 1) * SC]
                truth[idx] = tng5.seg_codeword(r["data"])
                mask[idx] = True
    err = ((flat > 0) != (truth > 0)) & mask
    ferr, fknown = err.reshape(nF, 96).sum(1), mask.reshape(nF, 96).sum(1)
    statuses = [("unk", "") if fknown[i] == 0 else (("fixed", str(int(ferr[i]))) if ferr[i] else ("ok", ""))
                for i in range(nF)]
    out["frame_err"] = [None if fknown[i] == 0 else int(ferr[i]) for i in range(nF)]
    out["frame_llr"] = np.abs(llr).mean(axis=1)
    out["statuses"] = statuses
    out["llr"] = llr_hist(flat[mask], truth[mask]) if mask.any() else llr_hist(flat)
    out["fec_total"] = int(mask.sum())
    out["fec_fixed"] = int(err.sum()) if mask.any() else None
    out["ber"] = float(err.sum() / mask.sum()) if mask.any() else None
    plan = Plan5(nF, nseg)
    rows = [{"ok": r["ok"], "decided": True} for r in segs]
    pc = timeline_pc()
    cfg = type("C", (), {"frame_len": L, "fs_base": BB_FS})()
    out["timeline"] = packet_timeline(nF, True, True, m.get("b") is not None, pc, cfg, statuses, plan, rows)
    # 해부 자료: 구간별 바이트 · CRC · 글자, base-40 묶음, 부호 비트 위치
    rows5 = []
    for r in segs:
        k, data = r["index"], r["data"]
        idx = q[k * SC:(k + 1) * SC]
        comb = flat[idx].reshape(tng5.REP, -1)
        body = data[tng5.LEN_BYTES:] if k == 0 else data
        groups = []
        for i in range(0, len(body) - 1, 2):
            v = int.from_bytes(body[i:i + 2], "big")
            groups.append((body[i:i + 2].hex().upper(), v,
                           "□□□" if v >= 64000 else tng5.unpack(body[i:i + 2])))
        rows5.append({"k": k, "ok": r["ok"], "data": data, "crc_calc": crc16_ccitt(data),
                      "len_field": int.from_bytes(data[:2], "big") if k == 0 else None,
                      "text": tng5.unpack(body) if r["ok"] else "", "groups": groups,
                      "frames": (int(idx.min() // 96) + 1, int(idx.max() // 96) + 1),
                      "llr_abs": float(np.abs(flat[idx]).mean()),
                      "rep_llr": np.abs(comb).mean(axis=1),                        # 반복 사본 4개 각각의 |LLR|
                      "err": int(err[idx].sum()) if mask[idx].any() else None})
    try:
        out.update(expert(res, llr, truth if mask.all() else None))
    except Exception as ex:                                       # 전문가 그림이 실패해도 해부 · 타임라인은 그대로
        out["expert_err"] = "{}: {}".format(type(ex).__name__, ex)
    out["dissect5"] = {"segments": rows5, "n_frames": nF, "nseg": nseg, "chars": info.get("chars"),
                       "timeline": out["timeline"], "frame_llr": out["frame_llr"], "frame_err": out["frame_err"],
                       "df": res.df_hz, "snr": res.snr_db, "text": res.text or info.get("partial") or "",
                       "has_b": m.get("b") is not None, "coded_pos": [q[k * SC:(k + 1) * SC] for k in range(nseg)]}
    return out


_ENG5 = None


def _eng5():
    """분석 스레드용 TNG5 엔진 (모델 · 템플릿). 앱 엔진과 따로 한 번만"""
    global _ENG5
    if _ENG5 is None:
        _ENG5 = Tng5Engine()
    return _ENG5


def expert(res, llr, truth):
    """
    TNG5 전문가 모듈 자료 (TNG5 모델 · 동기 구조 기준): 성상도 · 아이 · NN 특징 / 위상 궤적 · 코스타스 동기 맵.
    위상 기준 = TNG5 코스타스 A×4 상관 위상. 정답 비트 전부를 알면 (복원 · 루프백) 데이터 보조 등화.
    """
    from tngpkt.viz import constellation, eye as eye_fn, nn_features, data_aided_eq
    from tngpkt.analysis import _eye_from_seg, _trajectory
    from tngpkt.modem5 import _frames_to_baseband
    e = _eng5()
    cfg, info = e.cfg, res.info or {}
    viz = res.viz or {}
    m = viz.get("msg") or {}
    bb = audio_to_baseband(np.asarray(viz["audio"], dtype=np.float64), viz["fs"], cfg)
    a_bb, df = float(m["a"]["start"]), float(m.get("df") or 0.0)
    t = np.arange(len(bb)) / cfg.fs_base
    y = bb * np.exp(-2j * np.pi * df * t)
    nF = llr.shape[0]
    frame0 = int(round(a_bb + e.data0)) + int(info.get("h_off", 0))
    # 위상: A×4 코스타스 조각 상관의 합
    acc = 0j
    for off, tmpl, kind in e.A:
        if kind != "costas":
            continue
        p0 = int(round(a_bb)) + off
        seg = y[p0:p0 + len(tmpl)]
        if len(seg) == len(tmpl):
            acc += np.sum(seg * np.conj(tmpl))
    r = {"bbd": y, "frame0": frame0, "n_data": nF, "phase": float(np.angle(acc)), "bb": bb, "a_start": int(a_bb), "df": df}
    S = cfg.symbols_per_frame
    out = {"const_raw": constellation(r, cfg), "phase_ref": True}
    out["const_frame"] = np.arange(len(out["const_raw"])) // S
    ey = eye_fn(r, cfg)
    if ey is not None:
        out["eye_raw"] = (ey[0], ey[1], ey[2], np.zeros(len(ey[1]), int))
    if truth is not None:
        fr = np.asarray(truth).reshape(nF, 96).astype(np.float32)
        with e._lock:
            tx_bb = _frames_to_baseband(fr, tng5.GUARD5, e.model, cfg, e.device)
        da = data_aided_eq(r, tx_bb, {"data_start": 0}, cfg)
        if da is not None:
            c = da["centers"]
            out["const_eq"] = da["eq_seg"][c]
            out["const_ideal"] = da["tx_seg"][c]
            out["evm_raw"], out["evm_eq"], out["drift_hz"] = da["evm_raw_db"], da["evm_eq_db"], da["drift_hz"]
            out["eye"] = _eye_from_seg(da["eq_seg"], cfg, nF)
            out["eye_raw"] = _eye_from_seg(da["rx_seg"], cfg, nF)
            out["traj"] = [_trajectory(da["eq_seg"], cfg, f + 1) for f in range(nF)]
    with e._lock:
        out["feat"] = nn_features(r, e.model.rx, cfg, e.device, truth)
    if out.get("feat") is not None:
        out["feat_frame"] = np.arange(len(out["feat"]["xy"])) // S
    out["sync5"] = sync_map(e, y, a_bb, df, res.snr_db)
    return out


def sync_map(e, y, a_bb, df, snr=None, span_s=0.1):
    """
    TNG5 코스타스 A 동기 맵 (48 ms · 41.67 Hz 배열 × 4벌): 시간 × 주파수 오차 평균 ρ² (벌별 정규화 상관의 평균).
    SyncMapPanel 스냅샷 형식 {'rho' (주파수, 시간), 'f', 't', 'det', 'check'}
    """
    chunks = [c for c in e.A if c[2] == "costas"]
    n7 = len(chunks[0][1])
    starts = np.arange(int(a_bb - span_s * BB_FS), int(a_bb + span_s * BB_FS), 4)
    yt = torch.as_tensor(y.astype(np.complex64), device=e.device)[None]
    with e._lock:
        _, raw = tng5.metric(yt, chunks, starts, e.device)
    rho = (raw[0].cpu().numpy() / (len(chunks) * n7)).T               # (주파수, 시간)
    f = tng5.FREQ[tng5.BAND]
    o = np.argsort(f)
    rho, f = rho[o], f[o]
    t = (starts - a_bb) / BB_FS
    j0 = int(np.argmin(np.abs(t)))
    i0 = int(np.argmin(np.abs(f - (df if np.isfinite(df) else 0.0))))
    return {"rho": rho, "f": f, "t": t, "utc": time.time(), "mode": NAME,
            "det": {"df": 0.0, "rho": float(rho[:, j0].max())},
            "check": {"refine_rho": float(rho[i0, j0]), "map_at_det": float(rho[i0, j0])}}
