"""
TNG5 앱 연결 — 송신 오디오 · 실시간 수신 · 복조 (엔진은 tng5.py 그대로, TNG44 코드는 건드리지 않는다).

    Tng5Engine     모델 stage3_strong.pt, 동기 템플릿, 송신 오디오, 구간 복호
    Tx5Audio       stream_ui.TxAudio 와 같은 모양 (송신 패널 · 송신 모니터 · 송신 해부가 그대로 쓴다)
    Tng5Receiver   realtime_rx.RealtimeReceiver 와 같은 콜백 · 사건 (on_status, on_marker, on_decoded, on_live)
    MultiRx        TNG44 · TNG5 수신기 묶음 (단일 모드 운용: 선택한 모드 하나만)

실시간 수신 (Tng5Receiver) — TNG5 v2 (Welch12 시작 · 3.84 s 마다 중간 동기 M_k · 끝 B×1)
  · 탐색: A = Welch12 3조각 · B 묶음 (가드 + B×1) 템플릿 — tng5.detect (문턱 · 봉우리 · SNR 일치 검사)
  · 확정: 첫 중간 동기 M1 이 예상 위치 ±8 샘플 · ±4 Hz 에서 문턱 (9.21) 과 SNR 일치 (ρ² ≥ 0.25·s/(1+s)) 를 넘으면 확정.
    확인 실패여도 첫 CRC 가 통과하면 표시. 둘 다 없이 끝나면 조용히 폐기 (로그 · 'cancel' 사건만)
  · 확정된 수신 구간 안 ±125 Hz 의 새 A 는 무시 (자기 중복). 미확정 수신 중 새 A 는 더 세면 갈아타고, 약하면 대기 (백업)
    → 지금 수신이 확인 실패하면 백업으로 갈아탐 (링버퍼에서 다시 읽음)
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

from tngpkt import tng5
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
N_SYNC_A = tng5.N12                      # v2 시작 동기 Welch12 길이 (1536 = 0.768 s)
N_SYNC_B = tng5.NB                       # rev 3 B12 길이 (768 = 0.384 s)
N_SYNC = N_SYNC_A                        # (화면 모듈 호환 이름: 시작 동기 길이)
# 신호 소실: TNG5 는 -12 dB 이하에서 |LLR| 이 잡음과 겹친다 (측정: 잡음 2.5 s 평균 최대 0.40,
# 신호 -12 dB Moderate 2.5 s 평균 최소 0.28) → TNG44 (2.5 s) 보다 길게 5 s, 대역 에너지 · 구간 실패 조건 함께
LOSS_T = 26
LOSS_LLR = 0.45
LOSS_E_DB = 1.5
B_LLR_S = 0.6
B_DF_HZ = 6.0                             # A · B 주파수 일치 허용 (v2b: 3 Hz 는 -12 dB 에서 참 B 를 버림, +3.9 Hz 측정)
B_Q_S = 0.15                              # B 위치 = 구간 수로 정해지는 자리 ± 이 값 (클럭 100 ppm × 225 s = 23 ms)
RIVAL_S = 0.3                             # 경쟁 후보 대기 (v2 는 반복 배열이 없어 벌 단위 어긋남 없음 — A/B 교차 정리만)
# 중간 동기 확인
M_T = 9.21                                # 단일 조각 문턱 (Gamma(1) 꼬리 1e-4)
M_FRAC = 0.25                             # SNR 일치: ρ² ≥ M_FRAC · s/(1+s)
M_DF_HZ = 4.0
M_POS = 8                                 # 위치 ± (기저대역 샘플)
SAME_BAND_HZ = 125.0                      # 확정 구간 안 같은 대역 새 A 무시


def seg_count(field):
    """첫 구간 길이 필드 → 구간 수 (0.4.0 / wire rev 4: 필드 = 구간 수)"""
    return tng5.seg_count_field(field)


def frames_for_segs(nseg):
    return int(math.ceil(nseg * SC / tng5.BITS_PER_FRAME))


def b_quantized(off_bb, tol_s=None):
    """첫 데이터 프레임 → B 시작 거리 (기저대역 샘플) 가 어떤 구간 수 n 의 B 자리와 맞는가 (메시지는 구간 단위로만 끝난다)"""
    tol = (B_Q_S if tol_s is None else tol_s) * BB_FS
    return any(abs(off_bb - tng5.b_off(frames_for_segs(n))) <= tol for n in range(1, NSEG_MAX + 1))


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
    """packet_timeline 용: 코스타스 길이 = symbol_ms × n_tones → Welch12 (768 ms). B · 중간 동기는 timeline_v2 가 고친다"""
    import dataclasses
    return dataclasses.replace(tng5.PC, symbol_ms=1000.0 * N_SYNC_A / BB_FS / tng5.PC.n_tones)


def timeline_b_state(tl, state):
    """타임라인 끝 B 칸 (길이 필드로 먼저 복원한 메시지): 'wait' 아직 없음 (칸을 그리지 않음 · 가드는 다른 구간처럼 그대로)
    · 'ok' 인정된 자리에 B · 'missing' 예상 끝 + 2 s 까지 없음"""
    fr = list(tl["frames"])
    ib = max((i for i, x in enumerate(fr) if x[0] in ("B", "B?")), default=None)
    if ib is not None:
        base = fr[ib]
        tl = dict(tl, _b_slot=base[1:3])
        fr.pop(ib)
    elif tl.get("_b_slot") is None:
        return tl
    t, d = tl["_b_slot"]
    if state == "ok":
        fr.append(("B", t, d, "costas", "B"))
    elif state == "missing":
        fr.append(("B?", t, d, "missing", "B 없음"))
    return dict(tl, frames=fr)


def timeline_v2(tl):
    """packet_timeline 결과에 v2 구조 반영: 20프레임마다 중간 동기 M_k (336 ms) 끼우고, B 길이 336 ms"""
    m_ms = 1000.0 * tng5.NM / BB_FS
    b_ms = 1000.0 * N_SYNC_B / BB_FS
    out, shift, nfr = [], 0.0, 0
    for name, t, d, kind, lab in tl["frames"]:
        if name.startswith("F"):
            if nfr > 0 and nfr % tng5.MID_EVERY == 0:
                kq = nfr // tng5.MID_EVERY
                out.append(("M{}".format(kq), t + shift, m_ms, "costas", "M{}".format(kq)))
                shift += m_ms
            nfr += 1
        if name in ("B", "B?"):
            d = b_ms
        out.append((name, t + shift, d, kind, lab))

    def seg_t(t):                                                 # 구간 줄 (명목 위치) 도 같은 이동
        f = int(max(0.0, t - tl["frames"][2][1] - tl["frames"][2][2]) // (1000.0 * L / BB_FS))
        return t + m_ms * int(tng5.mids_before(f))
    segs = [(n, seg_t(t), seg_t(t + d - 1e-6) - seg_t(t) + 1e-6, k, lab) for n, t, d, k, lab in tl["segments"]]
    return dict(tl, frames=out, segments=segs)


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
    return t, owner                                         # rev 4: 구간을 풀면 글자가 정확 (끝 자르기 없음)


def _floor(bb):
    """잡음 바닥 (100 샘플 블록 전력 하위 5% / 0.84), 디지털 무음 블록 제외. 1 s 미만이면 None"""
    n = 100
    Lh = len(bb) // n * n
    if Lh < 20 * n:
        return None
    p = (np.abs(bb[:Lh]) ** 2).reshape(-1, n).mean(1)
    v = p[p > max(float(p.max()), 1e-30) * 1e-6]
    if len(v) < 20:
        return None
    return float(np.quantile(v, 0.05)) / 0.84


# ====================================================================== 엔진
class Tng5Engine:
    def __init__(self, device=None):
        dev = device or __import__("tngpkt.engine", fromlist=["*"]).APP_DEVICE           # 28부: CPU 만
        self.device = torch.device(dev)
        self.model, self.cfg = load_model(tng5.MODEL, self.device)
        self._lock = threading.Lock()
        _, info, data = tng5.tx_waveform("CQ", self.model, self.cfg, self.device)
        self.A, self.B = tng5.templates(data)
        self.B_v2 = tng5.templates_b_v2(data)                # rev 2 WAV 열기 전용
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
        # v2: 데이터 스트림 번호 → 송신 위치 (중간 동기 구간을 건너뜀). 창이 중간 동기를 걸쳐도 데이터 샘플만 모인다
        j = (f[None, :] * L + np.asarray(OFFS)[:, None] - C).reshape(-1)
        idx = int(round(d0)) + tng5.stream_to_tx(j[:, None] + np.arange(W)[None, :])
        win = y[np.clip(idx, 0, len(y) - 1)]
        with self._lock:
            llr = _rx_windows(win, self.model, self.device)
        fidx = int(round(d0)) + tng5.stream_to_tx(f[:, None] * L + np.arange(L)[None, :])
        fr = y[np.clip(fidx, 0, len(y) - 1)]
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
        data = tng5.whiten(by[:tng5.SEG_BYTES])                  # rev 4: 흰색화 되돌리기
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
                "declared_segments": seg_count(nchars) if nchars is not None else None,
                "chars": len(text) if ok else None,
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
            nF = tng5.nF_from_b(b_bb - d0)
        if nF is None:
            nF = max(1, int(np.searchsorted(tng5.frame_off(np.arange(2000)) + 2 * L, len(bb) - d0)) - 1)
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

    def decode_late(self, audio, fs, max_k=60):
        """
        v2 늦게 켠 수신기 (파일 · 버퍼): 시작 동기를 놓친 녹음에서 중간 동기 M_k (k mod 4 = 배열 종류) 로 합류해 복조.
        중간 동기 위치 + 종류 → 번호 후보 k (k ≡ 종류 mod 4) 마다 첫 데이터 프레임 위치를 정하고, 녹음 전 프레임은 LLR 0 (지움) 으로
        첫 구간 (길이 필드) 을 시험 → CRC 통과하는 k 로 전체 복조. 반환 DecodeResult (info['late_k'], info['erased_frames'])
        """
        bb, kk = self.to_bb(audio, fs)
        y = torch.as_tensor(bb.astype(np.complex64), device=self.device)[None]
        mids = []
        with self._lock:
            for i, t in enumerate(tng5.mid_templates()):
                for d in tng5.detect(y, [(0, t, "costas")], self.device):
                    mids.append(dict(d, cls=i))
        if not mids:
            return DecodeResult(False, reason="TNG5 중간 동기 미검출", mode=NAME + " · 합류", info={"format": NAME}, viz={})
        m = max(mids, key=lambda d: d.get("raw", d["metric"]))
        best = None
        for k in range(1, max_k + 1):
            if (k - 1) % 4 != m["cls"]:                          # M_k 템플릿 = [(k-1) % 4]
                continue
            frame0 = m["start"] - tng5.mid_off(k)                 # 첫 데이터 프레임 (bb 번호, 음수면 녹음 전)
            P = int(max(0, -frame0) + 4 * L)
            bbp = np.concatenate([np.zeros(P, bb.dtype), bb])
            f0 = frame0 + P
            nF = max(1, int(np.searchsorted(tng5.frame_off(np.arange(2000)) + 2 * L, len(bbp) - f0)) - 1)
            nF = min(nF, frames_for_segs(NSEG_MAX))
            g, _ = self.llr_frames(bbp, f0, m["df"], 0, nF)
            first = int(np.searchsorted(tng5.frame_off(np.arange(nF + 1)) + f0, P + 64))   # 녹음 시작 뒤 첫 온전한 프레임
            g[:, :first] = 0.0
            h = int(np.argmax(np.abs(g[:, first:]).mean(axis=(1, 2)))) if first < nF else 0
            flat = g[h].reshape(-1)
            for n in range(1, NSEG_MAX + 1):
                if frames_for_segs(n) > nF + 2:
                    break
                fl = flat[:n * SC] if len(flat) >= n * SC else np.concatenate([flat, np.zeros(n * SC - len(flat), np.float32)])
                r0 = self.decode_one(fl, 0, self.qmap(n))
                if not r0["ok"]:
                    continue
                nc = int.from_bytes(r0["data"][:tng5.LEN_BYTES], "big")
                if seg_count(nc) != n:
                    continue
                segs = self.decode_segments(fl, n)
                nok = sum(r["ok"] for r in segs)
                if best is None or nok > best[0]:
                    best = (nok, k, first, segs, nc, g[h][:frames_for_segs(n)])
                break
            if best is not None and best[0] == len(best[3]):
                break
        if best is None:
            return DecodeResult(False, reason="TNG5 합류 실패 (중간 동기 {} 종류 {}, 길이 필드 복원 안 됨)".format(
                len(mids), m["cls"]), mode=NAME + " · 합류", info={"format": NAME}, viz={})
        nok, k, first, segs, nc, llr = best
        r = self.result(audio, fs, None, m["df"], llr, segs, nc, "중간 동기 합류 (M{})".format(k))
        r.info["late_k"], r.info["erased_frames"] = k, first
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
        # 가장 센 A (파일 한 개 = 보통 메시지 한 개). B: A 와 주파수가 맞고 구간 경계 자리, 그중 가장 센 것
        a = max(dA, key=lambda d: d.get("raw", d["metric"]))
        bs = [d for d in dB if d["start"] > a["start"] + self.data0 + tng5.b_off(frames_for_segs(1)) - L // 2
              and abs(d["df"] - a["df"]) <= B_DF_HZ and b_quantized(d["start"] - a["start"] - self.data0)]
        b = max(bs, key=lambda d: d.get("raw", d["metric"])) if bs else None
        # 길이 필드 먼저, 실패하면 B 로 구간 수를 정해 다시 (송신 중지 판정)
        r = self.decode_at(audio, fs, a["start"], a["df"], why="코스타스 A · 길이 필드")
        if b is None:                                   # B12 없음: rev 2 파일 (끝 동기 B7) 인지 옛 B 로 확인
            with self._lock:
                d2 = tng5.detect(torch.as_tensor(bb.astype(np.complex64), device=self.device)[None], self.B_v2, self.device)
            bs = [d for d in d2 if d["start"] > a["start"] + self.data0 + tng5.b_off(frames_for_segs(1)) - L // 2
                  and abs(d["df"] - a["df"]) <= B_DF_HZ and b_quantized(d["start"] - a["start"] - self.data0)]
            if not bs:
                return r
            b = max(bs, key=lambda d: d.get("raw", d["metric"]))
            if not r.ok:
                r = self.decode_at(audio, fs, a["start"], a["df"], b_bb=b["start"], why="코스타스 A+B (rev 2 B7)")
            r.info = dict(r.info or {}, format_version="2")
            return r
        if r.ok:
            return r
        return self.decode_at(audio, fs, a["start"], a["df"], b_bb=b["start"], why="코스타스 A+B")

    def decode_file(self, audio, fs):
        """WAV 열기: v2 로 복조, 안 되면 v1 (A×4 · B×4) 파일로 보고 v1 엔진으로 (실시간 수신은 v2 만)"""
        r = self.decode(audio, fs)
        if r.ok:
            return r
        try:
            r1 = _eng_v1().decode(audio, fs)
        except Exception as ex:
            r.info = dict(r.info or {}, v1_err="{}: {}".format(type(ex).__name__, ex))
            return r
        if r1.ok or (r1.info or {}).get("segments_ok", 0) > (r.info or {}).get("segments_ok", 0):
            r1.mode = "{} v1 · {}".format(NAME, r1.mode.split(" · ", 1)[-1])
            r1.info["format_version"] = "1"
            return r1
        return r


_ENG_V1 = None


def _eng_v1():
    """v1 파일 복조용 엔진 (tng5_v1_app, 처음 쓸 때 한 번)"""
    global _ENG_V1
    if _ENG_V1 is None:
        from tngpkt import tng5_v1_app
        _ENG_V1 = tng5_v1_app.Tng5Engine()
    return _ENG_V1


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
        self.t_b_len = N_SYNC_B / fsb
        self.dur = len(self.audio) / self.fs
        self.data_bb = data

    def t_frame(self, f):
        """프레임 f 시작 시각 (중간 동기 포함)"""
        return self.frame_t0 + float(tng5.frame_off(f)) / self._eng.cfg.fs_base

    def progress(self, t):
        # 시각 → 프레임 (중간 동기 구간은 앞 프레임에 머묾)
        off = (t - self.frame_t0) * self._eng.cfg.fs_base
        per = tng5.MID_EVERY * L + tng5.NM
        kq = int(max(off, 0) // per)
        u = kq * tng5.MID_EVERY + min(off - kq * per, tng5.MID_EVERY * L) / L
        done = int(np.clip(np.floor(u), 0, self.plan.n))
        started = int(np.clip(np.ceil(u), 0, self.plan.n))
        return done, started, self.plan.seg_state_tx(done, started)

    def _seg_n(self):
        """구간별 글자 수 (rev 4: 가변 길이 — 구간을 풀어 셈)"""
        if getattr(self, "_segn", None) is None:
            self._segn = [len(t) for t in tng5.seg_texts(self.segs)]
        return self._segn

    def char_segments(self):
        """글자마다 (위치, 구간, 구간)"""
        out, i = [], 0
        for k, n in enumerate(self._seg_n()):
            for _ in range(n):
                out.append((i, k, k))
                i += 1
        return out[:len(self.text)]

    def stopped(self, engine, t_now, margin_s=0.35):
        """중지: t_now + margin 뒤 첫 프레임 경계에서 잘린 메시지로. 반환 (새 Tx5Audio, 이어 붙일 샘플) 또는 (None, None)"""
        tt = t_now + margin_s
        f_sw = 0
        while f_sw < self.plan.n and self.t_frame(f_sw) < tt:
            f_sw += 1
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
        t_sw = self.t_frame(f_sw) - 0.08
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
        full = tng5.whiten(data) + crc16_ccitt(data).to_bytes(2, "big")      # 보내는 바이트 (흰색화 뒤)
        kind = ["len" if (k == 0 and i < 2) else ("crc" if i >= len(data) else "body") for i in range(len(full))]
        bits = np.unpackbits(np.frombuffer(full, np.uint8)).astype(np.int64)
        coded = conv_encode(np.concatenate([bits, np.zeros(tng5.TAIL, np.int64)]), tng5.RATE_N)
        pos = self.plan.q[k * SC:(k + 1) * SC]
        return full, kind, bits, coded, pos

    def seg_chars(self, k):
        body = self.segs[k][tng5.LEN_BYTES:] if k == 0 else self.segs[k]
        return tng5.unpack(body)

    def chars_done(self, sent):
        return min(len(self.text), sum(self._seg_n()[:max(0, sent)]))


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
        self.m_state = "wait"                                     # 중간 동기 확인: wait → ok / fail
        self.crc_ok = False                                       # 첫 CRC 통과 (확인 실패여도 표시)
        self.m_info = None
        self.joined = False                                       # 중간 동기 M 으로 합류 (CRC 로만 확정)
        self.any_seg = False                                      # 길이 모름 + 첫 구간 없음: 최대 길이 맵 (NSEG_MAX) 으로 구간마다 CRC
        self.real_from = 0                                        # 이 프레임부터 실제 받은 소리 (앞은 버퍼 밖 → LLR 0)
        self.extra = 0                                            # 쌍 확인 가설: 프레임 0 을 j = jmax 로 당긴 만큼 (길이 모를 때 받을 프레임 상한에 더함)

    @property
    def shown(self):
        return self.m_state == "ok" or self.crc_ok

    def m_due(self):
        """첫 중간 동기 M1 이 버퍼에 다 들어오는 입력 번호"""
        return self.d0_abs + (tng5.mid_off(1) + 2 * tng5.NM + 2 * M_POS + 64) * self.k     # 상관 창 뒤 이웃 창 (metric 분모) 까지

    def _windows(self, ring, f_from, f_to):
        eng, k = self.eng, self.k
        W, C = eng.cfg.win_len, eng.cfg.ctx_len
        need_end = self.d0_abs + float(tng5.stream_to_tx(f_to * L + (W - C) + 8 + 64)) * k
        if ring.count < need_end:
            return None
        x0 = int(self.d0_abs + float(tng5.stream_to_tx(f_from * L - C - 8 - 160)) * k)
        s0, raw = ring.read(x0, int(need_end))
        s_real = s0
        if self.joined and s0 > x0:                               # 합류: 수신기 켜기 전 부분은 무음으로 채우고 LLR 은 0 (지움)
            n_ = int(need_end) - x0
            raw = np.concatenate([np.zeros(n_ - len(raw), np.float32), raw.astype(np.float32)])[:n_]
            s0 = x0
        if len(raw) < int((need_end - x0) * 0.98):
            return None
        bb, kk = eng.to_bb(raw, self.fs)
        d0 = (self.d0_abs - s0) / kk
        g, ratio = eng.llr_frames(bb, d0, self.df, f_from, f_to)
        if s_real > x0:
            f_ = np.arange(f_from, f_to)
            start_ = self.d0_abs + tng5.stream_to_tx(f_ * L - C) * k           # 프레임 창 시작 (입력 절대 번호)
            g = g.copy()
            g[:, start_ < s_real] = 0.0
        return g, ratio

    def first_real(self, ring_first):
        """창 시작이 버퍼 첫 샘플 이후인 첫 프레임 (그 앞은 버퍼 밖 → LLR 0)"""
        C = self.eng.cfg.ctx_len
        f = np.arange(frames_for_segs(NSEG_MAX))
        st = self.d0_abs + tng5.stream_to_tx(f * L - C - 8 - 160) * self.k
        return int(np.argmax(st >= ring_first)) if (st >= ring_first).any() else len(f)

    def prefill(self, ring_first):
        """합류: 창 시작이 버퍼 첫 샘플보다 앞인 프레임은 NN 없이 LLR 0 으로 채운다 (최대 길이 가설의 앞부분)"""
        n0 = self.first_real(ring_first)
        z = np.zeros((len(OFFS), 96), np.float32)
        self.g = [z] * n0
        self.band = [0.0] * n0
        self.fed = self.real_from = n0

    def best(self):
        G = np.stack(self.g, axis=1)                             # (H, n, 96)
        h = int(np.argmax(np.abs(G).mean(axis=(1, 2))))
        return G[h]

    def end_abs(self):
        """길이로 끝을 알 때 코스타스 B 예상 시작 (입력 절대 번호)"""
        return self.d0_abs + float(tng5.b_off(self.n_true)) * self.k

    def step(self, ring, max_frames=8, budget=None):
        """budget (28부, 합류 가설만): 수신기의 한 주기 몫 {'nn': 프레임, 'vit': 비터비}. 다 쓰면 이번 주기엔 여기까지, 다음 주기에 이어서 (같은 순서)"""
        if self.done:
            return []
        if budget is not None:
            max_frames = min(max_frames, budget["nn"])
            if max_frames <= 0 and self.fed < (self.n_true if self.n_true is not None else frames_for_segs(NSEG_MAX) + self.extra):
                self.pending = True
                return []
        n_to = self.n_true if self.n_true is not None else frames_for_segs(NSEG_MAX) + self.extra
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
        if budget is not None:
            budget["nn"] -= g.shape[1]
        for i in range(g.shape[1]):
            self.g.append(g[:, i])
            self.band.append(float(ratio[i]))
        self.fed += g.shape[1]
        Gall = np.stack(self.g, axis=1)                          # (H, n, 96)
        h_best = int(np.argmax(np.abs(Gall).mean(axis=(1, 2))))
        G = Gall[h_best]
        self.llr_lv = list(np.abs(G).mean(axis=1))
        flat = G.reshape(-1)
        new = []
        eng = self.eng
        if self.nseg is None:                                     # 길이 모름: 후보 구간 수마다 첫 구간 시험
            tried_h = self.__dict__.setdefault("tried_h", {})
            for n in range(1, NSEG_MAX + 1):
                if n in self.tried or self.fed < eng.need0[n]:
                    continue
                if tried_h.get(n) == h_best:
                    self.tried.add(n)
                    continue
                if budget is not None:
                    if budget["vit"] <= 0:
                        self.pending = True                       # 남은 길이 후보는 다음 주기에 (tried 에 안 넣음)
                        break
                    budget["vit"] -= 1
                self.tried.add(n)
                if tried_h.get(n) == h_best:
                    continue                                      # 27부: 같은 정렬 가설로 이미 시험 (첫 구간 비트는 need0[n] 앞 프레임뿐 → 결과 같음)
                tried_h[n] = h_best
                r = eng.decode_one(flat, 0, eng.qmap(n))
                if r["ok"]:
                    nc = int.from_bytes(r["data"][:tng5.LEN_BYTES], "big")
                    if seg_count(nc) == n:
                        self.nchars, self.nseg, self.n_true = nc, n, frames_for_segs(n)
                        if self.any_seg:
                            self.segments, self.any_seg = {}, False       # 길이를 알았다 → 그 길이의 맵으로 다시
                        break
        if self.nseg is None and self.any_seg:                    # 길이 모름: 최대 길이 맵 (끝 가까운 구간은 끝에서 B 로 다시)
            qx = eng.qmap(NSEG_MAX)
            for kk in range(NSEG_MAX):
                if kk in self.segments:
                    continue
                if self.fed < eng.need(NSEG_MAX, kk) + 1:
                    break
                r = eng.decode_one(flat, kk, qx)
                self.segments[kk] = r
                new.append(r)
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
        band = self.band[self.real_from:]                    # 합류: 버퍼 밖 (0 으로 채운) 프레임은 빼고
        if fast_lost(band):                                  # 강신호 갑자기 끊김: 약 1 s
            self.fast_loss = True
            return True
        if len(band) < LOSS_T:
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
    DET_DEV = "cpu"            # 27부: 실시간 동기 탐색 (A · B · M) 장치. NN 복조는 엔진 장치 (GPU 있으면 GPU) 그대로
    REUSE_LIVE = True          # 27부: 길이 필드로 끝낸 메시지 결과 = 실시간 LLR · 구간 재사용 (False = 예전처럼 오디오에서 다시 복조)
    # 가짜 B: B 뒤 0.6 s 평균 |LLR| (가설 최대) 이 이 값 이상이면 신호 계속. 측정: 잡음 최대 0.69 → 진짜 B 는 버리지 않음.
    # 신호 중앙 -8 dB 2.56 · -12 dB 0.86 · -16 dB 0.39 → 약한 신호의 가짜 B 는 못 거른다 (그때는 B 인정, 이전과 같음)
    LLR_T = 1.0
    MAX_S = 225.0             # 최대 메시지 (NSEG_MAX 구간)

    def __init__(self, eng5, fs, on_status, on_marker, on_decoded, sync=False):
        self.eng, self.fs, self.sync = eng5, float(fs), sync
        self.on_status, self.on_marker, self.on_decoded = on_status, on_marker, on_decoded
        self.on_live = None
        self.on_heartbeat = None
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
        self._mids, self._m_seen = None, []      # 중간 동기 M 4종 템플릿 · 합류에 쓴 M 위치
        self._hyps = []                          # M 합류 가설 (Live5, 대기 중에만)
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
        self._backup = None
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
                self.on_status("TNG5 수신 오류 · {}: {}".format(type(ex).__name__, ex), True)
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
    HYP_NN_BUDGET = 12         # 28부: 합류 가설 NN 프레임 한 주기 상한 (가설 전체 합, 프레임 하나 약 7 ms). 처음 따라잡을 때 5개 × 8 프레임 → 이어서
    VIT_BUDGET = 20            # 28부: 합류 가설 CRC 시험 (비터비, 쌍 확인 · 첫 구간 합) 한 주기 상한 (한 번 약 7 ms → 약 0.2 s). 늦게 켬 합류 순간 789회 5.6 s 이던 것

    def _step(self):
        now = self.ring.count
        fs = self.fs
        self._hyp_budget = {"nn": self.HYP_NN_BUDGET, "vit": self.VIT_BUDGET}
        if self.pending is not None and now >= self.pending[0]:
            p, self.pending = self.pending, None
            self._final(*p[1:])
        if self.b_hold is not None:
            self._check_b_hold()
        lv = self.live
        if (lv is not None and self.state == "rx" and lv.m_state == "fail" and not lv.crc_ok
                and lv.fed >= max(self.eng.need0.values()) + 5):
            # 확인 실패 + 모든 길이 후보의 첫 구간을 시험했는데 CRC 없음 → 가짜로 보고 폐기 (대기 A 있으면 갈아탐)
            b = self._backup
            self._drop_silent("중간 동기 확인 실패 · 첫 CRC 없음 ({} 프레임)".format(lv.fed))
            if b is not None:
                self._start_live(b[0], b[1], quiet=True)
            lv = self.live
        if lv is not None and self.state in ("rx", "decoding") and lv.m_state == "wait" and now >= lv.m_due():
            self._check_mid(lv)
            lv = self.live                                        # 확인 실패 → 백업 A 로 갈아탔을 수 있다
        if lv is not None and not lv.done and self.state == "rx":
            had = lv.n_true
            new = lv.step(self.ring)
            if not lv.crc_ok and any(r["ok"] for r in (new or [])):
                self._crc_first(lv)
            if lv.done and lv.end_reason == "length":
                segs = [lv.segments[k] for k in sorted(lv.segments)]
                if not lv.crc_ok and any(r["ok"] for r in segs):
                    self._crc_first(lv)
                self._ev({"type": "final_segments", "segs": segs, "fed": lv.fed, "length": lv.nchars})
                # B 확정 지연 (B 길이 + 경쟁 대기 + 가장자리 + 탐색 주기) 만큼 기다린다 — B 가 오면 _on_b 가 바로 끝냄
                # 길이 필드로 끝을 알면 바로 복조 (B 를 기다려도 결과가 같다: results/v2app/_bwait.py, 19/19 동일).
                # 예상 자리의 B 는 끝낸 뒤 'b_confirm' 사건으로 확인 표시만 더한다
                end = int(lv.end_abs() + 0.35 * fs)
                self.pending = (end, "length")
                self.state = "decoding"
                self.expect_b_until = int(lv.end_abs() + N_SYNC_B * self.k + 2.0 * fs)
                self._exp_b = int(lv.end_abs())
                self._b_wait_until = int(lv.end_abs() + N_SYNC_B * self.k + 2.0 * fs)     # 예상 B 끝 + 2 s
                if lv.shown:
                    self._protect_until(self.a["abs"] - 0.3 * fs, lv.end_abs() + (N_SYNC_B + 384) * self.k + 0.5 * fs,
                                        self.a["df"])
                self.on_status("TNG5 길이 필드로 메시지 끝 확인 · 복조 중…", False)
            elif lv.done and lv.end_reason == "loss":
                if new:
                    self._ev({"type": "segments", "segs": new, "fed": lv.fed})
                self._ev({"type": "lost", "fed": lv.fed})
                self.on_status("TNG5 신호 소실 · 받은 부분까지 복조", True)
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
        self._skip_hyps = False
        if getattr(self, "_b_defer", None) is not None:
            self._resolve_b_defer()
        if self.M_JOIN and not self._skip_hyps:
            self._step_hyps()
        bw_ = getattr(self, "_b_wait_until", None)
        if bw_ is not None and now > bw_ and self.pending is None:
            self._b_wait_until = None                            # 길이 필드로 끝낸 메시지의 B 가 대기 시간 안에 안 옴
            self._ev({"type": "b_timeout", "abs": getattr(self, "_exp_b", None)})
        if self.state == "rx" and self.pending is None and now - self.a["abs"] > int((self.MAX_S + 1.0) * fs):
            self._final("maxlen")

    # ---------------------------------------------------------------- 중간 동기 확인 (v2)
    def _say(self, msg, err=False):
        """수신 상태 문구: 확정 (첫 CRC · M1) 뒤에만 화면에, 그 전은 로그 ('note' 사건) 로만"""
        lv = self.live
        if lv is not None and lv.shown:
            self.on_status(msg, err)
        else:
            self._ev({"type": "note", "msg": msg})

    def _crc_first(self, lv):
        lv.crc_ok = True
        if lv.m_state != "ok":
            self._ev({"type": "confirm", "by": "crc", "m": lv.m_state})
            self.on_status("TNG5 수신 확정 · 첫 CRC 통과{}".format(" (중간 동기 확인 실패였음)" if lv.m_state == "fail" else ""), False)

    def _check_mid(self, lv):
        """첫 중간 동기 M1: 예상 위치 ±M_POS · 주파수 ±M_DF_HZ 에서 문턱 + SNR 일치"""
        fs, k, eng = self.fs, self.k, self.eng
        if getattr(self, "_mid_t", None) is None:
            self._mid_t = tng5.mid_templates()
        tm = self._mid_t[0]
        m_abs = lv.d0_abs + tng5.mid_off(1) * k
        lo = int(lv.a_abs - 3.0 * fs)
        s0, raw = self.ring.read(max(lo, 0), int(m_abs + (2 * tng5.NM + 2 * M_POS + 64) * k))
        ok, v, rho, need = False, 0.0, 0.0, 0.0
        if len(raw) > int(2.0 * fs):
            bb, kk = eng.to_bb(raw, fs)
            p = int(round((m_abs - s0) / kk))
            st = np.arange(p - M_POS, p + M_POS + 1, 2)
            st = st[(st >= len(tm)) & (st < len(bb) - 2 * len(tm) - 1)]
            if len(st):
                y = torch.as_tensor(bb.astype(np.complex64), device=self.DET_DEV)[None]     # 27부: 동기 탐색은 CPU (GPU 는 작은 연산마다 대기, 2.6 s vs 0.8 s)
                with eng._lock:
                    _, r = tng5.metric(y, [(0, tm, "costas")], st, self.DET_DEV)
                sel = torch.as_tensor(np.abs(tng5.FREQ[tng5.BAND] - lv.df) <= M_DF_HZ, device=r.device)
                rr = r[0][:, sel]
                v = float(rr.max())
                j = int(rr.max(1).values.argmax())
                a = int(st[j])
                pw = float(np.mean(np.abs(bb[a:a + len(tm)]) ** 2))
                pre = bb[:max(int(round((lv.a_abs - s0) / kk)) - 100, 0)]
                floor = _floor(pre) or _floor(bb)
                sh = max(pw / max(floor, 1e-30) - 1.0, 0.0)
                rho, need = v / len(tm), M_FRAC * sh / (1.0 + sh)
                ok = v >= M_T and rho >= need
        lv.m_info = {"v": v, "rho": rho, "need": need}
        if ok:
            lv.m_state = "ok"
            self._ev({"type": "confirm", "by": "mid", "v": v, "rho": rho, "need": need})
            self.on_status("TNG5 수신 확정 · 중간 동기 M1 확인 (ρ² {:.2f})".format(rho), False)
            self._backup = None
            return
        lv.m_state = "fail"
        self._ev({"type": "m_fail", "v": v, "rho": rho, "need": need, "crc": lv.crc_ok})    # 로그에만
        if lv.crc_ok:
            return
        b = getattr(self, "_backup", None)
        if b is not None and self.state == "rx":
            self._backup = None
            self._drop_silent("중간 동기 확인 실패 · 대기 중 A 로")
            self._start_live(b[0], b[1], quiet=True)

    def _drop_silent(self, why):
        """미확정 수신 조용히 폐기: 복조 결과 없음, 표시 지움, 로그 ('cancel' 사건) 만"""
        a = self.a
        self._ev({"type": "cancel", "reason": "TNG5 미확정 폐기 · " + why})
        if a is not None and a.get("mk_id"):
            self.on_marker({"kind": "A", "abs": a["abs"] + int(N_TONE * self.k), "df": a["df"], "mode": NAME,
                            "id": a["mk_id"], "remove": True, "dur": N_SYNC_A / self.eng.cfg.fs_base})
        self.stats["dropped"] = self.stats.get("dropped", 0) + 1
        self._reset_state()

    def _noise_floor(self, now):
        if self._floor is not None and now - getattr(self, "_floor_at", -10 ** 12) < int(2.0 * self.fs):
            return self._floor                                    # 27부: 소리 시간 2 s 마다 (앱 · 시험 같게, 벽시계 기준이면 느린 PC 에서 더 자주)
        s0, raw = self.ring.read(now - int(10.0 * self.fs), now)
        if len(raw) < int(1.0 * self.fs):
            return None
        bb, _ = self.eng.to_bb(raw, self.fs)
        self._floor = tng5.noise_floor(torch.as_tensor(bb.astype(np.complex64))[None])
        self._floor_t = time.time()
        self._floor_at = now
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
        y = torch.as_tensor(bb.astype(np.complex64), device=self.DET_DEV)[None]     # 27부: 동기 탐색은 CPU (GPU 는 작은 연산마다 대기, 2.6 s vs 0.8 s)
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
            lv_ = self.live
            if (which == "B" and self.state in ("rx", "decoding") and lv_ is not None and lv_.n_true is not None):
                e_ = (lv_.end_abs() - s0) / kk                      # 예상 B 시작 (이 창 기저대역 번호)
                keep_ = np.abs(st - e_) <= self.B_TOL_S * eng.cfg.fs_base
                d0_ = (lv_.d0_abs - s0) / kk                        # 송신 중지: 앞쪽 구간 경계 자리의 이른 B
                for n_ in range(1, lv_.nseg):
                    keep_ |= np.abs(st - (d0_ + tng5.b_off(frames_for_segs(n_)))) <= B_Q_S * eng.cfg.fs_base
                st = st[keep_]
                if len(st) == 0:
                    continue
            with eng._lock:
                ds = tng5.detect(y, chunks, self.DET_DEV, starts=st, floor=floor)
            hits.append((which, ds))
        if self.M_JOIN and self._m_free() and p_hiA > p_lo:
            self._scan_m(y, s0, kk, np.arange(p_lo, p_hiA, tng5.STEP), floor)
        if "A" in last and p_hiA > p_lo:
            self._scan_from = int(s0 + p_hiA * kk)
        # 후보로 모았다가 더 센 경쟁 후보가 더 나올 수 없는 시점에 인정: 같은 송신의 경쟁 후보(반복 배열의 벌 단위
        # 어긋남, A 템플릿이 B×4 에 걸림)는 후보 뒤 3벌 (1.008 s) 안에만 생긴다 → 탐색이 후보 + 3벌 + 0.1 s 를 지나면 확정.
        # 판단 기준 (2 s 안 가장 센 것 · A/B 정리) 은 그대로, 기다리는 시간만 고정 2 s → 경쟁 가능 구간.
        # 후보가 처음 잡히면 바로 워터폴에 임시 표시 (점선), 확정 = 실선, 탈락 = 삭제.
        for which, ds in hits:
            for d in ds:
                pos = int(round(s0 + d["start"] * kk))
                lv_ = self.live
                if (which == "B" and self.state == "rx" and lv_ is not None and lv_.n_true is None
                        and self.a is not None and (not b_quantized((pos - lv_.d0_abs) / self.k)
                                                    or abs(d["df"] - self.a["df"]) > B_DF_HZ)):
                    continue                                        # 길이 모름: 구간 경계 자리 · 같은 주파수의 B 만 후보
                if pos >= self.ignore_before and self._is_tng1(pos, d["df"], which):
                    continue                                        # TNG1 신호에 걸린 A · B (09-26, 워터폴 임시 표시도 없이 로그만)
                if pos >= self.ignore_before:
                    c = dict(d, kind=which, abs=pos, start=pos / kk, cid=self._cid)
                    self._cid += 1
                    self._cand.append(c)
        covered = self._scan_from
        dd = int(tng5.DEDUP_S * self.fs)
        hold = int(RIVAL_S * self.fs)
        nt = int(N_TONE * kk)                                     # B 후보는 A 기준 좌표 (B − 톤) 로 같은 묶음에
        ca = lambda c: c["abs"] - (nt if c["kind"] == "B" else 0)
        span_ = int(N_SYNC_A * kk)
        raw_ = lambda c: c.get("raw", c["metric"])

        def beaten(c):
            return any(x is not c and raw_(x) >= raw_(c) and
                       ((x["kind"] == c["kind"] and abs(x["abs"] - c["abs"]) < dd) or
                        (x["kind"] != c["kind"] and abs(ca(x) - ca(c)) < span_)) for x in self._cand + self._done)
        for c in self._cand:                                      # 임시 표시 = 묶음에서 지금까지 가장 센 후보 하나
            lead = not beaten(c) and c["kind"] == "A"             # B 는 인정될 때만 표시 (_on_detect)
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
        span = int(N_SYNC_A * kk)
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
            if c["kind"] == "B":
                # B 중복 = 인정된 B 기준만 (버린 가짜 B 가 2 s 안의 진짜 B 를 막지 않게)
                win_ = int(tng5.DEDUP_S * self.fs)
                acc_ = getattr(self, "_b_acc", [])
                if any(abs(q - c["abs"]) < win_ for q in acc_):
                    continue
                if self._on_detect("B", c["abs"], c) and getattr(self, "_b_ok", False):
                    self._b_acc = [q for q in acc_ if abs(q - c["abs"]) < int(60 * self.fs)] + [c["abs"]]
                    self.seen["B"].append(c["abs"])
                continue
            if self._dup(c["kind"], c["abs"]):
                if c.get("shown"):
                    self._mark(c, remove=True)
                continue
            self.seen[c["kind"]].append(c["abs"])
            self._on_detect(c["kind"], c["abs"], c)

    TNG1_LIKE_MAX = 0.11       # A · B 후보 자리의 TNG1 닮음 (tng1.likeness) 이 이보다 크면 버림. TNG5 A 자리 ≤0.081 · 잡음 ≤0.048 · TNG1 -10 dB ≥0.13 (측정)

    def _is_tng1(self, pos, df, which):
        """TNG1 이 TNG5 A · B 로 잡히는 오검출 거르기 (09-26 실제 앱: 깨끗한 TNG1 에 A 확정 · M1 확인 ρ² 0.06 으로 수신 확정 → 화면 문구).
        단일 모드 (09-27) 에서도 유지: 대역의 센 TNG1 에 TNG5 가짜 시작이 그대로 생김 (stream_mix S8, 이 거르기 끄면 가짜 시작 1) — 판단 필요.
        잡음 바닥을 최근 10 s 하위 5% 로 잡아 이어지는 TNG1 이 있으면 바닥 = 신호 세기가 되어 SNR 일치 검사가 무력해짐. 기호 길이 ·
        처프가 달라 역처프 160 ms 창의 에너지 집중도로 가름 (공중 형식 변경 없음)"""
        fs = self.fs
        s0, raw = self.ring.read(int(pos - 0.3 * fs), int(pos + 1.0 * fs))
        if len(raw) < int(0.8 * fs):
            return False
        from tngpkt import tng1
        v = tng1.likeness(raw, fs, f0=float(df))
        self.__dict__.setdefault("like_log", []).append((which, round(v, 3)))      # 시험용 (최근 50)
        del self.like_log[:-50]
        if v > self.TNG1_LIKE_MAX:
            self.stats["tng1_veto"] = self.stats.get("tng1_veto", 0) + 1
            self._ev({"type": "note", "msg": "TNG5 {} 후보 버림 · TNG1 신호 (닮음 {:.2f})".format(which, v)})
            return True
        return False

    M_JOIN = True              # 중간 동기 합류 (병렬 가설 · 앞뒤 M 쌍 지지 · 첫 CRC 로만 확정). 09-25: A 지움 7/12 · 늦게 켬 2/6 · 녹음 합류 0
    TRACE_MIN = 1.0            # B 단독: B 앞 데이터 2프레임 |LLR| / B 뒤 같은 계산. 1.5 는 페이딩 진짜 신호 5/24 막음 → 1.0 (3/24, 휘파람 차단 3/4 그대로)

    M_TOP = 3                  # 종류별 2 s 안 봉우리 보관 수 (1 = 예전: 가장 센 것만 → 0.16 s 어긋난 가짜 M 이 진짜를 지움)
    M_DEDUP_S = 0.12           # 그 안에서 같은 봉우리로 보는 폭 (7차 부엽 ±1 심볼 48 ms)
    M_HYP_MAX = 6              # M 하나뿐인 가설 보관 수 (센 순, 09-26 3 → 6: 종류별 상위 M_TOP 보관과 함께). 쌍 확인 가설은 제한 없음. M 템플릿은 시작 동기 · B · 다른 M 에도 어긋나 걸린다

    def _scan_m(self, y, s0, kk, st, floor):
        """
        대기 중 중간 동기 M 4종 탐색 → 합류 가설로 보관 (수신 상태는 바꾸지 않는다).
        가설 = M 종류 i → k = i + 1 (가장 이른 M_k), 프레임 위치 역산한 Live5 (버퍼 앞은 무음). 첫 구간 CRC 가 통과한 가설만 수신으로 올린다
        """
        eng = self.eng
        if self._mids is None:
            self._mids = tng5.mid_templates()
        hits = []
        for i, tm in enumerate(self._mids):
            with eng._lock:
                ds = tng5.detect(y, [(0, tm, "costas")], self.DET_DEV, starts=st, floor=floor,
                                 dedup_s=self.M_DEDUP_S if self.M_TOP > 1 else None)
            hits += [(i, d) for d in ds]
        for i, d in sorted(hits, key=lambda h: -h[1].get("raw", h[1]["metric"])):
            pos = int(round(s0 + d["start"] * kk))
            near = [q for c_, q in self._m_seen if c_ == i and abs(pos - q) < int(tng5.DEDUP_S * self.fs)]
            if pos < self.ignore_before or len(near) >= self.M_TOP or any(abs(pos - q) < int(self.M_DEDUP_S * self.fs) for q in near):
                continue                                            # 중복 = 같은 종류끼리만, 2 s 안 센 순 M_TOP 개 (어긋난 가짜 M 이 진짜 M 을 막지 않게)
            self._m_seen = [(c_, q) for c_, q in self._m_seen if q > pos - int(60 * self.fs)] + [(i, pos)]
            kq = i + 1
            a_abs = int(pos - (eng.data0 + tng5.mid_off(kq)) * self.k)
            if any(abs(a_abs - q) < int(0.5 * self.fs) for q in self._a_hist):
                continue                                            # 이미 A 로 받은 메시지의 M
            if self._own_m(pos, d["df"]):
                continue                                            # 27부: 진행 중 · 끝낸 메시지 자신의 M (다른 k) · 앞부분 가짜 M → 가설 안 만듦 (같은 신호를 두 번 복조하던 CPU)
            if self._support(i, pos, d["df"]):
                continue                                            # 기존 가설의 다음 M (정확한 간격 · 종류 순서) → 지지 +1
            lv = Live5(eng, a_abs, d["df"], self.fs, self.k)
            lv.joined, lv.m_state = True, "fail"
            lv.m_pos, lv.m_k, lv.m_raw, lv.m_sup = pos, kq, d.get("raw", d["metric"]), 1
            lv.prefill(max(self.ring.count - len(self.ring.buf), 0))   # 버퍼 밖 앞부분 = LLR 0 (NN 없이), 소실 · 한도는 실제 받은 프레임부터
            self._hyps.append(lv)
            self._trim_hyps()
            self.stats["m_hyp"] = self.stats.get("m_hyp", 0) + 1
            self._ev({"type": "m_hyp", "abs": pos, "k": kq, "df": d["df"], "tx_abs": a_abs})

    OWN_DF_HZ = 150.0          # 27부: 자기 신호의 가짜 M 은 ±30~66 Hz 어긋나 나옴 (TNG5 대역 445 Hz 안 = 겹치면 어차피 충돌)

    def _own_m(self, pos, df):
        """27부: 막 끝낸 메시지 보호 구간 (_protect, 예상 B 끝까지) 안 · 같은 대역 (±OWN_DF_HZ) M = 그 메시지 자신의 가짜 M → 가설 안 만듦"""
        return any(lo <= pos <= hi and abs(df - zdf) <= self.OWN_DF_HZ for lo, hi, zdf in self._protect)

    M_TOL = 16                 # 앞뒤 M 간격 허용 (기저대역 샘플, 8 ms)

    def _support(self, i, pos, df):
        """새 M (종류 i, 위치 pos) 이 기존 가설의 뒤 M 이면 그 가설 지지 +1. 간격 = mid_off 차 (4.176 s 배수), 종류 = 순서대로 (4 뒤 1 포함)"""
        for h in self._hyps:
            if abs(df - h.df) > M_DF_HZ:
                continue
            for dk in range(1, 5):                                  # 뒤 M 1~4 개 (중간 M 을 놓쳐도)
                k2 = h.m_k + dk
                if (k2 - 1) % 4 != i:
                    continue
                exp = h.m_pos + (tng5.mid_off(k2) - tng5.mid_off(h.m_k)) * self.k
                if abs(pos - exp) <= self.M_TOL * self.k:
                    h.m_sup += 1
                    self._ev({"type": "m_pair", "abs": pos, "k": k2, "first": h.m_pos, "sup": h.m_sup})
                    if h.m_sup == 2 and not getattr(h, "pair", False):
                        self._widen(h)
                    return True
        return False

    def _widen(self, h):
        """
        쌍 확인된 가설 → M 종류 순환 (4개 주기) 모호성 전부: k = k0 + 4j (j = 0..jmax, 최대 메시지 길이까지).
        가장 이른 위치 (j = jmax) 로 프레임을 받고 j 마다 80 프레임씩 밀어 CRC 시험 (_test_pair)
        """
        eng, k = self.eng, self.k
        jmax = max(0, (tng5.n_mids(frames_for_segs(NSEG_MAX)) - h.m_k) // 4)
        kq = h.m_k + 4 * jmax
        a_abs = int(h.m_pos - (eng.data0 + tng5.mid_off(kq)) * k)
        w = Live5(eng, a_abs, h.df, self.fs, k)
        w.joined, w.m_state, w.pair = True, "fail", True
        w.m_pos, w.m_k, w.m_raw, w.m_sup, w.jmax = h.m_pos, h.m_k, h.m_raw, h.m_sup, jmax
        w.extra = tng5.MID_EVERY * 4 * jmax                           # 09-26: 이게 없으면 j < jmax 가설이 1107 프레임 (실제 67) 에서 멈춤 → 10 s 늦게 켬 0/2
        w.tried = set(range(1, NSEG_MAX + 1))                         # Live5 자체 첫 구간 시험은 끄고 _test_pair 가 j 마다
        w.pair_tried = set()
        w.prefill(max(self.ring.count - len(self.ring.buf), 0))
        self._hyps[self._hyps.index(h)] = w
        self._ev({"type": "m_widen", "abs": h.m_pos, "k0": h.m_k, "jmax": jmax, "real_from": w.real_from})

    def _test_pair(self, h):
        """쌍 확인 가설: j 마다 (첫 구간 · 길이 후보) 또는 (아무 구간 · 최대 길이 맵) CRC → 처음 통과한 j 로 가설 고정"""
        eng, k = self.eng, self.k
        h.pair_pending = False
        if not h.g:
            return False
        G = h.best()
        MS = tng5.MID_EVERY * 4
        for j in range(h.jmax + 1):
            off = MS * (h.jmax - j)
            if G.shape[0] - off <= 0:
                continue
            flat = G[off:].reshape(-1)
            fed_j = G.shape[0] - off
            hit = None
            for n in range(1, NSEG_MAX + 1):
                if ("n", j, n) in h.pair_tried or fed_j < eng.need0[n] + 2:
                    continue
                if self._hyp_budget["vit"] <= 0:                              # 28부: 이번 주기 비터비 몫 다 씀 → 다음 주기에 같은 순서로 이어서
                    h.pair_pending = True
                    return False
                self._hyp_budget["vit"] -= 1
                h.pair_tried.add(("n", j, n))
                r = eng.decode_one(flat, 0, eng.qmap(n))
                if r["ok"]:
                    nc = int.from_bytes(r["data"][:tng5.LEN_BYTES], "big")
                    if seg_count(nc) == n:
                        hit = (n, nc, None)
                        break
            if hit is None:
                qx = eng.qmap(NSEG_MAX)
                for kk in range(NSEG_MAX):
                    if fed_j < eng.need(NSEG_MAX, kk) + 2:
                        break
                    if ("s", j, kk) in h.pair_tried:
                        continue
                    if self._hyp_budget["vit"] <= 0:
                        h.pair_pending = True
                        return False
                    self._hyp_budget["vit"] -= 1
                    h.pair_tried.add(("s", j, kk))
                    r = eng.decode_one(flat, kk, qx)
                    if r["ok"]:
                        hit = (None, None, r)
                        break
            if hit is None:
                continue
            # 가설 j 로 고정
            h.g, h.band, h.llr_lv = h.g[off:], h.band[off:], h.llr_lv[off:]
            h.fed -= off
            h.extra = 0
            h.real_from = max(0, h.real_from - off)
            h.d0_abs = h.m_pos - tng5.mid_off(h.m_k + 4 * j) * k
            h.a_abs = int(h.d0_abs - eng.data0 * k)
            h.m_k = h.m_k + 4 * j
            n, nc, r = hit
            if n is not None:
                h.nchars, h.nseg, h.n_true = nc, n, frames_for_segs(n)
            else:
                h.tried = set()                                         # 첫 구간 (길이) 은 계속 시험
                h.any_seg = True
                h.segments = {r["index"]: r}
            self._ev({"type": "m_fix", "abs": h.m_pos, "j": j, "k": h.m_k, "by": "len" if n else "seg", "seg": None if r is None else r["index"]})
            return True
        return False

    B_DEFER = True             # 28부: 대기 중 B + 쌍 확인 가설 → 한 주기에 몰아 받지 않고 나눠서 (늦게 켬 B 순간 1 s → 수백 ms)
    B_CATCHUP_STEPS = 2        # 한 주기 가설 step 수 (8 프레임씩)

    def _resolve_b_defer(self):
        """미룬 B: 가설을 다 받았으면 예전과 같은 판단 (_hyp_b → 합류, 아니면 되짚기). 그 사이 가설이 CRC 로 합류했으면 그 수신의 B 로"""
        pos, d = self._b_defer
        if self.state != "idle" or self.pending is not None:
            self._b_defer = None
            self._on_b(pos, d)                                     # 가설이 먼저 합류 → 진행 중 수신의 B 처리 (예전: 합류 뒤 _on_b)
            return
        if not any(getattr(h, "pair", False) and abs(d["df"] - h.df) <= B_DF_HZ for h in self._hyps):
            self._b_defer = None
            self._b_resolving = True
            try:
                self._on_b(pos, d)                                 # 가설이 사라짐 → 예전 흐름 (되짚기)
            finally:
                self._b_resolving = False
            return
        if not getattr(self, "_b_ready", False):
            self._b_ready = self._hyp_b_catchup(pos, d, self.B_CATCHUP_STEPS)
            return                                                 # 다 받은 주기와 판단 주기를 나눔 (한 주기 몰림 방지)
        self._b_ready = False
        self._b_defer = None
        self._skip_hyps = True                                     # 판단 (전 구간 복호) 하는 주기엔 가설 진행 쉼 (가설 판단은 받은 프레임 수 기준이라 결과 같음)
        if self._hyp_b(pos, d, caught_up=True):
            self._on_b(pos, d)
        else:
            self._b_resolving = self._b_skip_hyp = True
            try:
                self._on_b(pos, d)                                 # _hyp_b 실패 → 되짚기 (예전과 같은 순서, _hyp_b 다시 안 함)
            finally:
                self._b_resolving = self._b_skip_hyp = False

    def _hyp_b_catchup(self, pos, d, steps):
        """28부: 대기 중 B 를 맞춰 볼 쌍 확인 가설을 B 앞 프레임까지 받아 둔다 (한 주기 steps 번까지). 다 받았으면 True"""
        done = True
        for h in [h for h in self._hyps if getattr(h, "pair", False) and abs(d["df"] - h.df) <= B_DF_HZ]:
            for _ in range(steps):
                f_ = h.fed
                h.step(self.ring)
                if h.fed == f_:
                    break
            else:
                f_ = h.fed
                done = False
        return done

    def _hyp_b(self, pos, d, caught_up=False):
        """대기 중 B: 쌍 확인 가설마다 j 후보의 구간 경계 자리와 맞으면 구간 수 → 그 길이 맵으로 전 구간 → CRC 하나라도 → 합류"""
        eng, k, fs = self.eng, self.k, self.fs
        for h in [h for h in self._hyps if getattr(h, "pair", False)]:
            if abs(d["df"] - h.df) > B_DF_HZ:
                continue
            for _ in range(0 if caught_up else 40):                 # B 앞 프레임까지 받아 둔다 (28부: 보통은 _hyp_b_catchup 이 여러 주기에 나눠 미리)
                f_ = h.fed
                h.step(self.ring)
                if h.fed == f_:
                    break
            G = h.best()
            MS = tng5.MID_EVERY * 4
            for j in range(h.jmax + 1):
                d0 = h.m_pos - tng5.mid_off(h.m_k + 4 * j) * k
                n = next((n_ for n_ in range(1, NSEG_MAX + 1)
                          if abs(d0 + tng5.b_off(frames_for_segs(n_)) * k - pos) <= B_Q_S * fs), None)
                if n is None:
                    continue
                off = MS * (h.jmax - j)
                Gj = G[off:off + frames_for_segs(n)]
                if Gj.shape[0] < frames_for_segs(n) - 1:
                    continue
                segs = eng.decode_segments(Gj.reshape(-1)[:n * SC], n)
                if not any(r["ok"] for r in segs):
                    continue
                h.g, h.band, h.llr_lv = h.g[off:], h.band[off:], h.llr_lv[off:]
                h.fed -= off
                h.extra = 0
                h.real_from = max(0, h.real_from - off)
                h.d0_abs, h.m_k = d0, h.m_k + 4 * j
                h.a_abs = int(d0 - eng.data0 * k)
                h.nseg, h.n_true, h.any_seg = n, frames_for_segs(n), False
                h.segments = {r["index"]: r for r in segs}
                self._ev({"type": "m_fix", "abs": h.m_pos, "j": j, "k": h.m_k, "by": "b", "nseg": n})
                self._promote(h)
                for x in self._hyps:
                    if x is not h:
                        self._ev({"type": "m_drop", "abs": x.m_pos, "reason": "other"})
                self._hyps = []
                return True
        return False

    def _trim_hyps(self):
        """쌍 확인 (지지 ≥ 2) 가설은 모두, M 하나뿐인 가설은 센 순 M_HYP_MAX 개 (그 뒤는 로그만 남기고 버림)"""
        pair = [h for h in self._hyps if h.m_sup >= 2]
        one = sorted([h for h in self._hyps if h.m_sup < 2], key=lambda h: -h.m_raw)
        for h in one[self.M_HYP_MAX:]:
            self._ev({"type": "m_drop", "abs": h.m_pos, "reason": "cap"})
        self._hyps = pair + one[:self.M_HYP_MAX]

    def _step_hyps(self):
        """M 합류 가설 진행: 첫 CRC → 수신으로 올림, 모든 길이 후보 시험 뒤 CRC 없음 · 소실 → 조용히 폐기 (로그만)"""
        if not self._hyps:
            return
        if not self._m_free():
            for h in self._hyps:
                self._ev({"type": "m_drop", "abs": h.m_pos, "reason": "busy"})
            self._hyps = []
            return
        lv = self.live
        if lv is not None and self.state == "rx" and not lv.joined and not lv.shown:
            return                                                   # 27부: 미확정 A 수신 중엔 가설을 멈춰 둠 (NN 안 돌림). A 가 확인되면 아래 _m_free 에서
            #   'busy' 로 버리고, 가짜로 버려지면 다음 주기부터 링버퍼 (235 s) 에서 이어 받음 — 합류 판단은 같고 늦어질 뿐
            #   (09-28 진단: 자기 신호 위 가짜 M 가설 3~5개가 A 확인 전 4 s 동안 NN 을 따로 돌려 TNG5 CPU 여유 2.4배)
        own = [h for h in self._hyps if self._own_m(h.m_pos, h.df)]          # 27부: 뒤에 시작한 A 수신 · 끝낸 메시지 자신의 M 가설 → 버림
        for h in own:
            self._ev({"type": "m_drop", "abs": h.m_pos, "reason": "own"})
        self._hyps = [h for h in self._hyps if h not in own]
        keep = []
        for h in self._hyps:
            pair = getattr(h, "pair", False)
            if not pair and h.nseg is None and h.fed - getattr(h, "_retry_at", 0) >= 5:
                h.tried.clear()                                     # 길이 후보 첫 구간을 5프레임마다 다시 시험 (앞부분 지움 → 뒤 프레임이 도움)
                h._retry_at = h.fed
            h.pending = False
            if pair:                                                 # 쌍 확인 가설 (보통 1개) 은 NN 몫 제한 없음 → 늘 따라잡아 두어 B 검출 순간 (_hyp_b) 몰아 받지 않게
                bud = {"nn": 8, "vit": self._hyp_budget["vit"]}
                new = h.step(self.ring, budget=bud)
                self._hyp_budget["vit"] = bud["vit"]
            else:
                new = h.step(self.ring, budget=self._hyp_budget)
            if pair and self._test_pair(h):                          # 쌍 확인 + 어느 구간이든 첫 CRC
                new = []
            segs = list(h.segments.values()) if h.done else (new or [])
            if any(r["ok"] for r in segs) or any(r["ok"] for r in h.segments.values()) or (pair and h.nseg is not None):
                self._promote(h)
                for x in self._hyps:
                    if x is not h:
                        self._ev({"type": "m_drop", "abs": x.m_pos, "reason": "other"})
                self._hyps = []
                return
            lim = (h.real_from + max(self.eng.need(NSEG_MAX, 3), max(self.eng.need0.values())) + 15) if pair \
                else h.real_from + max(self.eng.need0.values()) + 15
            if h.done or (h.fed >= lim and not getattr(h, "pair_pending", False) and not getattr(h, "pending", False)):   # 28부: 시험이 남았으면 다 해 보고 버림 (판단 같게)
                self.stats["m_drop"] = self.stats.get("m_drop", 0) + 1
                self._ev({"type": "m_drop", "abs": h.m_pos, "reason": "no_crc", "fed": h.fed})
                continue
            keep.append(h)
        self._hyps = keep

    def _m_free(self):
        """M 합류 가능: 진행 중 (확정된) 수신 없음. 미확정 A 수신 (가짜 A 일 수 있음) 은 막지 않는다"""
        if self.pending is not None:
            return False
        if self.state == "idle":
            return True
        lv = self.live
        return self.state == "rx" and lv is not None and not lv.shown and not lv.joined

    def _promote(self, h):
        """확인된 M 합류 가설 → 진행 중 수신 (A 로 시작한 것과 같은 경로, 확정 = CRC)"""
        fs = self.fs
        if self.state == "rx":                                    # 미확정 A 수신이 있었다 → 조용히 버리고 합류
            self._drop_silent("M 합류 가설이 먼저 CRC 통과")
        self.stats["m_join"] = self.stats.get("m_join", 0) + 1
        self._ev({"type": "m_join", "abs": h.m_pos, "k": h.m_k, "df": h.df, "tx_abs": h.a_abs})
        self._start_live(h.a_abs, {"df": h.df, "metric": h.m_raw, "raw": h.m_raw}, quiet=True)
        if h.done and h.end_reason == "length":                    # 버퍼 안에서 이미 끝까지 받음 → 마지막 프레임 다시 넣어 정상 끝냄
            h.done, h.end_reason = False, None
            h.fed -= 1
            h.g, h.band = h.g[:h.fed], h.band[:h.fed]
            h.segments = {}
        self.live = h
        self._crc_first(h)
        if h.segments:
            mx = max(h.segments)
            for k_ in range(mx):                                     # 못 받은 앞 구간 = □
                h.segments.setdefault(k_, {"index": k_, "ok": False, "data": bytes(tng5.SEG_BYTES)})
            self._ev({"type": "segments", "segs": [h.segments[k_] for k_ in sorted(h.segments)], "fed": h.fed})
        if h.n_true is not None:
            self._ev({"type": "format", "format": NAME})

    def _data_trace(self, pos, df):
        """B 단독 검출 위치 앞 데이터 흔적: (B 앞 2프레임 |LLR|) / (B 뒤 같은 계산). 링버퍼에서"""
        fs, eng = self.fs, self.eng
        s0, x = self.ring.read(pos - int(1.5 * fs), pos + int(1.6 * fs))
        if len(x) < int(3.0 * fs):
            return None
        bb, kk = eng.to_bb(x, fs)
        b_bb = int(round((pos - s0) / kk))

        def at(d0):
            g, _ = eng.llr_frames(bb, d0, df, 0, 2)
            return float(np.abs(g).mean(axis=(1, 2)).max())
        return at(b_bb - 3 * L) / max(at(b_bb + N_SYNC_B + 400), 1e-6)

    def _mark(self, c, tentative=False, remove=False):
        """워터폴 표시: 임시 (점선) · 삭제. 확정 표시는 _on_detect 가 같은 id 로 보낸다"""
        m = {"kind": c["kind"], "abs": c["abs"] if c["kind"] == "B" else c["abs"] + int(N_TONE * self.k),
             "df": c["df"], "rho": c.get("metric", 0.0), "mode": NAME,
             "dur": (N_SYNC_B if c["kind"] == "B" else N_SYNC_A) / self.eng.cfg.fs_base,
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

    def _protect_until(self, lo, hi, df):
        """끝낸 메시지의 보호 구간 (예상 B 끝까지) — 상태를 지운 뒤에도 그 안 같은 대역 A 를 새 메시지로 열지 않음"""
        self._protect = [z for z in self._protect if z[1] > self.ring.count - 60 * self.fs] + [(int(lo), int(hi), df)]

    def _rho(self, d):
        return float(d.get("metric", 0.0)) / max(len(self.eng.A), 1) / 100.0

    def _on_detect(self, kind, pos, d):
        fs = self.fs
        self.stats[kind] += 1
        dur = (N_SYNC_B if kind == "B" else N_TONE + N_SYNC_A) / self.eng.cfg.fs_base
        mk = {"kind": kind, "abs": pos if kind == "B" else pos + int(N_TONE * self.k), "df": d["df"],
              "rho": d.get("metric", 0.0), "mode": NAME, "dur": (N_SYNC_B if kind == "B" else N_SYNC_A) / self.eng.cfg.fs_base,
              "id": "{}{}".format(NAME, d["cid"]) if "cid" in d else None}
        def ignore(why):
            self._ev({"type": "a_ignored", "abs": pos, "df": d["df"], "reason": why})      # 로그에만
            if mk["id"]:
                self.on_marker(dict(mk, remove=True))
            return dur
        raw_ = d.get("raw", d.get("metric", 0.0))
        if kind == "A" and self.state in ("rx", "decoding") and self.a is not None:
            lv = self.live
            same = abs(d["df"] - self.a["df"]) <= SAME_BAND_HZ
            inside = lv is not None and lv.n_true is not None and pos < lv.end_abs()
            if lv is not None and lv.shown:
                # 확정된 수신 구간 안 같은 대역의 새 A = 자기 중복 (겹친 두 송신은 어차피 함께 복조 못 한다)
                if same and (inside or lv.n_true is None or self.state == "decoding"):
                    return ignore("확정 수신 구간 안 같은 대역")
            elif self.state == "rx":
                # 미확정 수신 중 새 A: 더 센 쪽을 받고 다른 쪽은 백업 (지금 것이 확인 실패하면 갈아탄다)
                if raw_ < self.a.get("raw", 0.0) and not (lv is not None and lv.m_state == "fail"):
                    b = getattr(self, "_backup", None)
                    if b is None or raw_ > b[1].get("raw", 0.0):
                        self._backup = (pos, dict(d, mk_id=mk["id"]))
                    self._ev({"type": "a_backup", "abs": pos, "df": d["df"]})
                    return dur
                old = (self.a["abs"], dict(self._a_d, mk_id=self.a.get("mk_id"))) if getattr(self, "_a_d", None) else None
                if lv is not None and lv.m_state == "fail":
                    old = None                                    # 이미 확인 실패한 것은 대기로 남기지 않는다
                self._drop_silent("더 센 새 A 로 갈아탐")
                self._backup = old
                self.on_marker(mk)
                self._start_live(pos, dict(d, mk_id=mk["id"]))
                return dur
        if kind == "A" and self.state == "idle":
            for lo, hi, zdf in self._protect:
                if lo <= pos <= hi and zdf is not None and abs(zdf - d["df"]) <= SAME_BAND_HZ:
                    return ignore("끝낸 확정 메시지 구간 안 같은 대역")
        if kind == "A":
            self.on_marker(mk)
            if self.state == "rx" and self.pending is None:
                self._final("next_a", end=pos - int(0.3 * fs))
            elif self.state == "decoding" and self.pending is not None:
                p_, self.pending = self.pending, None           # 끝난 메시지 복조를 먼저 (새 A 가 덮어쓰지 않게)
                self._final(*p_[1:])
            self._start_live(pos, dict(d, mk_id=mk["id"]))
        else:
            # B 표시는 인정될 때만 (B×1 은 짧아 데이터 · 다른 모드 속 가짜 B 가 있다). 역방향은 A 를 찾으면 그때
            ok_b = self._on_b(pos, dict(d, _mk=mk))
            self._b_ok = bool(ok_b)
            self.on_marker(mk if ok_b else dict(mk, remove=True))
        return dur

    def _start_live(self, pos, d, quiet=False):
        fs = self.fs
        bk = getattr(self, "_backup", None)
        self.state = "rx"
        self._backup = bk
        self._a_hist = [p for p in self._a_hist if p > pos - int((self.MAX_S + 10) * fs)] + [pos]
        self.a = {"abs": pos, "df": d["df"], "rho": d.get("metric", 0.0), "raw": d.get("raw", 0.0), "mk_id": d.get("mk_id")}
        self._a_d = d
        self.live = Live5(self.eng, pos, d["df"], fs, self.k)
        self._ev({"type": "start", "abs": pos + int(N_TONE * self.k), "tx_abs": pos, "df": d["df"],
                  "rho": d.get("metric", 0.0), "fs": fs})
        if not quiet:
            self._say("TNG5 신호 감지 (주파수 오차 {:+.1f} Hz) · 중간 동기 확인 대기".format(d["df"]), False)

    def _on_b(self, pos, d):
        fs = self.fs
        lv = self.live
        if self.state == "rx" and self.a is not None and abs(d["df"] - self.a["df"]) > B_DF_HZ:
            self._say("TNG5 코스타스 B 무시 · A 와 주파수 불일치 ({:+.1f} Hz)".format(d["df"] - self.a["df"]), True)
            self._ev({"type": "b_ignored", "abs": pos, "df_off": d["df"] - self.a["df"]})
            return False
        if self.state == "rx" and lv is not None and not b_quantized((pos - lv.d0_abs) / self.k):
            self._say("TNG5 코스타스 B 무시 · 구간 경계 자리 아님 (데이터 속 가짜 B)", False)
            self._ev({"type": "b_ignored", "abs": pos, "reason": "quant"})
            return False
        if self.state == "rx" and lv is not None and not d.get("llr_ok"):
            if lv.n_true is not None:
                exp = lv.end_abs()
                if pos < exp - int(self.B_TOL_S * fs) and pos > self.a["abs"] + int((self.eng.data0 + 3 * L) * self.k):
                    self.b_hold = (pos, d)            # 예상보다 이른 B: 송신 중지일 수 있다 → B 뒤 신호 소실이면 인정
                    return False
                if abs(pos - exp) > int(self.B_TOL_S * fs):
                    self._say("TNG5 코스타스 B 무시 · 길이 필드 예상 끝과 {:+.1f} s 차이".format((pos - exp) / fs), True)
                    self._ev({"type": "b_ignored", "abs": pos, "off_s": (pos - exp) / fs})
                    return False
            elif pos > self.a["abs"] + int((self.eng.data0 + 3 * L) * self.k):
                self.b_hold = (pos, d)
                return False
        if self.state == "rx" and pos > self.a["abs"] + int((self.eng.data0 + 3 * L) * self.k):
            self.pending = (pos + int(N_SYNC_B * self.k + self.POST_S * fs), "b", pos)
            self.state = "decoding"
            if lv is not None:
                lv.done = True
            self._ev({"type": "b", "abs": pos, "df": d["df"], "fs": fs})
            self.on_status("TNG5 메시지 끝 감지 · 복조 중…", False)
            return True
        elif (self.state == "decoding" and self.pending is not None and self.pending[1] == "length"
              and self.a is not None and abs(pos - getattr(self, "_exp_b", -10 ** 12)) <= int(self.B_TOL_S * fs)
              and abs(d["df"] - self.a["df"]) <= B_DF_HZ):
            # 길이 필드로 끝을 안 메시지의 진짜 B: 기다리던 복조를 B 와 함께 바로
            self.pending = (pos + int(N_SYNC_B * self.k + self.POST_S * fs), "length_b", pos)
            self._ev({"type": "b", "abs": pos, "df": d["df"], "fs": fs})
            self.on_status("TNG5 메시지 끝 (길이 필드 · 코스타스 B 확인) · 복조 중…", False)
            return True
        elif self.state == "idle" and pos <= self.expect_b_until:
            ok_ = abs(pos - getattr(self, "_exp_b", -10 ** 12)) <= int(self.B_TOL_S * fs)   # 길이로 끝낸 메시지의 B
            if ok_ and getattr(self, "_b_wait_until", None) is not None:
                self._b_wait_until = None
                self._ev({"type": "b_confirm", "abs": pos, "df": d["df"], "off_s": (pos - self._exp_b) / fs})
                self.on_status("TNG5 코스타스 B 확인 · 길이 필드 끝과 {:+.2f} s".format((pos - self._exp_b) / fs), False)
            return ok_
        elif (self.state == "idle" and self.pending is None and self.B_DEFER and not getattr(self, "_b_resolving", False)
              and any(getattr(h, "pair", False) and abs(d["df"] - h.df) <= B_DF_HZ for h in self._hyps)):
            self._b_defer = (pos, d)                               # 28부: 쌍 확인 가설을 B 앞까지 여러 주기에 나눠 받은 뒤 _resolve_b_defer 가 판단
            return False
        elif self.state == "idle" and self.pending is None and not getattr(self, "_b_skip_hyp", False) and self._hyp_b(pos, d):
            return self._on_b(pos, d)                              # 쌍 확인 M 가설 + B 위치 → 길이 → CRC 통과 → 합류 뒤 정상 B 처리
        elif self.state == "idle" and self.pending is None:
            self._ev({"type": "b_back", "abs": pos, "df": d["df"], "fs": fs})
            self._say("TNG5 메시지 끝(B)만 감지 · 시작(A)을 되짚어 찾는 중…", False)
            self._reverse(pos, d)
        return False

    def _check_b_hold(self):
        pos, d = self.b_hold
        lv = self.live
        if self.state != "rx" or lv is None:
            self.b_hold = None
            return
        if lv.n_true is not None and abs(pos - lv.end_abs()) <= int(self.B_TOL_S * self.fs):
            self.b_hold = None
            if self._on_b(pos, d) and d.get("_mk") is not None:
                self.on_marker(d["_mk"])
            return
        off_ = (pos + N_SYNC_B * self.k + 0.05 * self.fs - lv.d0_abs) / self.k
        f0 = int(np.searchsorted(tng5.frame_off(np.arange(2000)), off_))
        nf = max(1, int(round(B_LLR_S * self.eng.cfg.fs_base / L)))
        got = lv._windows(self.ring, f0, f0 + nf)
        if got is None:
            return
        self.b_hold = None
        m = float(np.abs(got[0]).mean(axis=(1, 2)).max())
        if m < self.LLR_T:
            if self._on_b(pos, dict(d, llr_ok=True)) and d.get("_mk") is not None:
                self.on_marker(d["_mk"])
        else:
            self._say("TNG5 코스타스 B 무시 · B 뒤 신호 계속 (|LLR| {:.2f})".format(m), True)
            self._ev({"type": "b_ignored", "abs": pos, "llr": m})

    # ---------------------------------------------------------------- 복조
    def _final(self, why, b_pos=None, end=None):
        lv, a = self.live, self.a
        if lv is not None and lv.joined and not lv.shown:
            self._drop_silent("중간 동기 합류 · CRC 없음 ({})".format(why))
            return
        bk = getattr(self, "_backup", None)
        self._reset_state()
        if a is None:
            return
        fs, k = self.fs, self.k
        shown = lv is None or lv.shown
        if not shown and bk is not None and why in ("loss", "maxlen"):
            # 미확정 수신이 끝났는데 대기 중인 A 가 있다 → 그쪽으로 (링버퍼에서 다시 읽는다)
            self._ev({"type": "cancel", "reason": "TNG5 미확정 폐기 · 대기 중이던 A 로 갈아탐"})
            self._start_live(bk[0], bk[1], quiet=True)
            return
        # 보호 구간: 길이 필드 예상 끝 (B×4 포함) 또는 B 끝 또는 지금까지 (길이 · B 모두 모름)
        hi = self.ring.count
        if lv is not None and lv.n_true is not None:
            hi = max(hi, lv.end_abs() + (N_SYNC_B + 384) * k)
        if b_pos is not None:
            hi = max(hi, b_pos + N_SYNC_B * k)
        if shown:
            self._protect_until(a["abs"] - 0.3 * fs, hi + 0.5 * fs, a["df"])
        start = a["abs"] - int(self.PRE_S * fs)
        end = end if end is not None else (b_pos + int((N_SYNC_B * k) + self.POST_S * fs) if b_pos else self.ring.count)
        s0, x = self.ring.read(start, end)
        if len(x) < int(1.0 * fs):
            return
        a_bb = (a["abs"] - s0) / k
        b_bb = None if b_pos is None else (b_pos - s0) / k
        label = {"b": "코스타스 B", "length": "길이 필드", "length_b": "길이 필드 + 코스타스 B",
                 "loss": "신호 소실", "manual": "수동 중지",
                 "next_a": "B 미검출 · 다음 신호 시작", "maxlen": "B 미검출 · 최대 길이 초과"}.get(why, why)
        nF = lv.n_true if (lv is not None and lv.n_true is not None and (b_pos is None or why == "length_b")) else None
        if nF is None and b_pos is None and lv is not None:
            nF = max(1, lv.fed)
        self.stats["decoded"] += 1

        def work():
            try:
                if lv is not None and lv.joined and lv.g:
                    # 중간 동기 합류: 시작이 버퍼 밖일 수 있어 다시 읽지 않고 받은 프레임 (앞부분 LLR 0) 으로 구간 결정
                    G = lv.best()
                    ns_ = lv.nseg
                    if ns_ is None and b_pos is not None:                # 길이 모름: B 위치 → 구간 수 (구간 경계 자리)
                        ns_ = next((n for n in range(1, NSEG_MAX + 1) if abs(
                            lv.d0_abs + tng5.b_off(frames_for_segs(n)) * k - b_pos) <= self.B_TOL_S * fs), None)
                    if ns_ is not None:
                        nF_ = min(frames_for_segs(ns_), G.shape[0])
                        segs_ = self.eng.decode_segments(G[:nF_].reshape(-1)[:ns_ * SC], ns_)
                    else:
                        nF_ = G.shape[0]
                        mx = max(lv.segments) if lv.segments else -1
                        segs_ = [lv.segments.get(i_) or {"index": i_, "ok": False, "data": bytes(tng5.SEG_BYTES)}
                                 for i_ in range(mx + 1)]
                    r = self.eng.result(x, fs, a_bb, a["df"], G[:nF_], segs_, lv.nchars, label + " · 중간 동기 합류",
                                        b_bb=b_bb, rho=a.get("rho"))
                elif (self.REUSE_LIVE and lv is not None and why == "length" and lv.end_reason == "length" and lv.nseg is not None
                      and len(lv.g) >= lv.n_true and len(lv.segments) == lv.nseg):
                    # 27부: 길이 필드로 끝 = 실시간 경로가 이미 같은 정렬 (A 위치 · df) 로 전 프레임 LLR 과 전 구간 복호를 끝냄.
                    # 전 구간 CRC 통과 → NN 을 처음부터 다시 돌리지 않고 그 값으로 결과 (메시지 끝 한 번 0.7~2.3 s 걸리던 것).
                    # 실패 구간이 있으면 예전처럼 오디오에서 다시 복조하고, 구간마다 CRC 통과한 쪽을 씀 (두 LLR 이 조금 달라 어느 한쪽만 풀리는 경우 → 어느 경우도 예전보다 나쁘지 않음)
                    G = np.stack(lv.g[:lv.n_true], axis=1)
                    h_ = int(np.argmax(np.abs(G).mean(axis=(1, 2))))
                    live_segs = [lv.segments[i_] for i_ in range(lv.nseg)]
                    if all(q["ok"] for q in live_segs):
                        r = self.eng.result(x, fs, a_bb, a["df"], G[h_], live_segs, lv.nchars, label, b_bb=b_bb, rho=a.get("rho"))
                        r.info["h_off"] = OFFS[h_]
                        r.info["reuse_live"] = True
                    else:
                        r = self.eng.decode_at(x, fs, a_bb, a["df"], nF=nF, b_bb=b_bb, why=label, rho=a.get("rho"))
                        at = {q["index"]: q for q in (r.info or {}).get("segments") or []}
                        if len(at) == lv.nseg and any(live_segs[i_]["ok"] and not at[i_]["ok"] for i_ in range(lv.nseg)):
                            merged = [live_segs[i_] if live_segs[i_]["ok"] else at[i_] for i_ in range(lv.nseg)]
                            h_off = r.info.get("h_off")
                            r = self.eng.result(x, fs, a_bb, a["df"], r.info.get("llr"), merged, lv.nchars, label, b_bb=b_bb,
                                                rho=a.get("rho"))
                            r.info["h_off"] = h_off
                            r.info["merged_live"] = True
                else:
                    r = self.eng.decode_at(x, fs, a_bb, a["df"], nF=nF, b_bb=b_bb, why=label, rho=a.get("rho"))
                if lv is not None and r.info is not None:
                    r.info["confirm"] = {"m_state": lv.m_state, "crc_ok": bool(lv.crc_ok), "m_info": lv.m_info,
                                         "by": "mid" if lv.m_state == "ok" else ("crc" if lv.crc_ok else None)}
            except Exception as ex:
                r = DecodeResult(False, reason="{}: {}".format(type(ex).__name__, ex), mode=NAME,
                                 info={"format": NAME})
            if why == "length" and r.info is not None:
                r.info["b_wait"] = True                           # 가드 · B 는 결과 뒤에 도착 (b_confirm / b_timeout 로 갱신)
            if not shown and why != "manual" and not r.ok and not (r.info or {}).get("segments_ok"):
                # 미확정 (중간 동기 확인 실패 · 미확인) + CRC 통과 구간 없음 → 조용히 폐기 (로그만)
                self.stats["dropped"] = self.stats.get("dropped", 0) + 1
                self._ev({"type": "cancel", "reason": "TNG5 미확정 폐기 ({}, 중간 동기 {}, CRC 통과 없음)".format(
                    label, "실패" if lv is not None and lv.m_state == "fail" else "미확인")})
                if a.get("mk_id"):
                    self.on_marker({"kind": "A", "abs": a["abs"] + int(N_TONE * k), "df": a["df"], "mode": NAME,
                                    "id": a["mk_id"], "remove": True, "dur": N_SYNC_A / self.eng.cfg.fs_base})
                return
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
        end = pos + int(N_SYNC_B * self.k + self.POST_S * fs)

        def work():
            tr = self._data_trace(pos, d["df"])
            if tr is not None and tr < self.TRACE_MIN:                # B 앞에 TNG5 데이터 흔적 없음 → 되짚지 않음 (휘파람 등)
                self._ev({"type": "b_only", "abs": pos, "df": d["df"], "reason": "trace", "trace": tr})
                return
            s0, x = self.ring.read(pos - int(self.MAX_S * fs), pos + int(0.2 * fs))
            if len(x) < int(3 * fs):
                return
            bb, kk = self.eng.to_bb(x, fs)
            # A: B 앞 2 s 이상 · 주파수 일치 · 실시간으로 이미 받은 A 가 아닌 것 (옛 메시지를 다시 복조하지 않게)
            dA = [a for a in self.eng.find(bb, "A") if a["start"] * kk + s0 < pos - int(2 * fs)
                  and abs(a["df"] - d["df"]) <= B_DF_HZ
                  and not any(abs(a["start"] * kk + s0 - p) < int(0.5 * fs) for p in self._a_hist)]
            dA = [a for a in dA if b_quantized((pos - (s0 + (a["start"] + self.eng.data0) * kk)) / kk)]
            if not dA:
                self._ev({"type": "b_only", "abs": pos, "df": d["df"]})
                return
            a = dA[-1]
            if d.get("_mk") is not None:
                self.on_marker(d["_mk"])                          # 맞는 A 를 찾았을 때만 B 표시
            s1, x2 = self.ring.read(int(s0 + a["start"] * kk) - int(self.PRE_S * fs), end)
            a_bb = (s0 + a["start"] * kk - s1) / kk
            r = self.eng.decode_at(x2, fs, a_bb, a["df"], b_bb=(pos - s1) / kk, why="코스타스 B 역방향")
            if not (r.info or {}).get("segments_ok"):                 # CRC 통과 구간 없음 → 표시 없이 로그만
                self._ev({"type": "b_only", "abs": pos, "df": d["df"], "reason": "no_crc"})
                return
            self.on_decoded(r, "실시간 · {} · 코스타스 B 역방향".format(NAME))

        if self.sync:
            work()
        else:
            threading.Thread(target=work, daemon=True, name="rt-back5").start()


# ====================================================================== 두 모드 묶음
class MultiRx:
    """
    TNG44 수신기(RealtimeReceiver) · TNG5 수신기 묶음. 앱은 이전처럼 self._rx 하나로 부른다.
    속성은 TNG44 수신기로 넘긴다 (기존 시험 스크립트 호환). 단일 모드 운용 (09-27): 선택한 모드 하나만 active,
    다른 모드 검출은 버린다 (모드 간 공동 판정 · 보호 구간은 없앰)
    """

    def __init__(self, rx44, rx5, active):
        self.__dict__["rx44"] = rx44
        self.__dict__["rx5"] = rx5
        self.__dict__["active"] = set(active)
        rx44_detect = rx44._on_detect

        def gate(kind, pos, d, now):
            if "TNG44" not in self.active:
                return
            return rx44_detect(kind, pos, d, now)
        rx44._on_detect = gate

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
        self.rx44.enabled = "TNG44" in self.active
        if not self.rx44.enabled:
            self.rx44._reset_state()
        if self.rx5 is not None:
            self.rx5.enabled = "TNG5" in self.active
            if not self.rx5.enabled:
                self.rx5._reset_state()

    def start(self):
        """켠 모드 (set_active) 의 수신 스레드만 띄운다. 모드를 바꾸면 앱이 묶음을 새로 만든다 (_reset_rx_mode → _ensure_rx)"""
        if self.rx44.enabled:
            self.rx44.start()
        if self.rx5 is not None and self.rx5.enabled:
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
    if info.get("format_version") == "1":                          # v1 WAV: v1 구조 (A×4 · B×4, 중간 동기 없음) 로 분석
        from tngpkt import tng5_v1_app
        if tng5_v1_app._ENG5 is None:
            tng5_v1_app._ENG5 = _eng_v1()
        out = tng5_v1_app.analyze(res, job, last_tx_text)
        out["format_version"] = "1"
        return out
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
    if not mask.all() and segs and all(r["ok"] for r in segs):
        n_c = nseg * SC
        if nF * 96 > n_c:
            truth[n_c:] = np.random.default_rng(n_c).integers(0, 2, nF * 96 - n_c)     # tng5.encode_bits 채움과 같음
            mask[n_c:] = True
    if not mask[:nseg * SC].all():
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
    out["timeline"] = timeline_v2(packet_timeline(nF, True, True, m.get("b") is not None, pc, cfg, statuses, plan, rows))
    if m.get("b") is None and info.get("b_wait"):
        out["timeline"] = timeline_b_state(out["timeline"], "wait")
    # 해부 자료: 구간별 바이트 · CRC · 글자, base-40 묶음, 부호 비트 위치
    rows5 = []
    for r in segs:
        k, data = r["index"], r["data"]
        idx = q[k * SC:(k + 1) * SC]
        comb = flat[idx].reshape(tng5.REP, -1)
        body = data[tng5.LEN_BYTES:] if k == 0 else data
        groups = []                                               # rev 4: 글자 단위 부호 (charset1) — 해부 창 ② 층
        if r["ok"]:
            from tngpkt import charset1 as CS
            for ch in tng5.unpack(body):
                groups.append(("".join(map(str, CS.CODE[ch])), len(CS.CODE[ch]), ch))
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
    try:                                                          # 8층 해부 (TNG44 해부 창 형식)
        from tngpkt.dissect5 import dissect5
        info["mid_checks"] = (out.get("sync5") or {}).get("mids")
        out["dissect"] = dissect5(res, _eng5().cfg, _eng5(), llr, segs)
    except Exception as ex:
        out["dissect"] = None
        out["dissect_err"] = "{}: {}".format(type(ex).__name__, ex)
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
    y_full = y
    # v2: 20프레임마다 중간 동기 M → 성상도 · 아이 · NN 특징은 M 을 뺀 연속 데이터 스트림으로 (프레임 위치가 v1 과 같아진다)
    y, bb = data_stream(y, frame0, nF), data_stream(bb, frame0, nF)
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
    out["sync5"] = sync_map(e, y_full, a_bb, df, res.snr_db, info=info, nF=nF)
    return out


def data_stream(y, frame0, nF, tail=3):
    """송신 순서 기저대역 → 중간 동기 구간을 뺀 연속 스트림 (첫 데이터 프레임 앞은 그대로)"""
    j = np.arange((nF + tail) * L)
    idx = frame0 + tng5.stream_to_tx(j)
    return np.concatenate([y[:max(frame0, 0)], y[np.clip(idx, 0, len(y) - 1)]])


def mid_check(e, y, a_bb, df, k=1, floor=None):
    """중간 동기 M_k 재측정 (수신기 확인과 같은 규칙): {'k','v','rho','need','ok','t_s'}"""
    tm = tng5.mid_templates()[(k - 1) % 4]
    p = int(round(a_bb + e.data0 + tng5.mid_off(k)))
    st = np.arange(p - M_POS, p + M_POS + 1, 2)
    st = st[(st >= len(tm)) & (st < len(y) - 2 * len(tm) - 1)]
    if len(st) == 0:
        return None
    yt = torch.as_tensor(y.astype(np.complex64), device=e.device)[None]
    with e._lock:
        _, r = tng5.metric(yt, [(0, tm, "costas")], st, e.device)
    sel = np.abs(tng5.FREQ[tng5.BAND]) <= M_DF_HZ                   # y 는 주파수 보정 뒤
    rr = r[0].cpu().numpy()[:, sel]
    v = float(rr.max())
    a = int(st[int(rr.max(1).argmax())])
    pw = float(np.mean(np.abs(y[a:a + len(tm)]) ** 2))
    if floor is None:
        floor = _floor(y[:max(int(a_bb) - 100, 0)]) or _floor(y) or pw
    sh = max(pw / max(floor, 1e-30) - 1.0, 0.0)
    rho, need = v / len(tm), M_FRAC * sh / (1.0 + sh)
    return {"k": k, "v": v, "rho": rho, "need": need, "ok": bool(v >= M_T and rho >= need), "t_s": (a - p) / BB_FS}


def sync_map(e, y, a_bb, df, snr=None, span_s=0.1, info=None, nF=None):
    """
    TNG5 시작 동기 맵 (v2: Welch12 64 ms · 31.25 Hz, 4심볼 × 3조각): 시간 × 주파수 오차 평균 ρ² (조각별 정규화 상관의 평균).
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
    out = {"rho": rho, "f": f, "t": t, "utc": time.time(), "mode": NAME,
           "det": {"df": 0.0, "rho": float(rho[:, j0].max())},
           "check": {"refine_rho": float(rho[i0, j0]), "map_at_det": float(rho[i0, j0])}}
    # v2: 중간 동기 확인 (재측정 M1 · 실시간 확인 경로) — 제목 줄 · 도움말
    try:
        yc = y                                                     # expert 가 넘기는 y 는 이미 주파수 보정됨
        mids = []
        if nF is not None:
            for kq in range(1, tng5.n_mids(nF) + 1):
                mc = mid_check(e, yc, a_bb, 0.0, kq)
                if mc is not None:
                    mids.append(mc)
        cf = (info or {}).get("confirm") or {}
        by = {"mid": "M1 확인", "crc": "첫 CRC 구제"}.get(cf.get("by"), "파일 복조" if not cf else "미확정")
        note = " · 확정: {}".format(by)
        if mids:
            note += " · M1 ρ² {:.2f} {}".format(mids[0]["rho"], "✓" if mids[0]["ok"] else "✗")
        out["note"] = note
        out["note_tip"] = "\n".join(["중간 동기 M{k}: ρ² {rho:.3f} (기준 {need:.3f}) · 값 {v:.1f} (문턱 {t:.2f}) · 위치 {ts:+.4f} s · {ok}".format(
            k=m_["k"], rho=m_["rho"], need=m_["need"], v=m_["v"], t=M_T, ts=m_["t_s"], ok="통과" if m_["ok"] else "실패")
            for m_ in mids] + ["수신 확정 경로: {}".format(by)])
        out["mids"] = mids
    except Exception as ex:                                        # 동기 맵 본체는 그대로
        out["note"] = " · 중간 동기 재측정 실패 ({})".format(type(ex).__name__)
    return out
