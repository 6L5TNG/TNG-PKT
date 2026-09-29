"""
신호 시각화 계산 — 수신/복조 스레드와 **분리된** 별도 스레드에서 돈다.

복조가 끝난 결과(DecodeResult.viz)와 송신 오디오를 받아, 그림에 쓸 숫자만 만들어 돌려준다.
복조 로직은 다시 돌리지 않는다. 복조가 쓴 것과 같은 구간을 같은 방법으로 다시 만들어
(주파수 보정 → 데이터 구간) 그 위에서 성상도·아이 다이어그램·동기 맵·LLR 분포를 계산한다.

  · 성상도       : 주파수 보정 후 심볼 중앙 샘플. 위상 기준은 코스타스 A 의 상관 위상
  · 아이 다이어그램: 심볼 2개 길이로 겹쳐 그린 I, Q (8배 보간)
  · 코스타스 동기 맵: 시간 × 주파수 오차 의 ρ² 히트맵 (검출 지점 표시)
  · LLR 히스토그램 : 수신 NN 소프트 출력. 정답을 알면 0/1 로 나눈다
  · 패킷 타임라인  : 톤→A→프레임들→B, 프레임별 FEC 가 고친 비트 수
"""

import queue
import threading

import numpy as np
from scipy import signal as sg
from PySide6.QtCore import QObject, Signal

from modem import audio_to_baseband
from link4 import encode_text, GUARD_BITS
from preamble import CostasDetector

N_EYE = 160            # 아이 다이어그램에 겹칠 궤적 수
EYE_UP = 8             # 아이 다이어그램 보간 배율


# ------------------------------------------------------------------ 구간 재구성
def _phase_from_A(y, a_start, det):
    """주파수 보정된 신호에서 코스타스 A 와의 상관 위상 → 성상도 위상 기준"""
    n = det.n
    if a_start is None or a_start < 0 or a_start + n > len(y):
        return None
    z = np.sum(y[a_start:a_start + n] * np.conj(det.tA))
    return float(np.angle(z))


def rebuild_rx(viz, cfg, det):
    """
    복조가 쓴 데이터 구간을 똑같이 다시 만든다.
    반환: dict(bbd, frame0, n_data, phase, bb, a_start, df) 또는 None
    """
    m = viz.get("msg") or {}
    info = m.get("info") or {}
    if "frame0" not in info or info.get("llr") is None:
        return None
    bb = audio_to_baseband(np.asarray(viz["audio"], dtype=np.float64), viz["fs"], cfg)
    a = m.get("a")
    if a is None:                                     # 옛 형식: 버퍼 전체를 그대로 복조했다
        return {"bbd": bb, "frame0": int(info["frame0"]), "n_data": int(info["llr"].shape[0]),
                "phase": None, "bb": bb, "a_start": None, "df": 0.0}
    df = float(m["df"])
    t = np.arange(len(bb)) / cfg.fs_base
    y = bb * np.exp(-2j * np.pi * df * t)
    a_end, b_start = m["seg"]
    seg = y[a_end:b_start] if b_start is not None else y[a_end:]
    z = np.zeros(int(info.get("pad", 0)), dtype=complex)
    return {"bbd": np.concatenate([z, seg, z]), "frame0": int(info["frame0"]),
            "n_data": int(info["llr"].shape[0]), "phase": _phase_from_A(y, a["start"], det),
            "bb": bb, "a_start": int(a["start"]), "df": df}


def rebuild_tx(bb, info, cfg, det):
    """송신 기저대역 (잡음·주파수오차 없음). 첫 데이터 프레임 = 데이터 시작 + 가드 1프레임"""
    return {"bbd": bb, "frame0": int(info["data_start"] + cfg.frame_len),
            "n_data": int(info["n_data"]), "phase": _phase_from_A(bb, info["a_start"], det),
            "bb": bb, "a_start": int(info["a_start"]), "df": 0.0}


# ------------------------------------------------------------------ 그림용 숫자
def constellation(r, cfg):
    """심볼 중앙 샘플 (복소), RMS 1 로 정규화, A 위상으로 되돌림"""
    sps, S = cfg.sps, cfg.symbols_per_frame
    k = np.arange(r["n_data"] * S)
    idx = r["frame0"] + (k // S) * cfg.frame_len + (k % S) * sps + sps // 2
    idx = idx[(idx >= 0) & (idx < len(r["bbd"]))]
    p = r["bbd"][idx]
    if r["phase"] is not None:
        p = p * np.exp(-1j * r["phase"])
    return p / max(np.sqrt(np.mean(np.abs(p) ** 2)), 1e-12)


def eye(r, cfg, n_traces=N_EYE, rng=None):
    """심볼 2개 길이의 I, Q 궤적. 반환: (x축[심볼], I 궤적들 (n, m), Q 궤적들 (n, m))"""
    sps, S = cfg.sps, cfg.symbols_per_frame
    x = r["bbd"]
    if r["phase"] is not None:
        x = x * np.exp(-1j * r["phase"])
    up = sg.resample_poly(x, EYE_UP, 1)
    L = 2 * sps * EYE_UP
    k = np.arange(r["n_data"] * S - 1)
    st = (r["frame0"] + (k // S) * cfg.frame_len + (k % S) * sps) * EYE_UP
    st = st[(st >= 0) & (st + L <= len(up))]
    if len(st) == 0:
        return None
    rng = rng or np.random.default_rng(0)
    if len(st) > n_traces:
        st = rng.choice(st, n_traces, replace=False)
    seg = up[st[:, None] + np.arange(L)[None, :]]
    seg = seg / max(np.sqrt(np.mean(np.abs(seg) ** 2)), 1e-12)
    return np.arange(L) / (sps * EYE_UP), seg.real, seg.imag


def costas_map(r, det, span_s=0.3):
    """코스타스 A 둘레의 시간 × 주파수 오차 ρ² (행 = 주파수, 열 = 시간)"""
    if r.get("a_start") is None:
        return None
    pc, fs = det.pc, det.fs
    x = det._prefilter(r["bb"])
    a = r["a_start"]
    lo, hi = max(0, a - int(span_s * fs)), min(len(x) - det.n - 1, a + int(span_s * fs))
    if hi <= lo:
        return None
    starts = np.arange(lo, hi, 2)
    g = det._corr_grid(x, starts, det.tA, pc.coarse_nfft)
    f = np.fft.fftfreq(pc.coarse_nfft, 1.0 / fs)
    band = np.abs(f) <= pc.max_freq_off_hz
    order = np.argsort(f[band])
    rho = g[:, band][:, order].T                              # (주파수, 시간)
    return {"rho": rho, "t_ms": (starts - a) * 1000.0 / fs, "f": f[band][order],
            "peak": (0.0, r["df"])}


def llr_hist(llr, truth=None, lim=30.0, bins=60):
    """수신 NN 소프트 출력 분포. truth(0/1 배열)가 있으면 나눠서 센다."""
    edges = np.linspace(-lim, lim, bins + 1)
    v = np.clip(np.asarray(llr).ravel(), -lim, lim)
    if truth is None:
        return {"edges": edges, "all": np.histogram(v, edges)[0]}
    t = np.asarray(truth).ravel().astype(bool)
    return {"edges": edges, "zero": np.histogram(v[~t], edges)[0],
            "one": np.histogram(v[t], edges)[0],
            "wrong": int(np.sum((v > 0) != t)), "n": int(len(v))}


def frame_status(llr, text):
    """
    프레임별 상태. 복조 성공이면 정답 부호어를 다시 만들어 hard 판정과 비교한다.
    반환: 프레임마다 (상태, 고친 비트 수) — 상태: 'ok' / 'fixed' / 'fail'
    """
    n = llr.shape[0]
    if text is None:
        return [("fail", None)] * n, None
    exp, n_exp = encode_text(text)
    if n_exp != n:
        return [("fail", None)] * n, None
    err = ((llr > 0) != (exp > 0.5)).sum(axis=1)
    return [("ok" if e == 0 else "fixed", int(e)) for e in err], exp


def timeline(r_ok, text, n_data, has_tone, has_a, has_b, cfg, pc, statuses):
    """패킷 막대: [(이름, 시작 ms, 길이 ms, 종류, 라벨)]"""
    fr = 1000.0 * cfg.frame_len / cfg.fs_base
    cost = pc.symbol_ms * pc.n_tones
    out, t = [], 0.0
    if has_tone:
        out.append(("톤", t, pc.tone_ms, "tone", "톤"))
        t += pc.tone_ms
    if has_a:
        out.append(("A", t, cost, "costas", "A"))
        t += cost
    out.append(("G", t, fr, "guard", "G"))
    t += fr
    for i in range(n_data):
        st, fixed = statuses[i] if i < len(statuses) else ("fail", None)
        lab = "" if st == "ok" else ("{}".format(fixed) if st == "fixed" else "✕")
        out.append(("F{}".format(i + 1), t, fr, st, lab))
        t += fr
    out.append(("G", t, fr, "guard", "G"))
    t += fr
    if has_b:
        out.append(("B", t, cost, "costas", "B"))
    else:
        out.append(("B?", t, cost, "missing", "B 없음"))
    return out


# ------------------------------------------------------------------ 작업 스레드
class VizWorker(QObject):
    """큐에 쌓인 작업을 별도 스레드에서 계산하고 ready 시그널로 결과를 넘긴다."""
    ready = Signal(object)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.cfg = engine.cfg
        self.pc = engine.pc
        self.det = CostasDetector(self.pc, self.cfg.fs_base)     # 수신용과 따로 (스레드 안전)
        self.q = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self.last_tx_text = None
        self.t = threading.Thread(target=self._run, daemon=True, name="viz")
        self.t.start()

    def stop(self):
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass

    def submit(self, job):
        try:
            self.q.put_nowait(job)
        except queue.Full:                     # 밀리면 버린다 — 시각화 때문에 아무것도 막지 않는다
            pass

    def _run(self):
        while not self._stop.is_set():
            job = self.q.get()
            if job is None:
                break
            try:
                out = self._rx(job) if job["kind"] == "rx" else self._tx(job)
                if out is not None:
                    self.ready.emit(out)
            except Exception as ex:
                self.ready.emit({"kind": "error", "msg": "시각화 계산 오류 — {}: {}".format(
                    type(ex).__name__, ex)})

    def _rx(self, job):
        res = job["result"]
        viz = getattr(res, "viz", None) or {}
        r = rebuild_rx(viz, self.cfg, self.det) if viz else None
        m = viz.get("msg") or {}
        info = m.get("info") or {}
        out = {"kind": "rx", "ok": res.ok, "text": res.text, "label": job.get("label", ""),
               "df": res.df_hz, "snr": res.snr_db}
        llr = info.get("llr")
        if llr is not None:
            statuses, exp = frame_status(llr, res.text if res.ok else None)
            truth = exp
            # 복조 실패여도 루프백(직접 보낸 문장)이면 정답을 안다
            if truth is None and self.last_tx_text and job.get("loopback"):
                e2, n2 = encode_text(self.last_tx_text)
                truth = e2 if n2 == llr.shape[0] else None
            out["llr"] = llr_hist(llr, truth)
            out["statuses"] = statuses
            out["fixed_total"] = int(sum(f or 0 for s, f in statuses if s == "fixed"))
            out["timeline"] = timeline(res.ok, res.text, llr.shape[0], m.get("a") is not None,
                                       m.get("a") is not None, m.get("b") is not None,
                                       self.cfg, self.pc, statuses)
        if r is not None:
            out["const"] = constellation(r, self.cfg)
            out["eye"] = eye(r, self.cfg)
            out["cmap"] = costas_map(r, self.det)
            out["phase_ref"] = r["phase"] is not None
        return out

    def _tx(self, job):
        from modem5 import text_to_baseband5
        eng = self.engine
        with eng._lock:
            bb, info = text_to_baseband5(job["text"], eng.model, eng.cfg, eng.device, eng.pc)
        self.last_tx_text = job["text"]
        r = rebuild_tx(bb, info, self.cfg, self.det)
        return {"kind": "tx", "const": constellation(r, self.cfg), "eye": eye(r, self.cfg)}


# ====================================================================== 데이터 보조 분석
# 복조에 성공하면 보낸 비트를 안다. 그 비트로 이상적인 송신 파형을 다시 만들어 수신 파형과 맞대면
#   · 잔여 위상 드리프트 (프레임별 상관 위상의 기울기)
#   · 채널(스피커·방 반향·다중경로 등)의 선형 왜곡
# 을 추정해 되돌릴 수 있다. 계측기의 EVM 측정과 같은 방식이며, 복조 결과에는 영향이 없다.

def ideal_tx(text, model, cfg, dev, pc, fmt="stream"):
    """복조된 문장으로 이상적인 송신 기저대역을 다시 만든다 (시각화 전용 모델 복사본 사용). fmt: stream / block"""
    from modem5 import text_to_baseband5
    return text_to_baseband5(text, model, cfg, dev, pc, fmt=fmt)


def _sym_centers(n_frames, cfg, first_frame=1):
    """데이터 구간(가드 포함) 안에서 데이터 프레임 심볼 중앙 위치"""
    S, sps, L = cfg.symbols_per_frame, cfg.sps, cfg.frame_len
    k = np.arange(n_frames * S)
    return (first_frame + k // S) * L + (k % S) * sps + sps // 2


def data_aided_eq(r, tx_bb, tx_info, cfg, n_taps=65):
    """
    수신 데이터 구간을 이상적 송신과 정렬해 위상 드리프트 → 선형 등화(LS FIR) 를 추정한다.
    반환: dict(rx_seg, eq_seg, tx_seg, centers, evm_raw_db, evm_eq_db, drift_hz, n_data)
    """
    L, n = cfg.frame_len, r["n_data"]
    n_seg = (n + 2) * L                                        # 가드 + 데이터 + 가드
    a = r["frame0"] - L
    rx = r["bbd"][a:a + n_seg]
    tx = tx_bb[tx_info["data_start"]:tx_info["data_start"] + n_seg]
    if len(rx) < n_seg or len(tx) < n_seg:
        return None
    # 1) 프레임별 상관 위상 → 선형 맞춤 → 잔여 주파수·위상 드리프트 제거
    g = np.array([np.sum(rx[i * L:(i + 1) * L] * np.conj(tx[i * L:(i + 1) * L])) for i in range(n + 2)])
    ph = np.unwrap(np.angle(g))
    tf = (np.arange(n + 2) + 0.5) * L
    p = np.polyfit(tf, ph, 1)
    t = np.arange(n_seg)
    rx_d = rx * np.exp(-1j * (p[0] * t + p[1]))
    drift_hz = p[0] * cfg.fs_base / (2 * np.pi)
    # 2) LS 선형 등화: rx 를 n_taps 개 지연 합으로 tx 에 맞춘다
    h = n_taps // 2
    xp = np.concatenate([np.zeros(h, complex), rx_d, np.zeros(h, complex)])
    X = np.lib.stride_tricks.sliding_window_view(xp, n_taps)[:n_seg][:, ::-1]
    w, *_ = np.linalg.lstsq(X, tx, rcond=None)
    eq = X @ w
    c = _sym_centers(n, cfg)

    def evm(y):
        s = tx[c]
        gain = np.vdot(y[c], s) / max(np.vdot(y[c], y[c]).real, 1e-20)   # 복소 이득 1개만 맞춘다
        e = np.mean(np.abs(gain * y[c] - s) ** 2) / np.mean(np.abs(s) ** 2)
        return 10 * np.log10(max(e, 1e-12)), gain

    e_raw, g_raw = evm(rx_d)
    e_eq, g_eq = evm(eq)
    return {"rx_seg": rx_d * g_raw, "eq_seg": eq * g_eq, "tx_seg": tx, "centers": c,
            "evm_raw_db": e_raw, "evm_eq_db": e_eq, "drift_hz": drift_hz, "n_data": n,
            "eq_taps": w}


def nn_features(r, rx_model, cfg, dev, truth_bits=None):
    """
    수신 NN 의 마지막 층(비트 판정 직전) 특징을 심볼마다 꺼내 PCA 2차원으로 줄인다.
    AI 파형은 심볼 중앙 샘플 하나로 뜻이 정해지지 않을 수 있지만, 수신 NN 은 프레임 전체를
    보고 판단하므로 이 특징 공간에서는 비트쌍별로 무리를 이룬다.
    반환: dict(xy (N,2), cls (N,) 또는 None, sep 분리도)
    """
    import torch
    L, W, C, S = cfg.frame_len, cfg.win_len, cfg.ctx_len, cfg.symbols_per_frame
    n = r["n_data"]
    bb = r["bbd"]
    st = r["frame0"] + np.arange(n) * L - C
    if st[0] < 0 or st[-1] + W > len(bb):
        return None
    win = bb[st[:, None] + np.arange(W)[None, :]]
    feats = {}
    hook = rx_model.head.register_forward_hook(lambda m, i, o: feats.__setitem__("h", i[0].detach()))
    try:
        with torch.no_grad():
            rx_model(torch.tensor(win, dtype=torch.complex64, device=dev))
    finally:
        hook.remove()
    h = feats["h"].permute(0, 2, 1).reshape(n * S, -1).cpu().numpy()     # (심볼, 채널)
    h = h - h.mean(axis=0)
    u, s, vt = np.linalg.svd(h, full_matrices=False)
    xy = h @ vt[:2].T
    xy /= max(np.std(xy), 1e-12)
    out = {"xy": xy, "cls": None, "sep": None, "var2": float((s[:2] ** 2).sum() / (s ** 2).sum())}
    if truth_bits is not None:
        b = np.asarray(truth_bits).reshape(n, S, 2)
        cls = (2 * b[..., 0] + b[..., 1]).reshape(-1).astype(int)
        out["cls"] = cls
        # 분리도: 가장 가까운 무리 중심 분류 정확도 (2차원 PCA 공간)
        cen = np.stack([xy[cls == k].mean(axis=0) if np.any(cls == k) else np.full(2, 1e9) for k in range(4)])
        pred = np.argmin(((xy[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        out["sep"] = float(np.mean(pred == cls))
    return out
