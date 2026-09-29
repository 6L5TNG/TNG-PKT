"""
분석 계산 스레드 — 복조 스레드와 분리. 화면은 결과만 받아 그린다.

  PacketAnalyzer  패킷 단위 (복조될 때마다 1회)
      등화 성상도 · 아이 · 위상 궤적 · 수신 NN 특징 · LLR · 프레임별 오류/|LLR| · 동기 맵 · 채널 품질
      복조에 성공하면 보낸 비트를 알게 되므로 '데이터 보조 등화'로 채널 왜곡을 되돌려 보여준다
      (원인 조사: results/constellation_investigation.log). 모델은 시각화 전용 복사본을 쓴다.
  LiveAnalyzer    연속 (초당 10회)
      슬라이딩 코스타스 동기 맵 · 코스타스 A 검출 뒤 주파수 보정한 실시간 성상도 흐름
      · 검출 스냅샷: A 가 검출되면 검출기와 같은 ρ² 계산(같은 전처리 · 같은 정규화 · 같은 2ms 격자)으로
        검출 위치 둘레 맵을 한 번 만들어 고정 표시용으로 보낸다
  ZoomAnalyzer    연속 (초당 약 20회) — 프로토콜 대역 확대 워터폴
      1500Hz 로 복소 믹싱 → 저역 필터 → 약 2000Hz 로 데시메이션 → 겹침 큰 FFT (분해능 1~4Hz 선택)
"""

import copy
import queue
import threading
import time

import numpy as np
from scipy import signal as sg
from PySide6.QtCore import QObject, Signal

from tngpkt.modem import audio_to_baseband
from tngpkt.link4 import encode_text
from tngpkt.preamble import CostasDetector
from tngpkt.dissect import dissect
from tngpkt.viz import (rebuild_rx, constellation, eye, costas_map, llr_hist, frame_status, timeline,
                 ideal_tx, data_aided_eq, nn_features, EYE_UP)

FAIL_SNR_DB = -6.5            # 4단계 측정: 텍스트 완전 복원율 50% 지점 (AWGN) — 복원 여유의 기준


def _eye_from_seg(seg, cfg, n_data, max_traces=160, rng=None):
    """가드부터 시작하는 데이터 구간 seg 에서 심볼 2개 길이 궤적 (프레임 번호 포함)"""
    S, sps, L = cfg.symbols_per_frame, cfg.sps, cfg.frame_len
    up = sg.resample_poly(seg, EYE_UP, 1)
    W = 2 * sps * EYE_UP
    k = np.arange(n_data * S - 1)
    fr = k // S
    st = ((1 + fr) * L + (k % S) * sps) * EYE_UP
    ok = st + W <= len(up)
    st, fr = st[ok], fr[ok]
    rng = rng or np.random.default_rng(0)
    if len(st) > max_traces:
        sel = np.sort(rng.choice(len(st), max_traces, replace=False))
        st, fr = st[sel], fr[sel]
    x = up[st[:, None] + np.arange(W)[None, :]]
    x = x / max(np.sqrt(np.mean(np.abs(x) ** 2)), 1e-12)
    return np.arange(W) / (sps * EYE_UP), x.real, x.imag, fr


def _trajectory(seg, cfg, frame=1, up=4):
    """한 프레임의 I/Q 궤적 (위상 궤적) 과 심볼 중앙점"""
    L, sps = cfg.frame_len, cfg.sps
    a = frame * L
    x = seg[a:a + L]
    y = sg.resample_poly(x, up, 1)
    c = x[np.arange(cfg.symbols_per_frame) * sps + sps // 2]
    n = max(np.sqrt(np.mean(np.abs(y) ** 2)), 1e-12)
    return y / n, c / n


class PacketAnalyzer(QObject):
    ready = Signal(object)

    def __init__(self, engine):
        super().__init__()
        self.cfg, self.pc = engine.cfg, engine.pc
        self.dev = engine.device
        self.model = copy.deepcopy(engine.model).eval()        # 복조와 잠금을 다투지 않도록 복사본
        self.det = CostasDetector(self.pc, self.cfg.fs_base)
        self.q = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self.last_tx_text = None
        self.t = threading.Thread(target=self._run, daemon=True, name="packet-analyzer")
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
        except queue.Full:
            pass                                               # 밀리면 버린다 — 수신을 막지 않는다

    def _run(self):
        while not self._stop.is_set():
            job = self.q.get()
            if job is None:
                break
            t0 = time.time()
            try:
                out = self._rx(job) if job["kind"] == "rx" else self._tx(job)
                out["calc_s"] = time.time() - t0
                self.ready.emit(out)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                self.ready.emit({"kind": "error", "msg": "분석 계산 오류: {} {}".format(type(ex).__name__, ex)})

    # ---------------------------------------------------------------- 수신 패킷
    def _rx(self, job):
        res = job["result"]
        if (getattr(res, "info", None) or {}).get("format") == "TNG5":
            from tngpkt.tng5_app import analyze
            return analyze(res, job, self.last_tx_text)
        if (getattr(res, "info", None) or {}).get("format") == "TNG1":
            from tngpkt.tng1_app import analyze as analyze1
            return analyze1(res, job, self.last_tx_text)
        cfg = self.cfg
        viz = getattr(res, "viz", None) or {}
        m = viz.get("msg") or {}
        info = m.get("info") or {}
        out = {"kind": "rx", "id": job.get("id"), "ok": res.ok, "text": res.text,
               "label": job.get("label", ""), "df": res.df_hz, "snr": res.snr_db,
               "utc": job.get("utc"), "mode": res.mode, "reason": res.reason,
               "margin": (res.snr_db - FAIL_SNR_DB) if np.isfinite(res.snr_db) else None,
               "rho_a": (m.get("a") or {}).get("rho") if isinstance(m.get("a"), dict) else None}
        llr = info.get("llr")
        truth = None
        stream = info.get("format") == "스트리밍"
        has_a, has_b = m.get("a") is not None, m.get("b") is not None
        out["format"] = info.get("format") or "단일 블록"
        if stream:
            out.update({"segments_ok": info.get("segments_ok"), "segments_used": info.get("segments_used"),
                        "drift_ppm": info.get("drift_ppm"), "relocks": info.get("relocks"),
                        "partial": info.get("partial")})
        if llr is not None and stream:
            from tngpkt.stream_ui import stream_truth, StreamPlan, packet_timeline, encode_bytes
            t_, mask = stream_truth(llr, info)
            if not mask.all() and job.get("loopback") and self.last_tx_text:
                from tngpkt.link7 import encode_raw, stream_bytes
                P_ = self.last_tx_text.encode("utf-8")
                fr_, _ = (encode_raw(stream_bytes(P_, 2)) if info.get("version") == 2
                          else encode_bytes(P_))                         # 루프백: 내가 보낸 문장이 정답
                if fr_.shape == llr.shape:
                    t_, mask = fr_.astype(np.int64), np.ones(llr.shape, dtype=bool)
            err = ((llr > 0) != (t_ > 0)) & mask
            ferr, fknown = err.sum(axis=1), mask.sum(axis=1)
            statuses = [("unk", "") if fknown[i] == 0 else
                        (("fixed", str(int(ferr[i]))) if ferr[i] else ("ok", "")) for i in range(len(ferr))]
            out["frame_err"] = [None if fknown[i] == 0 else int(ferr[i]) for i in range(len(ferr))]
            out["frame_llr"] = np.abs(llr).mean(axis=1)
            out["statuses"] = statuses
            out["llr"] = llr_hist(llr[mask], t_[mask]) if mask.any() else llr_hist(llr)
            out["fec_total"] = int(mask.sum())
            out["fec_fixed"] = int(err.sum()) if mask.any() else None
            out["ber"] = float(err.sum() / mask.sum()) if mask.any() else None
            truth = t_.astype(np.float32) if mask.all() else None
            plan = StreamPlan(llr.shape[0])
            segs = {r["index"]: r for r in info.get("segments") or []}
            rows = [{"ok": segs[k]["ok"], "decided": True} if k in segs else None for k in range(plan.n_seg)]
            out["timeline"] = packet_timeline(llr.shape[0], has_a, has_a, has_b, self.pc, cfg, statuses, plan, rows)
        elif llr is not None:
            statuses, exp = frame_status(llr, res.text if res.ok else None)
            truth = exp
            if truth is None and job.get("loopback") and self.last_tx_text:
                e2, n2 = encode_text(self.last_tx_text)       # 루프백: 내가 보낸 문장이 정답
                truth = e2 if n2 == llr.shape[0] else None
            err = ((llr > 0) != (truth > 0.5)).sum(axis=1) if truth is not None else None
            out["frame_err"] = err
            out["frame_llr"] = np.abs(llr).mean(axis=1)
            out["statuses"] = statuses
            out["llr"] = llr_hist(llr, truth)
            out["fec_total"] = int(llr.size)
            out["fec_fixed"] = int(err.sum()) if (err is not None and res.ok) else None
            out["ber"] = float(err.sum() / llr.size) if err is not None else None
            from tngpkt.stream_ui import packet_timeline
            fs_ = [(st, "" if st == "ok" else ("{}".format(fx) if st == "fixed" else "✕")) for st, fx in statuses]
            out["timeline"] = packet_timeline(llr.shape[0], has_a, has_a, has_b, self.pc, cfg, fs_, ok=res.ok)
        r = rebuild_rx(viz, cfg, self.det) if viz else None
        if r is None:
            return out
        S = cfg.symbols_per_frame
        p = constellation(r, cfg)
        out["const_raw"] = p
        out["const_frame"] = np.arange(len(p)) // S
        out["phase_ref"] = r["phase"] is not None
        out["cmap"] = costas_map(r, self.det)
        da = None
        if res.ok:
            tx_bb, tx_info = ideal_tx(res.text, self.model, cfg, self.dev, self.pc,
                                      fmt="stream" if stream else "block")
            da = data_aided_eq(r, tx_bb, tx_info, cfg)
        if da is not None:
            c = da["centers"]
            out["const_eq"] = da["eq_seg"][c]
            out["const_ideal"] = da["tx_seg"][c]
            out["evm_raw"], out["evm_eq"], out["drift_hz"] = da["evm_raw_db"], da["evm_eq_db"], da["drift_hz"]
            out["eye"] = _eye_from_seg(da["eq_seg"], cfg, r["n_data"])
            out["eye_raw"] = _eye_from_seg(da["rx_seg"], cfg, r["n_data"])
            out["traj"] = [_trajectory(da["eq_seg"], cfg, f + 1) for f in range(r["n_data"])]
        else:
            e = eye(r, cfg)
            if e is not None:
                out["eye_raw"] = (e[0], e[1], e[2], np.zeros(len(e[1]), int))
        try:
            out["dissect"] = dissect(res, cfg, self.pc, self.det)
        except Exception as ex:                                   # 해부가 실패해도 다른 그림은 그대로
            out["dissect"] = None
            out["dissect_err"] = "{}: {}".format(type(ex).__name__, ex)
        out["feat"] = nn_features(r, self.model.rx, cfg, self.dev, truth)
        if out.get("feat") is not None:
            out["feat_frame"] = np.arange(len(out["feat"]["xy"])) // S
        return out

    # ---------------------------------------------------------------- 송신
    def _tx(self, job):
        from tngpkt.viz import rebuild_tx
        if job.get("mode") in ("TNG5", "TNG1"):           # TNG5 · TNG1 송신 성상도 · 아이는 없음 (정답 텍스트만 기억)
            self.last_tx_text = job["text"]
            return {"kind": "tx", "const": None, "eye": None}
        tx_bb, info = ideal_tx(job["text"], self.model, self.cfg, self.dev, self.pc)
        self.last_tx_text = job["text"]
        r = rebuild_tx(tx_bb, info, self.cfg, self.det)
        seg = tx_bb[info["data_start"]:info["data_start"] + (info["n_data"] + 2) * self.cfg.frame_len]
        return {"kind": "tx", "const": constellation(r, self.cfg),
                "eye": _eye_from_seg(seg, self.cfg, info["n_data"])}


class LiveAnalyzer(QObject):
    """
    연속 분석 (초당 hz 회). source() 는 (링버퍼, 입력 샘플레이트) 또는 None 을 돌려준다.
    track(a_abs, df) 로 코스타스 A 검출을 알려 주면, 그 뒤로 들어오는 심볼 중앙 샘플을
    주파수 보정·위상 기준을 맞춰 실시간 성상도로 흘려 보낸다.
    """
    ready = Signal(object)
    SPAN_S = 1.5
    TRACK_S = 90.0

    def __init__(self, engine, source, hz=10.0):
        super().__init__()
        self.cfg, self.pc = engine.cfg, engine.pc
        self.det = CostasDetector(self.pc, self.cfg.fs_base)
        self.source = source
        self.period = 1.0 / hz
        self._trk = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True, name="live-analyzer")
        self.t.start()

    def stop(self):
        self._stop.set()

    def track(self, a_abs, df):
        with self._lock:
            self._trk = {"a": int(a_abs), "df": float(df), "phase": None, "next": None, "t0": time.time()}

    def untrack(self):
        with self._lock:
            self._trk = None

    def snapshot(self, marker):
        """코스타스 A 검출 알림 {'abs','df','rho'} → 다음 계산 때 검출 스냅샷을 만든다"""
        with self._lock:
            self._snap_req = dict(marker, t_req=time.time())

    SNAP_BEFORE_S, SNAP_AFTER_S = 0.9, 0.6

    def _snapshot(self, ring, fs, req):
        """
        검출 위치 둘레 맵. 검출기와 같은 계산: 같은 전처리(_prefilter) · 같은 정규화(_corr_grid) ·
        같은 시간 간격(pc.coarse_step = 2ms) · 같은 주파수 격자(pc.coarse_nfft). 시간 격자는
        검출 시작 샘플이 정확히 한 열이 되도록 맞춘다. 검출 위치의 정밀 ρ² 도 함께 다시 잰다.
        """
        cfg, pc, det = self.cfg, self.pc, self.det
        k = int(round(fs / cfg.fs_base))
        if abs(fs / cfg.fs_base - k) > 1e-9:
            return None                                      # 정수배가 아닌 입력은 스냅샷 생략
        a_abs = int(req["abs"])
        lo = a_abs - int(self.SNAP_BEFORE_S * fs) - int(0.2 * fs)
        lo -= (lo - a_abs) % k                               # 검출 시작이 기저대역 정수 번호에 오게
        hi = min(ring.count, a_abs + int(self.SNAP_AFTER_S * fs) + det.n * k + int(0.2 * fs))
        s0, raw = ring.read(lo, hi)
        if len(raw) < det.n * k * 3:
            return None
        cut = (a_abs - s0) % k                               # 링이 앞을 잘랐을 때도 정수 번호가 되게
        raw, s0 = raw[cut:], s0 + cut
        bb = audio_to_baseband(raw.astype(np.float64), fs, cfg)
        x = det._prefilter(bb)
        ia = (a_abs - s0) // k                               # 검출 시작 (기저대역 번호)
        step = pc.coarse_step
        lo_j = max(0, ia - int(self.SNAP_BEFORE_S * cfg.fs_base))
        lo_j += (ia - lo_j) % step                           # 검출 시작이 정확히 한 열이 되게
        hi_j = min(len(x) - det.n - 1, ia + int(self.SNAP_AFTER_S * cfg.fs_base))
        starts = np.arange(lo_j, hi_j, step)
        if len(starts) < 5 or ia not in starts:
            return None
        g = det._corr_grid(x, starts, det.tA, pc.coarse_nfft)
        f = np.fft.fftfreq(pc.coarse_nfft, 1.0 / cfg.fs_base)
        band = np.abs(f) <= pc.max_freq_off_hz
        o = np.argsort(f[band])
        rho = g[:, band][:, o].T                             # (주파수, 시간)
        fb = f[band][o]
        t = (starts - ia) / cfg.fs_base                      # 열 = 창 가운데 시각, 0 = 검출된 창
        # 검출기의 정밀 위치(시작, df)에서 정밀 ρ² 를 다시 잰다 — 검출기가 보고한 값과 같아야 한다
        r_start, r_df, r_rho = det.refine(bb, int(ia), float(req["df"]), "A")
        ci = int(np.nonzero(starts == ia)[0][0])
        fi = int(np.argmin(np.abs(fb - float(req["df"]))))
        return {"rho": rho, "f": fb, "t": t,
                "det": {"df": float(req["df"]), "rho": float(req.get("rho", np.nan)), "abs": a_abs},
                "check": {"map_at_det": float(rho[fi, ci]), "map_max": float(rho.max()),
                          "refine_rho": float(r_rho), "refine_df": float(r_df),
                          "refine_shift": int(r_start - ia), "col_of_det": ci, "row_of_det": fi},
                "utc": time.time()}

    def _run(self):
        while not self._stop.wait(self.period):
            src = self.source()
            if src is None:
                continue
            ring, fs = src
            try:
                out = self._tick(ring, fs)
                if out:
                    self.ready.emit(out)
            except Exception:
                pass

    def _bb(self, ring, fs, a, b):
        s0, raw = ring.read(a, b)
        if len(raw) < int(0.3 * fs):
            return None, s0
        x = raw.astype(np.float64)
        f = fs
        if f % self.cfg.fs_base != 0:
            from tngpkt.engine import resample, MODEM_FS
            x, f = resample(x, fs, MODEM_FS), MODEM_FS
        return audio_to_baseband(x, f, self.cfg), s0

    def _tick(self, ring, fs):
        cfg, pc = self.cfg, self.pc
        now = ring.count
        k = fs / cfg.fs_base
        out = {"kind": "live"}
        bb, s0 = self._bb(ring, fs, now - int(self.SPAN_S * fs), now)
        if bb is None:
            return None
        # 1) 슬라이딩 동기 맵 (코스타스 A 템플릿, 시간 × 주파수 오차)
        x = self.det._prefilter(bb)
        starts = np.arange(0, len(x) - self.det.n - 1, 8)
        if len(starts):
            g = self.det._corr_grid(x, starts, self.det.tA, pc.coarse_nfft)
            f = np.fft.fftfreq(pc.coarse_nfft, 1.0 / cfg.fs_base)
            band = np.abs(f) <= pc.max_freq_off_hz
            o = np.argsort(f[band])
            out["sync"] = {"rho": g[:, band][:, o].T, "f": f[band][o],
                           "t": (starts + self.det.n / 2 - len(x)) / cfg.fs_base}   # 0 = 지금
        # 1-2) 검출 스냅샷 (A 검출 알림이 있으면 한 번)
        with self._lock:
            req = getattr(self, "_snap_req", None)
            self._snap_req = None
        if req is not None:
            try:
                sn = self._snapshot(ring, fs, req)
            except Exception as ex:
                sn = {"error": "{}: {}".format(type(ex).__name__, ex)}
            if sn is not None:
                out["snap"] = sn
        # 2) 실시간 성상도 흐름
        with self._lock:
            trk = dict(self._trk) if self._trk else None
        if trk is not None:
            if time.time() - trk["t0"] > self.TRACK_S:
                self.untrack()
            else:
                pts = self._live_points(ring, fs, trk, now, k)
                if pts is not None:
                    out["live_pts"] = pts
        return out

    def _live_points(self, ring, fs, trk, now, k):
        cfg = self.cfg
        L, sps, S = cfg.frame_len, cfg.sps, cfg.symbols_per_frame
        n = self.det.n
        if trk["phase"] is None:
            a0 = trk["a"] - int(0.1 * fs)
            bb, s0 = self._bb(ring, fs, a0, trk["a"] + int((n / cfg.fs_base + 0.1) * fs))
            if bb is None:
                return None
            i = int(round((trk["a"] - s0) / k))
            t = (s0 / k + np.arange(len(bb))) / cfg.fs_base          # 절대 시각 기준 → 덩어리 사이 위상이 이어진다
            y = bb * np.exp(-2j * np.pi * trk["df"] * t)
            if i < 0 or i + n > len(y):
                return None
            ph = float(np.angle(np.sum(y[i:i + n] * np.conj(self.det.tA))))
            with self._lock:
                if self._trk is not None:
                    self._trk["phase"] = ph
                    self._trk["next"] = 0
            return None
        # A 끝 = 데이터(가드) 시작. 심볼 j 의 중앙 (기저대역 샘플, A 시작 기준)
        a_bb = trk["a"] / k
        lo, hi = now - int(0.4 * fs), now - int(0.05 * fs)
        bb, s0 = self._bb(ring, fs, lo, hi)
        if bb is None:
            return None
        base = s0 / k
        t = (base + np.arange(len(bb))) / cfg.fs_base
        y = bb * np.exp(-2j * np.pi * trk["df"] * t - 1j * trk["phase"])
        j0 = trk["next"] or 0
        j = np.arange(j0, j0 + 400)
        ctr = a_bb + n + (j // S) * L + (j % S) * sps + sps // 2
        idx = np.round(ctr - base).astype(int)
        ok = (idx >= 8) & (idx < len(y) - 8)
        if not np.any(ok):
            return None
        last = j[ok][-1]
        with self._lock:
            if self._trk is not None:
                self._trk["next"] = int(last + 1)
        p = y[idx[ok]]
        return p / max(np.sqrt(np.mean(np.abs(p) ** 2)), 1e-12)


class ZoomAnalyzer(QObject):
    """
    프로토콜 대역 확대 워터폴 (복조 스레드와 분리된 자체 스레드).
      입력(링버퍼) → 1500Hz 로 복소 믹싱 (절대 샘플 번호로 위상 연속) → 8차 저역 필터 (상태 유지)
      → 정수배 데시메이션 (약 2000Hz) → 길이 N 한 창 FFT 를 hop 마다 → dBFS (풀스케일 사인 = 0dBFS)
    분해능 프리셋: N 과 hop 을 함께 바꾼다 (주파수 분해능 = fs_z / N, 시간 간격 = hop / fs_z).
    결과 줄은 큐에 쌓고, 화면 쪽이 최대 20fps 로 가져간다.
    """
    PRESETS = {                      # 이름: (N, hop) at fs_z ≈ 2000Hz
        "시간 우선 (4Hz)": (512, 32),
        "균형 (2Hz)": (1024, 64),
        "주파수 우선 (1Hz)": (2048, 128),
    }
    DEFAULT = "균형 (2Hz)"
    CENTER = 1500.0

    def __init__(self, source, hz=20.0):
        super().__init__()
        self.source = source
        self.period = 1.0 / hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.rows = []                       # (dB 줄, 주파수) 쌓아 둔 것 — 화면이 take() 로 가져간다
        self.active = False                  # 패널이 보일 때만 계산한다
        self.set_preset(self.DEFAULT)
        self._reset()
        self.t = threading.Thread(target=self._run, daemon=True, name="zoom-analyzer")
        self.t.start()

    def stop(self):
        self._stop.set()

    SPEEDS = {"느림": 2.0, "보통": 1.0, "빠름": 0.5}      # 줄 사이 시간 배율
    speed = "보통"

    def set_preset(self, name):
        with self._lock:
            self.preset = name if name in self.PRESETS else self.DEFAULT
            N, hop = self.PRESETS[self.preset]
            self.N, self.hop = N, max(8, int(hop * self.SPEEDS[self.speed]))
            self._need_reset = True

    def set_speed(self, name):
        self.speed = name if name in self.SPEEDS else "보통"
        self.set_preset(self.preset)

    def _reset(self):
        self.fs_in = None
        self.cursor = None
        self.zi = None
        self.zbuf = np.zeros(0, dtype=np.complex128)
        self.since = 0
        self._need_reset = False

    def take(self):
        with self._lock:
            r, self.rows = self.rows, []
        return r

    def info(self):
        fsz = (self.fs_in / self.D) if self.fs_in else 2000.0
        return {"fs_z": fsz, "N": self.N, "hop": self.hop, "df_hz": fsz / self.N, "dt_s": self.hop / fsz,
                "fs_in": self.fs_in or 48000.0}

    def _setup(self, fs):
        self.fs_in = float(fs)
        self.D = max(1, int(round(fs / 2000.0)))
        fsz = fs / self.D
        self.sos = sg.butter(8, 0.8 * fsz / 2, fs=fs, output="sos")        # 통과 ±800Hz, 에일리어싱 방지
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.complex128)
        self.win = np.hanning(self.N)
        self.wsum = np.sum(self.win)

    def _run(self):
        while not self._stop.wait(self.period):
            src = self.source() if self.active else None
            if src is None:
                self.cursor = None                           # 다시 보이면 최근부터
                continue
            ring, fs = src
            try:
                self._step(ring, fs)
            except Exception:
                self._reset()

    def _step(self, ring, fs):
        with self._lock:
            need = self._need_reset or self.fs_in != float(fs)
        if need:
            self._reset()
            self._setup(fs)
        now = ring.count
        if self.cursor is None or now - self.cursor > int(2 * fs):       # 처음 · 한참 밀렸으면 최근 0.5초부터
            self.cursor = now - int(0.5 * fs)
            self.cursor -= self.cursor % self.D
            self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.complex128)
            self.zbuf = np.zeros(0, dtype=np.complex128)
        s0, raw = ring.read(self.cursor, now)
        if len(raw) == 0:
            return
        n_use = (len(raw) // self.D) * self.D
        if n_use == 0:
            return
        raw = raw[:n_use].astype(np.float64)
        idx = s0 + np.arange(n_use)
        mixed = raw * np.exp(-2j * np.pi * self.CENTER * idx / self.fs_in)
        y, self.zi = sg.sosfilt(self.sos, mixed, zi=self.zi)
        off = (-s0) % self.D                                              # 절대 번호가 D 의 배수인 샘플만
        z = y[off::self.D]
        self.cursor = s0 + n_use
        self.zbuf = np.concatenate([self.zbuf, z])[-(self.N + 64 * self.hop):]
        self.since += len(z)
        out, ends = [], []
        last_abs = s0 + off + (len(z) - 1) * self.D                      # zbuf 마지막 샘플의 입력 절대 번호
        fsz = self.fs_in / self.D
        freqs = self.CENTER + np.fft.fftshift(np.fft.fftfreq(self.N, 1.0 / fsz))
        while self.since >= self.hop and len(self.zbuf) >= self.N:
            end = len(self.zbuf) - (self.since - self.hop)
            if end < self.N:
                self.since -= self.hop
                continue
            seg = self.zbuf[end - self.N:end] * self.win
            ends.append(last_abs - (len(self.zbuf) - end) * self.D)        # 이 줄 창 끝의 입력 절대 번호
            X = np.fft.fftshift(np.fft.fft(seg))
            # 실수 풀스케일 사인(진폭 1)은 복소 믹싱 뒤 진폭 1/2 → |X| = wsum/2 → 0 dBFS 가 되게 2/wsum
            db = 20 * np.log10(np.abs(X) * 2.0 / self.wsum + 1e-12)
            out.append(db.astype(np.float32))
            self.since -= self.hop
        if out:
            with self._lock:
                self.rows.extend((r, freqs, e) for r, e in zip(out, ends))
                self.rows = self.rows[-400:]
