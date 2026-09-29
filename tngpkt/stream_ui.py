"""
7단계 스트리밍 형식의 UI 쪽 도우미. 엔진(link7 · modem5 복조 로직)은 고치지 않고 그 함수만 불러 쓴다.

  StreamPlan       프레임 수 → 구조 (구간 ↔ 정보 비트 ↔ 부호 비트 ↔ 송신 위치 ↔ 프레임)
  encode_bytes     바이트열 부호화 (link7.encode_stream 과 같은 결과 — 송신 중지 때 바이트 단위로 자르려고)
  stop_payload     송신 중지: 이미 나간 프레임과 어긋나지 않게 자를 바이트 수
  TxAudio          송신 오디오 + 시각 정보 (프레임 · 구간 진행 표시, 중지 이어 붙이기)
  LiveStream       실시간 수신: 코스타스 A 이후 도착하는 프레임을 StreamDecoder.feed() 로 흘려 구간 단위 결과
  assemble_text    구간 결과 → 표시 텍스트 (실패 구간 □)
  dissect_stream   패킷 해부 (새 형식)
"""

import codecs

import numpy as np

from tngpkt.link7 import (layout, coded_len, capacity, frames_for, interleaver_src, conv_encode_rows, SEG_BYTES,
                   CRC_BYTES, TAIL, GUARD_STREAM, OFFSETS, StreamDecoder, PLACEHOLDER, DEPTH, ILV_J, ILV_M,
                   GUARDS, LEN_BYTES, stream_bytes, encode_raw)
from tngpkt.link4 import BITS_PER_FRAME
from tngpkt.framing import crc16_ccitt

BPF = BITS_PER_FRAME
FRAME_S = 0.192
STREAM = "스트리밍"
BLOCK = "단일 블록"


def is_stream(info):
    return (info or {}).get("format") == STREAM


FAST_K, FAST_REF = 5, 10          # 빠른 신호 소실: 최근 5프레임 (0.96 s) · 그 앞 10프레임 중앙 = 끊기기 전 세기
FAST_STRONG = 15.0                # 끊기기 전 대역비 ≥ 15 dB (≈ SNR +6 dB 이상) 일 때만
FAST_NOISE, FAST_DROP = 4.0, 10.0 # 최근 5프레임 최대 < 4 dB (잡음 수준) 이고 전보다 10 dB 이상 떨어짐
# 측정 (results/remote/loss_fast.log): 살아 있는 신호 AWGN · Moderate · Poor × -12~+30 dB, 37 s × 100회씩 (21.6 시간) 오종료 0


def fast_lost(band):
    """강신호가 갑자기 끊김 (송신 즉시 중단 등) → True. 약한 신호는 False (기존 보수적 판단에 맡김)"""
    if len(band) < FAST_K + FAST_REF:
        return False
    ref = float(np.median(band[-FAST_K - FAST_REF:-FAST_K]))
    rec = max(band[-FAST_K:])
    return ref >= FAST_STRONG and rec < FAST_NOISE and rec < ref - FAST_DROP


def band_ratio_db(frame_bb):
    """
    프레임 창 (…, L) 기저대역 → (±250Hz 대역 평균 전력) / (300~900Hz 평균 전력) [dB].
    잡음만 있으면 약 0 dB. evaluate_longmsg.band_ratio_frames 와 같은 계산 (문턱은 그 측정으로 정했다).
    """
    x = np.asarray(frame_bb)
    L = x.shape[-1]
    X = np.abs(np.fft.fft(x, axis=-1)) ** 2
    f = np.abs(np.fft.fftfreq(L, 1.0 / 2000.0))
    pin = X[..., f <= 250].mean(-1)
    pout = X[..., (f >= 300) & (f <= 900)].mean(-1)
    return 10 * np.log10(pin / np.maximum(pout, 1e-20))


# ------------------------------------------------------------------ 구조
class StreamPlan:
    """
    데이터 프레임 n_frames 개인 스트리밍 메시지의 구조. 프레임 번호 0 = 첫 데이터 프레임 (가드 다음).
      seg_bits[k]   = (정보비트 시작, 끝, 페이로드 바이트 시작, 바이트 수)
      seg_frames[k] = 구간 k 의 부호 비트가 실린 프레임들 (인터리버 때문에 약 ±3.2초에 흩어진다)
      seg_nominal[k]= (시작, 끝) 프레임 단위 실수 — 인터리빙 전 순서의 위치 (그림에서 구간을 놓는 자리)
    """

    def __init__(self, n_frames):
        self.n = int(n_frames)
        self.cap = capacity(self.n)
        self.lay = layout(self.cap)
        self.n_coded = coded_len(self.cap)
        self.T = self.n_coded // 2
        self.q = interleaver_src(self.n_coded)                   # 부호 비트 p → 송신 위치 q[p]
        self.seg_bits, ib = [], 0
        for s, n in self.lay:
            self.seg_bits.append((ib, ib + 8 * (n + CRC_BYTES), s, n))
            ib += 8 * (n + CRC_BYTES)
        fr = self.q // BPF
        self.seg_frames = [np.unique(fr[2 * a:2 * e]) for a, e, _, _ in self.seg_bits]
        self.seg_first = np.array([f.min() for f in self.seg_frames])
        self.seg_last = np.array([f.max() for f in self.seg_frames])
        self.seg_nominal = [(2 * a / BPF, 2 * e / BPF) for a, e, _, _ in self.seg_bits]
        self.n_seg = len(self.seg_bits)

    def frame_segments(self, f):
        return [k for k, fr in enumerate(self.seg_frames) if f in fr]

    def seg_state_tx(self, frames_done, frames_started):
        """송신 진행: 0 대기 / 1 송신 중 (비트 일부 나감) / 2 송신 완료"""
        st = np.zeros(self.n_seg, dtype=int)
        st[self.seg_first < frames_started] = 1
        st[self.seg_last < frames_done] = 2
        return st


def encode_bytes(P):
    """바이트열 → (데이터 프레임 (n, 96), 정보). link7.encode_stream(text) 과 같은 규칙 (NUL 채움 · 채움 비트)."""
    P = bytes(P) or b" "
    nf = frames_for(len(P))
    cap = capacity(nf)
    Pp = P + b"\x00" * (cap - len(P))
    info = []
    for s, n in layout(len(Pp)):
        d = Pp[s:s + n]
        info += [np.unpackbits(np.frombuffer(d, np.uint8)).astype(np.int64),
                 np.unpackbits(np.frombuffer(crc16_ccitt(d).to_bytes(CRC_BYTES, "big"), np.uint8)).astype(np.int64)]
    info = np.concatenate(info + [np.zeros(TAIL, np.int64)])
    c = conv_encode_rows(info)[0]
    q = interleaver_src(len(c))
    y = np.empty_like(c)
    y[q] = c
    fill = np.random.default_rng(0xF111).integers(0, 2, nf * BPF - len(y))
    frames = np.concatenate([y, fill]).reshape(nf, BPF).astype(np.float32)
    return frames, {"n_data": nf, "payload": len(P), "capacity": cap, "n_segments": len(layout(cap)),
                    "n_coded": len(c), "Pp": Pp}


def stop_payload(P, plan, f_switch):
    """
    송신 중지: 프레임 f_switch 부터 새 메시지로 갈아탄다. 이미 나간 프레임(< f_switch)에 비트가 하나라도 실린
    구간까지만 남긴다. 지연순 인터리버는 끝 부분 말고는 송신 순서가 길이와 무관하므로, 이렇게 자른 메시지의
    앞 f_switch 프레임은 원래 메시지와 비트가 똑같다 (stream_ui 자가검사로 확인).
    반환: 남길 바이트 수 (원래 길이와 같으면 자를 필요 없음)
    """
    started = np.nonzero(plan.seg_first < f_switch)[0]
    k = int(started.max()) if len(started) else 0
    return min(len(P), SEG_BYTES * (k + 1))


# ------------------------------------------------------------------ 송신 오디오
class TxAudio:
    """
    송신 오디오 한 건과 시각 정보. 오디오 = 엔진 송신 경로 그대로 (modem5 송신 NN · 코스타스 조립 · 오디오 변환).
    frame_t0 : 첫 데이터 프레임 시작 시각 [s, 오디오 시작 기준]
    """

    def __init__(self, engine, text=None, fs=48000.0, P=None, gain=None, S=None):
        from tngpkt.modem5 import _frames_to_baseband
        from tngpkt.preamble import assemble
        from tngpkt.modem import baseband_to_audio
        from tngpkt.engine import resample, MODEM_FS
        cfg, pc = engine.cfg, engine.pc
        self.text = text if text is not None else P.decode("utf-8", errors="replace")
        self.P = P if P is not None else self.text.replace("\x00", "").encode("utf-8")
        # S = 구간에 실을 바이트열 (v2: 길이 2바이트 + 페이로드). 송신 중지 때는 원래 길이를 둔 채 잘린 S 를 받는다
        self.S = S if S is not None else stream_bytes(self.P, 2)
        frames, si = encode_raw(self.S)
        si["Pp"] = self.S + b"\x00" * (si["capacity"] - len(self.S))
        with engine._lock:
            data = _frames_to_baseband(frames, GUARDS["stream2"], engine.model, cfg, engine.device)
        bb, info = assemble(pc, cfg.fs_base, data)
        fs_m = fs if (fs % cfg.fs_base == 0) else MODEM_FS
        a = baseband_to_audio(bb, cfg, fs_m)                  # 앞뒤 0.2초 무음
        a = resample(a, fs_m, fs)
        self.raw = a
        self.gain = gain if gain is not None else 10 ** (-3 / 20) / max(np.max(np.abs(a)), 1e-12)
        self.audio = a * self.gain
        self.fs = float(fs)
        self.frames, self.si = frames, si
        self.plan = StreamPlan(si["n_data"])
        self.lead = 0.2
        fsb = cfg.fs_base
        self.t_data = self.lead + info["data_start"] / fsb                   # 가드 프레임 시작
        self.frame_t0 = self.t_data + cfg.frame_len / fsb
        self.t_a = self.lead + info["a_start"] / fsb
        self.t_b = self.lead + info["b_start"] / fsb
        self.t_tone = self.lead
        self.dur = len(self.audio) / self.fs
        self.frame_s = cfg.frame_len / fsb
        self.data_bb = data

    def progress(self, t):
        """경과 t [s] → (프레임 완료 수, 시작된 프레임 수, 구간 상태 배열)"""
        u = (t - self.frame_t0) / self.frame_s
        done = int(np.clip(np.floor(u), 0, self.plan.n))
        started = int(np.clip(np.ceil(u), 0, self.plan.n))
        return done, started, self.plan.seg_state_tx(done, started)

    def char_segments(self):
        """글자마다 (글자 위치, 첫 바이트 구간, 끝 바이트 구간)"""
        out, b = [], LEN_BYTES                                   # 첫 2바이트 = 길이 필드
        for i, ch in enumerate(self.text):
            n = len(ch.encode("utf-8"))
            out.append((i, b // SEG_BYTES, (b + n - 1) // SEG_BYTES))
            b += n
        return out

    def stopped(self, engine, t_now, margin_s=0.35):
        """
        중지: t_now 뒤 margin_s 이후 첫 프레임 경계에서 잘린 메시지로 갈아탄다.
        반환: (새 TxAudio, 이어 붙일 오디오 샘플 번호) 또는 (None, None) — 이미 끝 부분이면 그냥 다 보낸다.
        """
        f_sw = int(np.ceil((t_now + margin_s - self.frame_t0) / self.frame_s))
        if f_sw >= self.plan.n - 1:
            return None, None
        f_sw = max(f_sw, 1)
        keep = stop_payload(self.S, self.plan, f_sw)
        if keep >= len(self.S):
            return None, None
        # 길이 필드(원래 길이)는 이미 나간 첫 구간에 있으므로 그대로 둔다 → 수신측은 B 로 끝을 알고 '송신 중지' 로 판정
        new = TxAudio(engine, self.text, self.fs, P=self.P, gain=self.gain, S=self.S[:keep])
        # 이어 붙이는 자리: 프레임 f_sw 시작보다 송신 LPF 반폭 + 여유(80ms) 앞. 두 오디오가 여기까지는 같아야 한다.
        t_sw = self.frame_t0 + f_sw * self.frame_s - 0.08
        i_sw = int(round(t_sw * self.fs))
        # 송신 NN 의 전력 정규화 때문에 두 오디오는 크기가 아주 조금 다를 수 있다 → 겹치는 0.5초로 배율을 맞춘다
        i0 = max(0, i_sw - int(0.5 * self.fs))
        x, y = self.audio[i0:i_sw], new.audio[i0:i_sw]
        c = float(np.dot(x, y) / max(np.dot(y, y), 1e-20))
        new.audio = new.audio * c
        new.gain *= c
        new.splice_err = float(np.max(np.abs(x - c * y)) / max(np.max(np.abs(x)), 1e-12))
        new.f_switch = f_sw
        return new, i_sw


# ------------------------------------------------------------------ 실시간 수신 (코스타스 B 전)
class LiveStream:
    """
    코스타스 A 검출 뒤 들어오는 프레임을 차례로 복조한다. 메시지 길이(프레임 수)를 아직 모르므로 최대 길이로
    StreamDecoder 를 만든다 — 지연순 인터리버는 끝 3.2초 말고는 송신 순서가 길이와 무관해서 앞 구간 결과는 같다.

    끝내는 방법 (우선순위: 코스타스 B > 길이 필드 > 신호 소실; B 는 realtime_rx 가 따로 본다)
      length : v2 첫 구간의 길이로 프레임 수를 알고 그만큼 받으면, 받아 둔 LLR 로 정확한 길이 복조를 다시 해 확정
      loss   : 대역 에너지가 잡음 수준으로 LOSS_T 프레임 + 최근 LOSS_K 구간 연속 실패 + 평균 |LLR| 잡음 수준
               (문턱 근거: evaluate_longmsg.py --part loss, README 9단계)
    """
    LOSS_T = 13              # 프레임 (2.5 s). 측정: 8 프레임은 페이딩 오종료 5/162, 13 프레임은 0/160
    LOSS_K = 1               # K 를 늘리면 오종료는 같고 종료만 늦다 (K=1 중앙 2.5 s · K=3 중앙 4.8 s)
    LOSS_E_DB = 1.5          # 대역 에너지 비 문턱 [dB] (잡음만 있으면 약 0 dB)
    LOSS_LLR = 2.85          # 평균 |LLR| 문턱 = 잡음 99.9% 분위 (측정, 신호 중앙 6.0)
    # 가짜 시작 (코스타스 A 오검출): 가드 패턴이 어느 형식과도 뚜렷하지 않고 + 첫 구간 실패 + |LLR| 잡음 수준
    # → 대역 에너지와 무관하게 폐기. 가드 뚜렷함 = 최선 가설의 상관 / 그 가설의 Σ|LLR| (참 신호 ≈ 1, 잡음 ≈ 0)
    FAKE_GUARD_Q = 0.5
    # 끔: -6.5 dB Watterson 에서 진짜 메시지 오폐기 Moderate 16.7% · Poor 25% (fake_rate.py, 조건별 60회).
    # 대신 첫 CRC16 통과 전에는 수신 패널에 표시하지 않는다 (neuromod_app, 송신 비용 · 성능 손실 0)
    FAKE_ENABLED = False
    BLOCK_GUARD_Q = 0.5      # 단일 블록 인정: 가드 뚜렷함 (참 신호 ≈ 1, 잡음 ≈ 0 — FAKE_GUARD_Q 와 같은 기준)

    def __init__(self, engine, a_abs, df, fs, n_max):
        self.eng, self.cfg = engine, engine.cfg
        self.fs = float(fs)
        self.k = self.fs / self.cfg.fs_base                       # 기저대역 1샘플 = 입력 k 샘플
        n_costas = int(round(engine.pc.symbol_ms * 1e-3 * engine.pc.n_tones * self.cfg.fs_base))
        L = self.cfg.frame_len
        self.a_abs, self.df = a_abs, df
        self.f0_abs = a_abs + (n_costas + L) * self.k           # 첫 데이터 프레임 시작 (입력 절대 번호)
        self.dec = StreamDecoder(n_max)
        self.fed = 0
        self.format = None                                       # 'stream2' / 'stream1' / 'block'
        self.version = None
        self.segments = []
        self.done = False
        self.end_reason = None                                   # 'length' / 'loss'
        self.length = None
        self.n_true = None
        self.final = None                                        # 길이 확정 뒤 (텍스트, 부분, 정보)
        self._llr = []                                           # 프레임별 가설 LLR (H, 96) — 길이 확정 복조용
        self.guard_q = None                                      # 가드 패턴 뚜렷함 (0 ~ 1)
        self.band = []                                           # 프레임별 대역 에너지 비 [dB]
        self.llr_lv = []                                         # 프레임별 평균 |LLR| (현재 정렬 가설)

    def _windows(self, ring, f_from, f_to):
        """프레임 [f_from, f_to) (가드 = -1) 의 (가설별 LLR (H, k, 96), 대역 에너지 비 (k,)). 오디오가 아직 없으면 None"""
        from tngpkt.modem4 import _rx_windows
        from tngpkt.modem import audio_to_baseband
        from tngpkt.engine import resample, MODEM_FS
        cfg, k = self.cfg, self.k
        L, W, C = cfg.frame_len, cfg.win_len, cfg.ctx_len
        mo = int(np.abs(OFFSETS).max())
        need_end = self.f0_abs + ((f_to * L) + (W - C) + mo + 64) * k
        if ring.count < need_end:
            return None
        x0 = int(self.f0_abs + (f_from * L - C - mo - 96) * k)
        s0, raw = ring.read(x0, int(need_end))
        if len(raw) < int((need_end - x0) * 0.98):
            return None
        fs = self.fs
        x = raw.astype(np.float64)
        if fs % cfg.fs_base != 0:
            x = resample(x, fs, MODEM_FS)
            fs = MODEM_FS
        bb = audio_to_baseband(x, fs, cfg)
        t = np.arange(len(bb)) / cfg.fs_base
        bb = bb * np.exp(-2j * np.pi * self.df * t)
        base = (self.f0_abs - s0) / k                             # 첫 데이터 프레임의 bb 위치 (실수)
        offs = np.asarray(OFFSETS)
        f = np.arange(f_from, f_to)
        p0 = np.round(base + f * L).astype(np.int64)
        st = (p0[None, :] + offs[:, None] - C).reshape(-1)
        st = np.clip(st, 0, max(0, len(bb) - W))
        win = bb[st[:, None] + np.arange(W)[None, :]]
        with self.eng._lock:
            llr = _rx_windows(win, self.eng.model, self.eng.device)
        fr = bb[np.clip(p0, 0, max(0, len(bb) - L))[:, None] + np.arange(L)[None, :]]
        return llr.reshape(len(offs), len(f), BPF), band_ratio_db(fr)

    def step(self, ring):
        """새로 도착한 프레임을 흘려 넣는다. 반환: 이번에 결정된 구간 목록 (길이 확정 시에는 확정된 전체 목록)"""
        if self.done:
            return []
        if self.format is None:
            got = self._windows(ring, -1, 0)
            if got is None:
                return []
            g = got[0]
            corr = {kk: float((g * (2.0 * v - 1.0)).sum(axis=(1, 2)).max()) for kk, v in GUARDS.items()}
            self.format = max(corr, key=corr.get)
            v = GUARDS[self.format]
            q = (g * (2.0 * v - 1.0)).sum(axis=(1, 2)) / np.maximum(np.abs(g).sum(axis=(1, 2)), 1e-9)
            self.guard_q = float(q.max())
            if self.format == "block" and self.guard_q < self.BLOCK_GUARD_Q:
                # 옛 형식(단일 블록) 은 가드 상관이 뚜렷할 때만. 아니면 스트리밍 가설로 계속 (가짜 시작은 CRC 통과 없이 끝남)
                self.format = max((kk for kk in corr if kk != "block"), key=corr.get)
                v = GUARDS[self.format]
                q = (g * (2.0 * v - 1.0)).sum(axis=(1, 2)) / np.maximum(np.abs(g).sum(axis=(1, 2)), 1e-9)
                self.guard_q = float(q.max())
            if self.format == "block":
                self.done = True
                return []
            self.version = 2 if self.format == "stream2" else 1
        out = []
        while self.fed < self.dec.nf and not self.done:
            got = self._windows(ring, self.fed, self.fed + 1)
            if got is None:
                break
            llr, ratio = got
            self._llr.append(llr[:, 0])
            self.band.append(float(ratio[0]))
            h = int(np.argmin(np.abs(np.asarray(OFFSETS) - self.dec.center)))
            self.llr_lv.append(float(np.abs(llr[h, 0]).mean()))
            new = self.dec.feed(llr)
            self.fed += 1
            out += new
            self.segments += new
            # v2: 첫 구간이 통과하면 길이 → 전체 프레임 수
            if self.version == 2 and self.n_true is None:
                s0 = next((r for r in self.segments if r["index"] == 0), None)
                if s0 is not None and s0["ok"]:
                    self.length = int.from_bytes(s0["data"][:LEN_BYTES], "big")
                    self.n_true = frames_for(LEN_BYTES + self.length)
            if self.n_true is not None and self.fed >= self.n_true:
                return self._finish_length()
            if self._fake():
                self.done, self.end_reason = True, "fake"
                return out
            if self._lost():
                self.done, self.end_reason = True, "loss"
        return out

    def _finish_length(self):
        """길이만큼 받았다: 정확한 프레임 수로 다시 복조해 끝 부분 구간까지 확정한다 (B 를 기다리지 않음)"""
        d = StreamDecoder(self.n_true)
        d.feed(np.stack(self._llr[:self.n_true], axis=1))
        d.finish()
        self.final = d.result(version=2)
        self.segments = sorted(d.segments, key=lambda r: r["index"])
        self.done, self.end_reason = True, "length"
        return list(self.segments)

    def _fake(self):
        """가짜 시작: 가드 뚜렷하지 않음 + 첫 구간 결정됨 · 실패 + 지금까지 평균 |LLR| 잡음 수준 (에너지 조건 없음)"""
        if not self.FAKE_ENABLED or self.guard_q is None or self.guard_q >= self.FAKE_GUARD_Q or \
                any(r["ok"] for r in self.segments):
            return False
        s0 = next((r for r in self.segments if r["index"] == 0), None)
        return s0 is not None and not s0["ok"] and float(np.mean(self.llr_lv)) < self.LOSS_LLR

    def _lost(self):
        """신호 소실: 대역 에너지 잡음 수준 LOSS_T 프레임 + 최근 LOSS_K 구간 연속 실패 + 평균 |LLR| 잡음 수준"""
        if fast_lost(self.band):
            self.fast_loss = True
            return True
        T = self.LOSS_T
        if len(self.band) < T:
            return False
        if max(self.band[-T:]) >= self.LOSS_E_DB or np.mean(self.llr_lv[-T:]) >= self.LOSS_LLR:
            return False
        dec = sorted(self.segments, key=lambda r: r["index"])
        return len(dec) >= self.LOSS_K and not any(r["ok"] for r in dec[-self.LOSS_K:])

    def end_abs(self):
        """길이로 끝났을 때 데이터 끝 (가드 프레임 끝) 의 입력 절대 번호"""
        L = self.cfg.frame_len
        return self.f0_abs + (self.n_true + 1) * L * self.k


def collapse_tail(text):
    """신호 소실: 끝에 이어진 실패 구간 □□□… 를 □ 하나 + 안내로 줄인다. 반환: (앞 텍스트, 끝 실패 구간 수)"""
    body = text.rstrip(PLACEHOLDER)
    return body, len(text) - len(body)


def assemble_text(segs, n_known=None, pending_mark="", version=1):
    """
    구간 결과 [{'index','ok','data'}] → 표시 텍스트. 실패 구간은 □, 구간 경계에서 잘린 UTF-8 은 다음 구간과 이어 푼다.
    반환: (텍스트, 글자별 구간 번호 목록)
    """
    segs = sorted(segs, key=lambda r: r["index"])
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text, owner = [], []
    expect = 0
    for r in segs:
        if r["index"] != expect:
            break                                                 # 순서대로만 (중간이 비면 거기서 멈춤)
        expect += 1
        if r["ok"]:
            data = r["data"][LEN_BYTES:] if (version == 2 and r["index"] == 0) else r["data"]
            end = data.find(b"\x00")
            if end >= 0:
                data = data[:end]
            s = dec.decode(data, final=False)
            text.append(s)
            owner += [r["index"]] * len(s)
            if end >= 0:
                break
        else:
            dec.reset()
            text.append(PLACEHOLDER)
            owner.append(r["index"])
    return "".join(text) + pending_mark, owner


# ------------------------------------------------------------------ 패킷 해부 (새 형식)
def dissect_stream(res, cfg, pc, det, common):
    """
    dissect.py 가 만든 공통 층(오디오 · 기저대역 · 구간 · LLR 송신 순서) 위에 새 형식 층을 얹는다.
    common: dissect 공통 결과 dict (n, llr_tx, t_bit, ...)
    """
    m = res.viz.get("msg") or {}
    info = m.get("info") or {}
    n = common["n"]
    plan = StreamPlan(n)
    llr_tx = common["llr_tx"]
    ntx = n * BPF
    q = plan.q
    llr_coded = llr_tx[q]
    T = plan.T
    segs = sorted(info.get("segments") or [], key=lambda r: r["index"])
    # 정보 비트 (통과 구간 = 복호 값, 실패 구간 = 모름 -1, 꼬리 = 0)
    info_bits = np.full(T, -1, dtype=np.int64)
    info_bits[T - TAIL:] = 0
    byts = np.zeros(T // 8, dtype=np.int64)
    kinds = ["unk"] * (T // 8)
    seg_rows = []
    for k, (a, e, s, nb) in enumerate(plan.seg_bits):
        r = next((x for x in segs if x["index"] == k), None)
        ok = bool(r and r["ok"])
        row = {"k": k, "ok": ok, "decided": r is not None, "offset": None if r is None else r["offset"],
               "relock": bool(r and r.get("relock")), "score": None if r is None else r.get("score"),
               "bytes": (s, nb), "info": (a, e), "frames": plan.seg_frames[k], "crc_rx": None, "crc_calc": None,
               "text": ""}
        if ok:
            data = r["data"]
            crc = crc16_ccitt(data)
            full = data + crc.to_bytes(CRC_BYTES, "big")
            bits = np.unpackbits(np.frombuffer(full, np.uint8)).astype(np.int64)
            info_bits[a:e] = bits
            b0 = a // 8
            byts[b0:b0 + len(full)] = np.frombuffer(full, np.uint8)
            for i in range(nb):
                kinds[b0 + i] = "pad" if data[i] == 0 else "body"
            if k == 0 and info.get("version") == 2:                           # v2: 첫 2바이트 = 길이 필드
                kinds[b0] = kinds[b0 + 1] = "len"
            kinds[b0 + nb] = kinds[b0 + nb + 1] = "crc"
            row["crc_rx"] = row["crc_calc"] = crc
            body = data[LEN_BYTES:] if (k == 0 and info.get("version") == 2) else data
            row["text"] = body.split(b"\x00")[0].decode("utf-8", errors="replace")
        seg_rows.append(row)
    known_info = info_bits >= 0
    rec = conv_encode_rows(np.where(known_info, info_bits, 0))[0]           # 재부호화 (모르는 비트는 0 가정)
    # 부호 비트 p (단계 p//2) 가 확인 가능 = 그 단계와 앞 6단계 정보 비트를 모두 안다
    kn = known_info.astype(np.int64)
    win = np.convolve(kn, np.ones(TAIL + 1, dtype=np.int64), mode="full")[:T] == (TAIL + 1)
    win[:TAIL] = kn.cumsum()[:TAIL] == np.arange(1, TAIL + 1)
    known_coded = np.repeat(win, 2)[:plan.n_coded]
    hard = (llr_coded > 0).astype(np.int64)
    fixed_coded = np.nonzero(known_coded & (hard != rec))[0]
    fixed_info = np.unique(fixed_coded // 2)
    err_tx = np.zeros(ntx, dtype=bool)
    err_tx[q[fixed_coded]] = True
    known_tx = np.zeros(ntx, dtype=bool)
    known_tx[q[known_coded]] = True
    frame_fixed = err_tx.reshape(n, BPF).sum(axis=1)
    frame_known = known_tx.reshape(n, BPF).sum(axis=1)
    # 텍스트
    text, _ = assemble_text([{"index": r["k"], "ok": r["ok"], "data": (b"" if not r["ok"] else
                             bytes(byts[r["info"][0] // 8:r["info"][0] // 8 + r["bytes"][1]].astype(np.uint8)))}
                             for r in seg_rows if r["decided"]], version=info.get("version", 1))
    n_ok = sum(r["ok"] for r in seg_rows)
    used = info.get("segments_used", plan.n_seg)
    d = dict(common)
    d.update({
        "format": STREAM, "plan": plan, "llr_coded": llr_coded, "tx_of_coded": q,
        "info_bits": info_bits, "n_info": T, "fixed_coded": fixed_coded, "fixed_info": fixed_info,
        "err_tx": err_tx, "frame_fixed": frame_fixed, "frame_known": frame_known,
        "bytes": byts.astype(np.uint8), "byte_kinds": kinds, "segments": seg_rows, "text": text,
        "crc": {"ok": n_ok == used, "rx": None, "calc": None, "reason": "구간 {}/{} 통과".format(n_ok, used)},
        "drift_ppm": info.get("drift_ppm"), "relocks": info.get("relocks", 0), "used": used,
        "known_coded": int(known_coded.sum()),
    })
    d["summary"].update({"info_bits": T, "fixed": int(len(fixed_coded)), "nbytes": T // 8, "plen": None,
                         "segments_ok": n_ok, "segments_used": used, "known_coded": int(known_coded.sum())})
    return d


def stream_truth(llr, info):
    """
    패킷 분석용: 통과 구간으로 송신 비트를 재구성 → (프레임 비트 정답 (n,96) 0/1, 확인 가능 마스크 (n,96))
    FEC 정정 비트 = 확인 가능한 비트 중 hard 판정 불일치 수.
    """
    n = llr.shape[0]
    plan = StreamPlan(n)
    T = plan.T
    info_bits = np.full(T, -1, dtype=np.int64)
    info_bits[T - TAIL:] = 0
    for r in info.get("segments") or []:
        if r["ok"]:
            a, e, _, _ = plan.seg_bits[r["index"]]
            full = r["data"] + crc16_ccitt(r["data"]).to_bytes(CRC_BYTES, "big")
            info_bits[a:e] = np.unpackbits(np.frombuffer(full, np.uint8))
    kn = (info_bits >= 0).astype(np.int64)
    rec = conv_encode_rows(np.where(kn > 0, info_bits, 0))[0]
    win = np.convolve(kn, np.ones(TAIL + 1, dtype=np.int64), mode="full")[:T] == (TAIL + 1)
    win[:TAIL] = kn.cumsum()[:TAIL] == np.arange(1, TAIL + 1)
    known_coded = np.repeat(win, 2)[:plan.n_coded]
    truth = np.zeros(n * BPF, dtype=np.int64)
    mask = np.zeros(n * BPF, dtype=bool)
    truth[plan.q] = rec
    mask[plan.q] = known_coded
    return truth.reshape(n, BPF), mask.reshape(n, BPF)


def packet_timeline(n_data, has_tone, has_a, has_b, pc, cfg, frame_status, plan=None, seg_rows=None, ok=None):
    """
    패킷 타임라인 두 줄: {'frames': [(이름, 시작 ms, 길이 ms, 종류, 라벨)], 'segments': [...], 'seg_frames': {k: [프레임]}}
    프레임 줄 이름 'F{i}' 는 기존 타임라인과 같다. 구간 줄은 인터리빙 전 순서 위치에 놓는다.
    """
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
    t_data = t
    for i in range(n_data):
        st, lab = frame_status[i] if i < len(frame_status) else ("unk", "")
        out.append(("F{}".format(i + 1), t, fr, st, lab))
        t += fr
    out.append(("G", t, fr, "guard", "G"))
    t += fr
    out.append(("B", t, cost, "costas", "B") if has_b else ("B?", t, cost, "missing", "B 없음"))
    segs, seg_frames = [], {}
    if plan is not None:
        for k in range(plan.n_seg):
            a, b = plan.seg_nominal[k]
            r = seg_rows[k] if seg_rows is not None and k < len(seg_rows) else None
            kind = "pending" if r is None else ("ok" if r.get("ok") else ("fail" if r.get("decided", True) else "pending"))
            segs.append(("S{}".format(k + 1), t_data + a * fr, (b - a) * fr, kind, str(k + 1)))
            seg_frames[k] = [int(f) for f in plan.seg_frames[k]]
    else:
        segs.append(("패킷", t_data, n_data * fr, "ok" if ok else "fail", "패킷 전체 CRC16 1개 (단일 블록)"))
    return {"frames": out, "segments": segs, "seg_frames": seg_frames}


# ------------------------------------------------------------------ 송신 재생
class TxPlayer:
    """
    송신 오디오 재생 (출력 콜백이 버퍼에서 꺼내 간다). splice() 로 아직 안 나간 뒷부분을 바꿀 수 있다 (송신 중지).
    dry=True: 장치 없이 advance(n) 으로 위치만 옮긴다 (가상 루프백 시험용).
    lead_s (18부): 송신 오디오 앞 여유 — PTT 켠 뒤 소리까지 무음 (송신 지연), lead_tone=(Hz, 진폭) 이면 그 톤 (VOX 앞 여유).
    t · dur · splice 위치는 lead 를 뺀 송신 오디오 기준 (진행 표시 · 중지 계산 그대로).
    """

    def __init__(self, audio, fs, device=None, dry=False, lead_s=0.0, lead_tone=None):
        import threading
        self.fs = float(fs)
        a = np.asarray(audio, dtype=np.float32)
        self.lead = int(round(max(0.0, float(lead_s)) * self.fs))
        if self.lead:
            pad = np.zeros(self.lead, dtype=np.float32)
            if lead_tone is not None:
                f0, amp = lead_tone
                n = np.arange(self.lead)
                ramp = np.minimum(1.0, np.minimum(n, self.lead - 1 - n) / max(1.0, 0.005 * self.fs))
                pad = (amp * ramp * np.sin(2 * np.pi * f0 * n / self.fs)).astype(np.float32)
            a = np.concatenate([pad, a])
        self.buf = a
        self.pos = 0
        self.done = False
        self.dry = dry
        self._lock = threading.Lock()
        self.stream = None
        if not dry:
            import sounddevice as sd
            self.stream = sd.OutputStream(device=device, samplerate=self.fs, channels=1,
                                          blocksize=int(self.fs * 0.02), callback=self._cb,
                                          finished_callback=self._fin)

    def start(self):
        if self.stream is not None:
            self.stream.start()

    def _cb(self, outdata, frames, t, status):
        import sounddevice as sd
        with self._lock:
            chunk = self.buf[self.pos:self.pos + frames]
            self.pos += len(chunk)
        outdata[:len(chunk), 0] = chunk
        outdata[len(chunk):, 0] = 0.0
        if len(chunk) < frames:
            raise sd.CallbackStop

    def _fin(self):
        self.done = True

    def advance(self, n):
        """dry 모드: n 샘플 '재생' → 그 오디오를 돌려준다"""
        with self._lock:
            chunk = self.buf[self.pos:self.pos + n].copy()
            self.pos += len(chunk)
            if self.pos >= len(self.buf):
                self.done = True
        return chunk

    @property
    def t(self):
        return max(0, self.pos - self.lead) / self.fs

    @property
    def dur(self):
        return (len(self.buf) - self.lead) / self.fs

    def splice(self, i, new_audio, margin_s=0.05):
        """송신 오디오 샘플 i 부터 new_audio[i:] 로 바꾼다. 이미 나갔거나 곧 나갈 자리면 False"""
        with self._lock:
            j = i + self.lead
            if j < self.pos + int(margin_s * self.fs):
                return False
            self.buf = np.concatenate([self.buf[:j], np.asarray(new_audio[i:], dtype=np.float32)])
            return True

    def abort(self):
        self.done = True
        if self.stream is not None:
            try:
                self.stream.abort()
                self.stream.close()
            except Exception:
                pass

    def close(self):
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass


if __name__ == "__main__":
    from tngpkt import _io_utf8  # noqa: F401
    from tngpkt.link7 import encode_stream
    for t in ("HI", "CQ CQ DE HL1ABC HL1ABC PSE KN.", ("한글 시험 문장입니다. ABC 123. " * 30)[:500]):
        a, _ = encode_stream(t, version=1)
        b, _ = encode_bytes(t.encode())
        assert np.array_equal(a, b), "encode_bytes 가 encode_stream 과 다르다"
    # 중지: 잘린 메시지의 앞 f_sw 프레임이 원래 메시지와 같은가
    t = ("긴 메시지 중지 시험. 가나다라마바사 0123456789 " * 30)[:500]
    P = t.encode()
    fr, si = encode_bytes(P)
    plan = StreamPlan(si["n_data"])
    bad = 0
    for f_sw in range(1, si["n_data"] - 1, 7):
        keep = stop_payload(P, plan, f_sw)
        fr2, si2 = encode_bytes(P[:keep])
        same = np.array_equal(fr[:f_sw], fr2[:f_sw])
        bad += int(not same)
    assert bad == 0, "중지 메시지가 이미 나간 프레임과 어긋난다 ({}건)".format(bad)
    print("  encode_bytes = encode_stream, 중지 절단 {}개 지점 모두 앞 프레임 일치".format(len(range(1, si["n_data"] - 1, 7))))
    txt, own = assemble_text([{"index": 0, "ok": True, "data": "가나".encode()[:4]},
                              {"index": 1, "ok": True, "data": "가나".encode()[4:] + b"\x00"}])
    assert txt == "가나", txt
    print("stream_ui 자가 검사 통과")
