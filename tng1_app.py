"""
TNG1 앱 연결 (송신 오디오 · 실시간 수신 스레드 · 3모드 묶음 · 타임라인 · 워터폴 블록). 엔진은 tng1.py (TNG1 1.0 / wire rev 1).

수신 원칙: CRC16 통과 + 표시 규칙 (tng1.receive) 을 지난 블록만 화면에 보낸다. 첫 블록이 통과할 때 'start' 사건을 낸다
(그 전에는 아무 표시 없음). 같은 격자 (블록 주기 위상) · 주파수 (±8 Hz) 의 연속 블록 = 한 메시지.
메시지 끝 = EOT 블록, 또는 마지막 블록 끝에서 2.5 블록 동안 새 블록 없음 (신호 소실). 반복 합치기로 풀린 블록은 "· N회 결합".
"""
import math
import threading
import time
import unicodedata
from fractions import Fraction

import numpy as np

import charset1
import tng1
from realtime_rx import RingBuffer

NAME = "TNG1"
C = tng1.CFGS["TNG1"]
FC = 1500.0
BLOCK_S = C.block_s


# ====================================================================== 송신
def bad_char_names(bad):
    """문자표 밖 글자 → 보이는 이름 (보이지 않는 글자는 'Line break' 처럼 이름으로)"""
    out = []
    for ch in bad:
        if ch.isprintable() and not ch.isspace():
            out.append("'{}'".format(ch))
        else:
            nm = {"\n": "Line break", "\t": "Tab", "\r": "Carriage return", " ": "No-break space"}.get(ch)
            out.append(nm or unicodedata.name(ch, "U+{:04X}".format(ord(ch))).title())
    return out


def normalize(text):
    """→ (보낼 문자열, 집합 밖 글자 목록). 소문자 → 대문자, \\r\\n → 줄바꿈, 탭 → 공백 (charset1.normalize)"""
    return charset1.normalize(text)


class Plan1:
    """송신 진행 · 타임라인용: 블록 = '구간' (n = n_seg = 블록 수)"""

    def __init__(self, n_blocks):
        self.n = self.n_seg = int(n_blocks)
        self.seg_first = np.arange(self.n)
        self.seg_last = np.arange(self.n)

    def seg_state_tx(self, done, started):
        st = np.zeros(self.n_seg, dtype=int)
        st[self.seg_first < started] = 1
        st[self.seg_last < done] = 2
        return st


class Tx1Audio:
    """TNG1 송신 오디오 (stream_ui.TxAudio · tng5_app.Tx5Audio 와 같은 모양). 앞 lead 초 무음 뒤 블록 연속, 끝 램프 20 ms.
    새 송신 (다시 보내기 포함) 은 항상 블록 경계에서 시작 → 같은 문장이면 같은 블록 = 수신 쪽 반복 합치기 가능"""
    lead = 0.2

    def __init__(self, text, fs=48000.0, blocks=None, gain=None, fc=None):
        self.fs = float(fs)
        self.fc = float(fc) if fc is not None else FC          # 16부: 송신 중심 (선택 주파수로 바로 올림)
        t, self.bad = normalize(text)
        self.text = t
        self.blocks = blocks if blocks is not None else tng1.text_blocks(C, t)
        syms = np.concatenate([b[0] for b in self.blocks])
        x = tng1.gfsk(C, syms)
        r = int(0.02 * tng1.FS)
        w = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, r))
        x[:r] *= w
        x[-r:] *= w[::-1]
        a = tng1.to_audio(x, int(self.fs), self.fc)
        a = np.concatenate([np.zeros(int(self.lead * self.fs)), a, np.zeros(int(0.1 * self.fs))])
        self.gain = gain if gain is not None else 10 ** (-3 / 20) / max(np.max(np.abs(a)), 1e-12)   # PEP -3 dBFS (TNG5 와 같음)
        self.audio = (a * self.gain).astype(np.float32)
        self.plan = Plan1(len(self.blocks))
        self.frame_s = BLOCK_S
        self.frame_t0 = self.lead
        self.t_tone = self.t_a = self.t_data = self.lead
        self.dur = len(self.audio) / self.fs
        self.P = self.text.encode("utf-8")
        self.keep = len(self.blocks)
        self.splice_err = 0.0
        self._nchars = np.cumsum([len(b[1]) for b in self.blocks])

    def progress(self, t):
        u = (t - self.frame_t0) / self.frame_s
        done = int(np.clip(np.floor(u), 0, self.plan.n))
        started = int(np.clip(np.ceil(u), 0, self.plan.n))
        return done, started, self.plan.seg_state_tx(done, started)

    def char_segments(self):
        out, k = [], 0
        for i in range(len(self.text)):
            while k < len(self._nchars) - 1 and i >= self._nchars[k]:
                k += 1
            out.append((i, k, k))
        return out

    def chars_done(self, nblocks):
        return int(self._nchars[nblocks - 1]) if nblocks > 0 else 0

    def stopped(self, engine, t_now, margin_s=0.35):
        """중지: 지금 보내는 블록까지 보내고, 다음 블록 자리에 EOT 만 있는 짧은 블록 하나로 끝낸다. → (새 Tx1Audio, 바꿀 샘플) 또는 (None, None)"""
        done, started, _ = self.progress(t_now + margin_s)
        keep = max(started, 1)
        if keep >= self.plan.n:
            return None, None
        text = self.text[:self.chars_done(keep)]
        bl = list(self.blocks[:keep])
        eot_bits, _, _ = charset1.pack_block("", C.KI, end=True)
        bl.append((tng1.encode_block(C, eot_bits, first=False), ""))
        new = Tx1Audio(text, self.fs, blocks=bl, gain=self.gain, fc=self.fc)
        new.text = text
        new.keep = keep
        new.P = self.P
        # 이어 붙이는 자리: 블록 경계 60 ms 앞. 경계 바로 앞은 가우스 (BT 2) · 리샘플 필터가 다음 기호를 이미 반영해 옛 · 새 오디오가 달라
        # 순간 주파수가 1.7 kHz 튐 (측정). 40 ms 앞부터는 옛 · 새가 같음 (차 0, 튐 0.9 Hz = 측정 바닥) → 60 ms 로 여유
        i_sw = int(round((self.frame_t0 + keep * self.frame_s - 0.06) * self.fs))
        return new, i_sw


def timeline_tx(txa):
    """송신 타임라인 (두 줄): 블록 = 프레임 줄 · 구간 줄 같은 칸"""
    fr = 1000.0 * BLOCK_S
    frames = [("F{}".format(i + 1), i * fr, fr, "wait", "") for i in range(txa.plan.n)]
    segs = [("S{}".format(i + 1), i * fr, fr, "pending", str(i + 1)) for i in range(txa.plan.n)]
    return {"frames": frames, "segments": segs, "seg_frames": {i: [i] for i in range(txa.plan.n)}}


# ====================================================================== 실시간 수신
class Tng1Receiver:
    """입력 (링버퍼, 입력 샘플레이트) → 1 kHz 복소 (중심 1500 Hz, 절대 시각 유지) → tng1.Tng1Stream → 메시지 사건.
    제어 방법은 Tng5Receiver 와 같다 (start · stop · feed · pause · enabled · stop_req · ring)"""
    STEP_S = 0.25                  # 입력 변환 · 블록 끝 확인 주기 (무거운 처리는 반 블록마다 또는 블록 끝 직후만)
    LOST_BLOCKS = 2.5
    MARGIN_S = 0.05
    DUE_S = 0.1                    # 블록 끝 + 이만큼 지나면 반 블록을 기다리지 않고 바로 처리 (동기 미세 탐색 여유 포함)
    DUE_CANDS = 3                  # 수신 중 메시지가 없을 때: 동기 누적 상위 이만큼 후보의 블록 끝에서 바로 처리
    PEEK_S = 1.0                   # 첫 블록: 이 주기로 최근 (1 블록 + PEEK_TAIL) 만 K=1 로 접어 막 끝난 블록이 표시 문턱을 넘으면 바로 처리
    PEEK_TAIL = 1.5
    DUE_SV = 1.0                   # 단, 후보 동기 지표 ≥ 표시 문턱 (잡음 99%) × 이 값일 때만 — 잡음 후보마다 처리하던 대기 CPU 2.4배 줄이기 (09-26)

    def __init__(self, fs, on_status, on_marker, on_decoded, sync=False, cfg=None):
        self.c = cfg or C
        self.fs, self.sync = float(fs), sync
        self.decoded = {}                      # 전문가 창: 풀린 블록 시작 (1 kHz 절대) → 92 기호 (최종 판단)
        self.on_status, self.on_marker, self.on_decoded = on_status, on_marker, on_decoded
        self.on_live = None
        self.ring = RingBuffer(self.fs * 30.0)
        fr = Fraction(1000, int(round(self.fs))) if abs(self.fs - round(self.fs)) < 1e-6 else Fraction(1000 / self.fs).limit_denominator(1000)
        self.up, self.down = fr.numerator, fr.denominator
        self.m_in = int(math.ceil(self.MARGIN_S * self.fs / self.down)) * self.down     # 여유 (down 의 배수)
        self.stream = tng1.Tng1Stream(self.c)
        self.n_out = None                      # 다음 1 kHz 출력 번호 (절대, up 의 배수)
        self.paused = False
        self.enabled = True
        self.stop_req = False
        self.state = "idle"
        self.msg = None
        self.stats = {"blocks": 0, "msgs": 0, "comb": 0, "late": 0}
        self.done_end = -10 ** 12            # 끝난 메시지들의 마지막 블록 끝 (1 kHz 절대) — 늦은 결합 판단
        self.on_late = None                  # 늦은 결합 블록 (DecodeResult, 라벨) → 결과 표에만
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()      # 짧은 구간만 (입력 넣기 · 사건 처리 · 스냅샷). 동기 누적 · 복호는 잠금 밖
        self._fold_view = None             # 전문가 창 동기 누적 지도: 수신 스레드가 미리 접어 둔 결과 (화면은 그리기만)
        self.detect = None                 # TNG1 감지 표시 (_detect_from)
        self.done_keys = {}                # 끝난 메시지의 블록 내용 (정보 비트 bytes → 시각) — 늦은 결합 판정 (17부)
        self.comb_age = []
        self.det_hist = []
        self.fast_hist = []                # 빠른 감지 기록 (입력 끝 1 kHz, D, 시작 1 kHz, 주파수) — 시험용
        self.forced = 0

    # ------------------------------------------------------------ 제어
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rt-rx1")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def feed(self, x):
        self.ring.write(x)

    def pause(self, on, holdoff_s=0.3):
        """송신 중 (루프백 아님) 멈춤. 다시 켤 때 멈춘 동안은 빈 시간으로 건너뜀 (보관한 반복 합치기 블록은 유지)"""
        with self._lock:
            if on and not self.paused and self.msg is not None:
                self._finish("pause")
            self.paused = bool(on)
            if not on:
                self._jump_to(self.ring.count + int(holdoff_s * self.fs))

    def _reset_state(self):
        with self._lock:
            if self.msg is not None:
                self._finish("reset")
            self.state = "idle"
            self.detect = None

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                if not self.paused and self.enabled:
                    self.step()
            except Exception as ex:
                self.on_status("TNG1 수신 오류 · {}: {}".format(type(ex).__name__, ex), True)
            self._stop.wait(max(0.0, self.STEP_S - (time.time() - t0)))

    # ------------------------------------------------------------ 변환 (입력 → 1 kHz 복소)
    def _in_of(self, n_out):
        return n_out * self.down // self.up

    def _jump_to(self, in_abs):
        n = (int(in_abs) * self.up // self.down) // self.up * self.up
        if self.n_out is not None and n > self.n_out:
            self.stream.skip(n - self.n_out)
        self.n_out = n if self.n_out is None else max(self.n_out, n)

    def _convert(self):
        from scipy.signal import resample_poly
        cnt = self.ring.count
        if self.n_out is None:
            self._jump_to(max(0, cnt - int(2.0 * self.fs)))
        n0 = self.n_out
        n1 = ((cnt - self.m_in) * self.up // self.down) // self.up * self.up
        if n1 - n0 < self.up * 10:
            return 0
        i0, i1 = self._in_of(n0), self._in_of(n1)
        a, x = self.ring.read(i0 - self.m_in, i1 + self.m_in)
        if a != i0 - self.m_in or len(x) != i1 - i0 + 2 * self.m_in:          # 링버퍼에서 이미 사라진 구간: 건너뜀
            self._jump_to(max(a + self.m_in, cnt - int(2.0 * self.fs)))
            return 0
        ii = np.arange(a, a + len(x), dtype=np.float64)
        z = x * np.exp(-2j * math.pi * FC * (ii / self.fs)) * 2.0
        y = resample_poly(z, self.up, self.down)
        k0 = self.m_in * self.up // self.down
        self.stream.push(y[k0:k0 + (n1 - n0)])
        self.n_out = n1
        return n1 - n0

    # ------------------------------------------------------------ 한 걸음
    def step(self):
        """입력 변환 · 사건 처리만 잠금 안. 무거운 stream.step (동기 누적 · 복호) 은 잠금 밖 — 화면 스레드의 스냅샷이
        이 계산을 기다리며 멈추던 문제 (09-26 실제 앱: 약 8 s 마다 0.2~0.8 s 멈춤)"""
        with self._lock:
            self._convert()
            force = self._due()
        self._fast_detect()                                   # 표시 전용 (15부)
        evs = self.stream.step(force=force)
        if force and evs is not None:
            self.forced += 1
        fo = self.stream.fold
        if fo is not None and (self._fold_view is None or self._fold_view[0] is not fo):
            self._fold_view = (fo, self._fold_fold(fo))
        with self._lock:
            for e in evs:
                self._on_block(e)
            if self.stop_req:
                self.stop_req = False
                if self.msg is not None:
                    self._finish("manual")
            if self.msg is not None and self.stream.end > self.msg["last_end"] + self.LOST_BLOCKS * self.c.L:
                self._finish("lost")
            self.state = "rx" if self.msg is not None else "idle"

    def _due(self):
        """블록 끝 직후인가: 수신 중 메시지의 격자, 없으면 동기 누적 상위 후보 격자에서 마지막 처리 뒤에 블록이 끝났으면 True.
        → CRC 통과 블록을 블록 끝 약 0.3 s 뒤에 표시 (반 블록 주기만 쓰면 블록 끝 뒤 최대 7.4 s)"""
        st, L = self.stream, self.c.L
        if len(st.z) < L:
            return False
        grids = []
        if self.msg is not None:
            grids.append(self.msg["a0"])
        fo = st.fold
        if fo and fo.get("cands"):
            K = fo["K"]
            thr = (tng1.THR_TAB[K] if K in tng1.THR_TAB else tng1.THR_TAB[4] + 16.0 * (K - 4)) * self.c.NS / 12.0
            grids += [fo["base"] + ph * fo["hop"] for ph, _, sv in fo["cands"][:self.DUE_CANDS] if sv >= self.DUE_SV * thr]
        lim = st.end - int(self.DUE_S * tng1.FS)
        for g in grids:
            e = g + L + ((lim - g - L) // L) * L                     # 이 격자에서 lim 이전 마지막 블록 끝
            if e > st.last - int(self.DUE_S * tng1.FS) and e <= lim:
                return True
        return self.msg is None and self._peek(lim)

    def _peek(self, lim):
        """첫 블록 즉시 처리 (09-26): 동기 누적은 끝난 블록만 쓰므로 첫 블록은 끝난 뒤 반 블록 처리까지 기다렸다 (블록 끝 뒤 최대 7.4 s).
        PEEK_S 마다 최근 L + PEEK_TAIL 만 K=1 로 접어, 마지막 처리 뒤에 끝난 블록 자리의 동기 지표가 K=1 표시 문턱을 넘으면 True.
        비용: 한 번 약 3.4 ms (측정) × 초당 1번 = 코어 1개 약 0.3%"""
        st, c = self.stream, self.c
        if st.end - getattr(self, "_peek_at", -10 ** 12) < int(self.PEEK_S * tng1.FS):   # 입력 샘플 기준 주기 (동기 방식 시험도 같게)
            return False
        self._peek_at = st.end
        n = c.L + int(self.PEEK_TAIL * tng1.FS)
        if len(st.z) < n:
            return False
        z = st.z[-n:]
        zb = st.end - n
        P, hop = tng1.spectrogram(c, z)
        self._detect_from(P, hop, zb)                     # TNG1 감지 표시 (표시 전용, 같은 스펙트로그램)
        S, fr = tng1.fold_metric(c, P, hop, 1, fmax=self.stream.rp.fmax)
        if S is None:
            return False
        ends = zb + np.arange(S.shape[0]) * hop + c.L
        new = (ends > st.last - int(self.DUE_S * tng1.FS)) & (ends <= lim)
        if not new.any():
            return False
        thr1 = tng1.THR_TAB[1] * c.NS / 12.0
        self.stats["peek_max"] = max(self.stats.get("peek_max", 0.0), float(S[new].max()))
        return bool(S[new].max() >= thr1)

    DET_Z = 7.5                    # TNG1 감지 표시 문턱 (부분 동기 z, 동기 칸 DET_M 개 이상). 녹음 · TNG44 · TNG5 · 잡음 25분 최대 6.8 → 7.5 (10부 측정)
    DET_M = 3
    FAST_N = 8                     # 빠른 감지 (소리 모양): 최근 8 심볼 (1.28 s)
    FAST_D = 0.35                  # 문턱 (15부 측정, exp_tng1_detect, 0.5 s 주기): 녹음 · FT8 · TNG44 · TNG5 · 잡음 약 25분 D 최대 0.291 (말소리) → 가짜 0 (여유 20%).
                                   # TNG1 AWGN (6시행): -10 dB 6/6 중앙 1.9 s · -16 dB 0/6 (약한 신호는 부분 동기 z 7.5 그대로)

    FAST_S = 0.5                   # 빠른 감지 주기 (입력 기준)

    def _fast_detect(self):
        """빠른 감지 (15부, 표시 전용): 최근 FAST_N + 2 심볼만 역처프 스펙트로그램 → 소리 모양 D (tng1.shape_detect).
        엿보기 · 처리 판단 (_due · _peek) 과 따로 돎 — 수신기를 막 켜 창이 덜 찼거나 처리 판단이 먼저 끝나도 감지는 함.
        비용: 한 번 약 40 FFT (320 점) × 초당 2번"""
        st, c = self.stream, self.c
        if st.end - getattr(self, "_fast_at", -10 ** 12) < int(self.FAST_S * tng1.FS):
            return
        self._fast_at = st.end
        n = (self.FAST_N + 2) * c.Ns
        z = st.z
        if len(z) < n:
            return
        end = st.base + len(z)
        zb = end - n
        P, hop = tng1.spectrogram(c, z[-n:])
        fz = tng1.shape_detect(c, P, hop, n=self.FAST_N, fmax=st.rp.fmax)
        if fz is None:
            return
        a_st = zb + fz[3] * hop
        self.fast_hist.append((end, fz[0], a_st, fz[2]))
        del self.fast_hist[:-400]
        if fz[0] >= self.FAST_D and a_st >= self.done_end - c.L / 2 and self.msg is None:
            self.detect = {"z": None, "fast": fz[0], "abs_in": self._abs_in(a_st), "f": fz[2], "at": end, "m": 0}

    def _detect_from(self, P, hop, zb):
        """TNG1 감지 (표시 전용, 10부): 엿보기 스펙트로그램으로 끝나지 않은 블록까지 부분 동기 z. 복호 · 동기 · 결합 경로와 무관.
        self.detect = {z, abs_in (그 블록 시작, 입력 샘플), f, at (1 kHz 끝)} (문턱 넘을 때만 갱신), det_hist = 엿보기마다 최대 z (시험용)"""
        c = self.c
        zz, m, fr = tng1.partial_sync(c, P, hop, fmax=self.stream.rp.fmax)
        ok = m >= self.DET_M
        if not ok.any():
            return
        zz = np.where(ok[:, None], zz, -1e9)
        i, j = np.unravel_index(int(np.argmax(zz)), zz.shape)
        v = float(zz[i, j])
        a1k = zb + i * hop
        self.det_hist.append((self.stream.end, v, a1k, float(fr[j]), int(m[i])))
        del self.det_hist[:-400]
        if v >= self.DET_Z and a1k >= self.done_end - c.L / 2:     # 방금 끝난 메시지 (CRC 통과) 의 꼬리는 감지 아님
            self.detect = {"z": v, "abs_in": self._abs_in(a1k), "f": float(fr[j]), "at": self.stream.end, "m": int(m[i])}

    def detect_snapshot(self):
        """화면 스레드: 최근 TNG1 감지 (수신 중 메시지가 없을 때만 의미). 잠금 없음 (dict 한 번 읽기)"""
        return self.detect

    def _abs_in(self, a1k):
        return int(round(a1k * self.down / self.up))

    def _same(self, e):
        m = self.msg
        if m is None:
            return False
        ph = (e["abs"] - m["a0"]) / self.c.L
        dt = (ph - round(ph)) * self.c.L                          # 격자 어긋남 (샘플)
        # 처프: 격자가 dt 어긋나 풀린 블록은 주파수도 기울기 × dt 만큼 옮겨짐 (1/4 기호 → 12.5 Hz). 빼고 비교 (
        # 블록 끝 직후 처리에서 1/4 기호 어긋난 격자로 풀리는 일이 잦아져 메시지가 블록마다 나뉨)
        return abs(e["f"] - m["f"] - tng1.ridge(self.c, dt)) < 8.0 and abs(dt) < self.c.Ns and e["abs"] < m["last_end"] + 3 * self.c.L

    def _ev(self, ev):
        ev.setdefault("mode", NAME)
        if self.on_live is not None:
            self.on_live(ev)

    def _late(self, e):
        """이미 지나간 시각의 결합 블록 (7부 결정, 11부: 앞 메시지 유무와 관계없이) = 같은 내용의 더 새 송신이 이미 있음 (Tng1Stream 이 판단,
        e['late']) → 새 메시지로 열지 않고 결과 표에 '늦은 결합' 만. 수신 중 메시지의 같은 격자 블록은 제외.
        (시각 기준은 못 씀: 정상 결합도 블록 끝 뒤 0.36~3.43 블록에 나오고 옛 사본 사례는 약 2.3 블록, 11부 측정)"""
        if e["n_comb"] <= 1:
            return False
        if self.msg is not None and self._same(e):
            return False
        # 17부 (7부 결정): 결합 블록 내용이 이미 끝난 (표시된) 메시지의 블록 내용과 같으면 늦은 결합 (자리 무관)
        key = np.asarray(e["info"], np.int8).tobytes()
        if key in self.done_keys:
            return True
        return bool(e.get("late"))

    def _on_block(self, e):
        if e["n_comb"] > 1:                                       # 시험용: 결합 블록이 블록 끝 뒤 몇 블록 만에 나왔나 (최근 100)
            self.comb_age.append(round((self.stream.end - e["abs"] - self.c.L) / self.c.L, 2))
            del self.comb_age[:-100]
        if self._late(e):
            # 09-26 (승인): 옛 송신 블록이 뒤늦게 보관 · 결합되면 따로 된 새 메시지로 열던 것 → 결과 표에 '늦은 결합' 으로만
            self.stats["late"] += 1
            if self.on_late is not None:
                from engine import DecodeResult
                txt = e["text"]
                r = DecodeResult(ok=False, text="", snr_db=float("nan"), df_hz=float(e["f"]), reason="늦은 결합",
                                 mode="TNG1 스트리밍 · {}회 결합".format(e["n_comb"]),
                                 info={"format": NAME, "late": True, "partial": txt, "comb": e["n_comb"], "segments_ok": 1,
                                       "abs_in": self._abs_in(e["abs"])})
                self.on_late(r, "실시간 · TNG1 · 늦은 결합")
            return
        if not self._same(e):
            if self.msg is not None:
                self._finish("next")
            self.msg = {"a0": e["abs"], "f": e["f"], "blocks": {}, "last_end": e["abs"] + self.c.L, "t0": time.time(), "es": []}
            self.stats["msgs"] += 1
            self._ev({"type": "start", "abs": self._abs_in(e["abs"]), "fs": self.fs, "df": float(e["f"]),
                      "block_in": self.c.L * self.down / self.up})
            if self.c.first_flag and not e.get("first"):              # 첫 블록 표시 없음 = 앞부분 미수신 (□)
                self._ev({"type": "segments", "segs": [{"index": -1, "ok": False, "text": "□", "n_comb": 0, "head": True}]})
        m = self.msg
        k = int(round((e["abs"] - m["a0"]) / self.c.L))
        m["blocks"][k] = e
        self.decoded[int(round(e["abs"]))] = tng1.encode_block(self.c, list(np.asarray(e["info"], int)), first=bool(e.get("first")))
        if len(self.decoded) > 400:
            for key in sorted(self.decoded)[:100]:
                del self.decoded[key]
        m["last_end"] = max(m["last_end"], e["abs"] + self.c.L)
        if e.get("es"):
            m["es"].append(e["es"])
        self.stats["blocks"] += 1
        if e["n_comb"] > 1:
            self.stats["comb"] += 1
        self._ev({"type": "segments", "segs": [{"index": k, "ok": True, "text": e["text"], "n_comb": e["n_comb"],
                                                 "abs": self._abs_in(e["abs"])}]})
        if e["eot"]:
            self._finish("eot")

    def _finish(self, why):
        m, self.msg = self.msg, None
        if m is None:
            return
        self.done_end = max(self.done_end, m["last_end"])
        for e_ in m["blocks"].values():                           # 17부: 끝난 메시지 블록 내용 (늦은 결합 판정, 최근 200개)
            self.done_keys[np.asarray(e_["info"], np.int8).tobytes()] = time.time()
        while len(self.done_keys) > 200:
            del self.done_keys[next(iter(self.done_keys))]
        from engine import DecodeResult
        ks = sorted(m["blocks"])
        k0 = min(0, ks[0])
        segs = []
        for k in range(k0, ks[-1] + 1):
            e = m["blocks"].get(k)
            segs.append({"index": k - k0, "ok": e is not None, "text": e["text"] if e else "□",
                         "n_comb": e["n_comb"] if e else 0})
        head_missing = self.c.first_flag and not m["blocks"][ks[0]].get("first")
        if head_missing:                                          # 앞 블록 (들) 을 못 받음: 앞에 □ 하나
            segs = [{"index": 0, "ok": False, "text": "□", "n_comb": 0, "head": True}] + \
                   [dict(x, index=x["index"] + 1) for x in segs]
        n_ok = sum(1 for s in segs if s["ok"])
        eot = any(m["blocks"][k]["eot"] for k in ks)
        text = "".join(s["text"] for s in segs)
        ncomb = max(s["n_comb"] for s in segs)
        es = float(np.median(m["es"])) if m["es"] else None
        snr = 10 * math.log10(es) - 10 * math.log10(2500 * self.c.T) if es else float("nan")   # 심볼 Es/N0 → 2500 Hz SNR (PEP)
        ok = eot and n_ok == len(segs) and not head_missing
        why_txt = {"eot": "", "lost": "신호 소실", "manual": "수동 중지", "next": "다음 신호", "pause": "송신 시작", "reset": "초기화"}[why]
        r = DecodeResult(ok=ok, text=text if ok else "", snr_db=snr, df_hz=float(m["f"]),
                         reason="" if ok else ("앞부분 미수신" if head_missing else
                                               ("끝 표식 (EOT) 없음 · " + why_txt if n_ok == len(segs) else "블록 일부 실패")),
                         mode="TNG1 스트리밍" + (" · {}회 결합".format(ncomb) if ncomb > 1 else ""),
                         viz={"msg": {"a": {"start": self._abs_in(m["a0"])}}},
                         info={"format": NAME, "segments": segs, "segments_ok": n_ok, "segments_used": len(segs),
                               "partial": text, "comb": ncomb, "blocks_abs": [self._abs_in(m["a0"] + (k + k0) * self.c.L) for k in range(len(segs))],
                               "block_in": self.c.L * self.down / self.up, "end": why, "head_missing": head_missing,
                               "blocks": [{"k": k - k0 + (1 if head_missing else 0), "abs": m["blocks"][k]["abs"],
                                           "abs_in": self._abs_in(m["blocks"][k]["abs"]), "f": float(m["blocks"][k]["f"]),
                                           "info": np.asarray(m["blocks"][k]["info"], int), "E": m["blocks"][k].get("E"),
                                           "n_comb": m["blocks"][k]["n_comb"], "first": bool(m["blocks"][k].get("first")),
                                           "text": m["blocks"][k]["text"], "t_s": (m["blocks"][k]["abs"] - m["a0"]) / 1000.0}
                                          for k in ks]})
        label = "실시간 · TNG1" + (" · 신호 소실" if why == "lost" else "") + (" · 수동 중지" if why == "manual" else "") + \
                (" · {}회 결합".format(ncomb) if ncomb > 1 else "")
        self._ev({"type": "lost" if why == "lost" else ("manual_stop" if why == "manual" else "end")})
        self.on_decoded(r, label)


    # ------------------------------------------------------------ 전문가 창 (UI 스레드에서 부름, 잠금 짧게)
    def grid_snapshot(self, span_s=32.0):
        """16음 격자 · 확신도용: 지금 격자 (수신 중 메시지, 없으면 동기 누적 1위 후보) 에 맞춘 최근 기호 에너지"""
        c = self.c
        with self._lock:
            st = self.stream
            if len(st.z) < c.L // 4:
                return None
            if self.msg is not None:
                a0, f, note = self.msg["a0"], self.msg["f"], "수신 중 메시지 격자"
            elif st.fold and st.fold["cands"]:
                ph, f, sv = st.fold["cands"][0]
                a0, note = st.fold["base"] + ph * st.fold["hop"], "동기 누적 1위 후보 (미확정)"
            else:
                return None
            end, base, z = st.end, st.base, st.z
            n = int(span_s / c.T)
            s1 = (end - a0) // c.Ns - 1
            s0 = max(s1 - n + 1, -((a0 - base) // c.Ns))
            if s1 < s0:
                return None
            ss = np.arange(s0, s1 + 1)
            starts = a0 + ss * c.Ns - base
            ok = (starts >= 0) & (starts + c.Ns <= len(z))
            ss, starts = ss[ok], starts[ok]
            seg = z[starts[:, None] + np.arange(c.Ns)]
            tt = (base + starts[:, None] + np.arange(c.Ns)) / tng1.FS
            zz = seg * np.exp(-2j * math.pi * f * tt) * tng1._dechirp(c)[None]
            E = np.abs(np.fft.fft(zz, axis=1)[:, tng1._bins(c) % c.Ns]) ** 2
            dec = dict(self.decoded)
        E = E / max(np.median(E) / math.log(2), 1e-30)
        pos, tone, _, _, _ = tng1.lay(c)
        blk = np.floor_divide(ss, c.nsym)
        j = ss - blk * c.nsym
        is_sync = np.isin(j, pos)
        tsync = np.zeros(len(ss), int)
        tmap = dict(zip(pos.tolist(), tone.tolist()))
        tsync[is_sync] = [tmap[x] for x in j[is_sync]]
        final = np.full(len(ss), -1)
        for b in np.unique(blk):
            key = int(a0 + b * c.L)
            hit = next((dec[k] for k in (key, key - 1, key + 1) if k in dec), None)
            if hit is None:
                hit = next((v for k, v in dec.items() if abs(k - key) < c.Ns // 2), None)
            if hit is not None:
                m = blk == b
                final[m] = hit[j[m]]
        final[is_sync] = tsync[is_sync]
        t = (a0 + ss * c.Ns - end) / tng1.FS
        blocks = [((a0 + b * c.L - end) / tng1.FS, "B{}".format(int(b) + 1 if self.msg is not None else int(b)),
                   self._abs_in(a0 + b * c.L)) for b in np.unique(blk)]
        return {"t": t, "T": c.T, "E": E, "sync": is_sync, "tone_sync": tsync, "final": final, "blocks": blocks,
                "note": "{} · 주파수 {:+.1f} Hz · 기호 {}개 (160 ms) · 테두리 흰색 = 최종 판단, 청록 = 동기, × = 에너지 1등과 다름".format(
                    note, f, len(ss))}

    def live_blocks(self):
        """수신 중 메시지의 CRC 통과 블록 (해부용)"""
        with self._lock:
            m = self.msg
            if m is None:
                return []
            return [{"k": k, "abs": e["abs"], "abs_in": self._abs_in(e["abs"]), "f": float(e["f"]),
                     "info": np.asarray(e["info"], int), "E": e.get("E"), "n_comb": e["n_comb"],
                     "first": bool(e.get("first")), "text": e["text"], "t_s": (e["abs"] - m["a0"]) / 1000.0}
                    for k, e in sorted(m["blocks"].items())]

    def _fold_fold(self, fo):
        """수신 스레드에서: 동기 누적 지도 (블록 주기 위상으로 접은 최대 · 문턱 · 후보) 를 미리 계산"""
        K = fo["K"]
        thr = (tng1.THR_TAB[K] if K in tng1.THR_TAB else tng1.THR_TAB[4] + 16.0 * (K - 4)) * self.c.NS / 12.0
        hop_s = fo["hop"] / tng1.FS
        Lh = self.c.L // fo["hop"]
        S = fo["S"]
        n = S.shape[0] // Lh * Lh
        ph = S[:n].reshape(-1, Lh, S.shape[1]).max(0) if n else np.full((Lh, S.shape[1]), -1.0)
        if n < S.shape[0]:                                     # 남은 조각 (블록 주기 위상으로 접은 최대, 후보 고르기와 같음)
            r = S[n:]
            ph[:len(r)] = np.maximum(ph[:len(r)], r)
        return {"S": ph, "fr": fo["fr"], "hop_s": hop_s, "K": K, "thr": thr,
                "cands": [(p * hop_s, f) for p, f, _ in fo["cands"][:8]]}

    def fold_snapshot(self):
        """화면 스레드: 미리 계산한 지도만 돌려줌 (잠금 · 계산 없음)"""
        fv = self._fold_view
        return fv[1] if fv is not None else None

    def sync_snapshot(self):
        """가장 최근 풀린 블록 (없으면 수신 중 격자 마지막 완전 블록) 의 동기 12음 점수"""
        c = self.c
        with self._lock:
            m = self.msg
            if m is not None and m["blocks"]:
                k = max(m["blocks"])
                e = m["blocks"][k]
                E = e.get("E")
                lab = "B{} (CRC 통과)".format(k + 1)
            else:
                return None
        if E is None:
            return None
        return {"score": sync_scores(np.asarray(E), c), "label": lab}


def sync_scores(E, c=None):
    """블록 에너지 (92, 16) → 동기 12음 점수 [dB] (동기 칸 ÷ 같은 기호 16음 평균)"""
    pos, tone, _, _, _ = tng1.lay(c or C)
    return 10 * np.log10(np.maximum(E[pos, tone], 1e-9) / np.maximum(E[pos].mean(1), 1e-9))


class Multi3:
    """기존 수신 묶음 (MultiRx: TNG44 · TNG5) + TNG1 수신기. 앱은 self._rx 하나로 부른다 (속성은 기존 묶음으로).
    단일 모드 운용 (09-27): active 는 모드 하나 — 나머지 수신기는 상태를 비우고 검출을 버림"""

    def __init__(self, multi, rx1, active):
        self.__dict__["multi"] = multi
        self.__dict__["rx1"] = rx1
        self.set_active(active)

    def __getattr__(self, k):
        return getattr(self.multi, k)

    def __setattr__(self, k, v):
        if k == "stop_req":
            self.rx1.stop_req = v
        setattr(self.multi, k, v)

    def set_active(self, active):
        self.multi.set_active([a for a in active if a != NAME])
        self.rx1.enabled = NAME in set(active)
        if not self.rx1.enabled:
            self.rx1._reset_state()

    def start(self):
        self.multi.start()
        if self.rx1.enabled:                          # 단일 모드: TNG1 일 때만 스레드
            self.rx1.start()

    def stop(self):
        self.multi.stop()
        self.rx1.stop()

    def feed(self, x):
        self.multi.feed(x)
        self.rx1.feed(x)

    def step(self):
        self.multi.step()
        if self.rx1.enabled and not self.rx1.paused:
            self.rx1.step()

    def pause(self, on, holdoff_s=0.3):
        self.multi.pause(on, holdoff_s)
        self.rx1.pause(on, holdoff_s)

    @property
    def paused(self):
        return self.multi.paused


# ====================================================================== 분석 · 표시
def analyze(res, job, last_tx_text=None):
    """수신 결과 → 패킷 자료 (타임라인: 위 줄 블록 (결합 수), 아래 줄 블록 CRC16)"""
    info = res.info or {}
    segs = info.get("segments") or []
    fr = 1000.0 * BLOCK_S
    frames, rows = [], []
    for s in segs:
        k = s["index"]
        st = "ok" if s["ok"] else "fail"
        lab = "{}×".format(s["n_comb"]) if s.get("n_comb", 1) > 1 else ""
        frames.append(("F{}".format(k + 1), k * fr, fr, "fixed" if lab else st, lab))
        rows.append(("S{}".format(k + 1), k * fr, fr, st, str(k + 1)))
    out = {"kind": "rx", "id": job.get("id"), "ok": res.ok, "text": res.text, "label": job.get("label", ""),
           "df": res.df_hz, "snr": res.snr_db, "utc": job.get("utc"), "mode": res.mode, "reason": res.reason,
           "format": NAME, "segments_ok": info.get("segments_ok"), "segments_used": info.get("segments_used"),
           "partial": info.get("partial"), "margin": None, "rho_a": None,
           "timeline": {"frames": frames, "segments": rows, "seg_frames": {i: [i] for i in range(len(segs))}},
           "tng1_blocks": info.get("blocks") or []}
    return out


def blocks_overlay(o, center=1500.0):
    """워터폴 구간 표시: TNG1 블록 (14.7 s) 마다 한 칸. 상태: CRC 통과 (결합 수 표시) · 실패 □ · 대기"""
    a, L = o["a"], o["block_in"]
    fc = center + float(o.get("df") or 0.0)
    st = o.get("states") or {}
    info = o.get("binfo") or {}
    if not st:
        return []
    k1 = max(st)
    if not o.get("ended") and o.get("now") is not None:
        k1 = max(k1, int((o["now"] - a) // L))
    out = []
    for k in range(min(0, min(st)), k1 + 1):
        s_ = st.get(k, "fail" if o.get("ended") else "pending")
        n = info.get(k, 1)
        b1 = a + (k + 1) * L
        if s_ == "pending" and o.get("now") is not None:
            b1 = max(a + k * L + 1, min(b1, o["now"]))              # 수신 중 블록: 받은 만큼만 (10부)
        out.append({"abs0": a + k * L, "abs1": b1, "kind": "seg", "state": s_,
                    "label": "B{}{}".format(k + 1, " ×{}".format(n) if n > 1 else ""),
                    "tip": "TNG1 블록 {} (14.7 s · 동기 12 + 데이터 80 심볼)\n{}{}".format(
                        k + 1, {"ok": "CRC16 통과", "fail": "CRC16 실패 (□)", "pending": "수신 중"}[s_],
                        " · {}회 결합".format(n) if n > 1 else ""),
                    "f0": fc - 80, "f1": fc + 80})
    return out


def decode_file(x, fs):
    """WAV 열기: 파일 전체를 실시간 수신기로 (동기 방식) 흘려 결과 목록 [(DecodeResult, 라벨)].
    1.1 (wire rev 2) 로 먼저, 결과가 없거나 모두 '앞부분 미수신' 이면 1.0 (rev 1, 첫 블록 표시 없음) 으로 다시"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    out = []
    for cfg, lab in ((tng1.CFGS["TNG1"], "TNG1"), (tng1.CFGS["TNG1r1"], "TNG1 rev 1")):
        res = []
        rx = Tng1Receiver(fs, lambda m, e: None, lambda m: None, lambda r, l: res.append((r, l)), sync=True, cfg=cfg)
        rx.on_live = lambda ev: None
        n = int(fs)
        xx = np.concatenate([x, np.zeros(int((Tng1Receiver.LOST_BLOCKS + 1.5) * BLOCK_S * fs))])
        for i in range(0, len(xx), n):
            rx.feed(xx[i:i + n])
            rx.step()
        if rx.msg is not None:
            with rx._lock:
                rx._finish("lost")
        res = [(r, "파일 · " + lab + (" · {}회 결합".format((r.info or {}).get("comb")) if (r.info or {}).get("comb", 1) > 1 else ""))
               for r, l in res]
        if res and not all((r.info or {}).get("head_missing") for r, _ in res):
            return res
        out = out or res
    return out
