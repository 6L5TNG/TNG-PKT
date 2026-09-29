"""
실시간 수신 — 입력 장치 오디오를 계속 받아 코스타스 A/B 로 메시지를 잡아 복조한다.

    입력 콜백 ─► 링버퍼 (최대 max_message_s + 여유)
                    │ 0.25초마다
                    ▼
    수신 스레드: 최근 1.5초를 기저대역으로 바꿔 코스타스 A/B 탐색
        A 검출  → "신호 감지 (주파수 오차 ±xx Hz)" 알림, 워터폴 표시
        B 검출  → 무음을 기다리지 않고 A~B 구간을 즉시 복조 (별도 스레드)
        B 를 놓친 경우 → 다음 A 가 오거나 최대 길이를 넘기면 A 위치부터 복조 시도
        A 검출 뒤 (7단계 스트리밍 형식) → 도착하는 프레임을 stream_ui.LiveStream 으로 흘려
                  32바이트 구간이 확정되는 대로 on_live 로 알린다 (최종 판정은 B 뒤 전체 복조)
        끝내는 순서: 코스타스 B > 길이 필드 (v2, 받을 프레임을 다 받으면 B 를 기다리지 않고 복조)
                  > 신호 소실 (대역 에너지 · 구간 실패 · |LLR| 모두 잡음 수준이면 즉시 복조)

변조/복조 로직은 새로 만들지 않는다. 검출은 preamble.CostasDetector, 복조는 engine.decode
(= modem5 → modem4) 를 그대로 쓴다. 송신 중에는 수신을 멈춘다 (자기 신호를 받지 않도록).
"""

import threading
import time

import numpy as np
import scipy.signal as sg

from config import PreambleConfig
from engine import resample, MODEM_FS
from modem import audio_to_baseband
from preamble import CostasDetector


class RingBuffer:
    """절대 샘플 번호로 읽고 쓰는 링버퍼 (콜백 스레드가 쓰고 수신 스레드가 읽는다)."""

    def __init__(self, capacity):
        self.buf = np.zeros(int(capacity), dtype=np.float32)
        self.count = 0                       # 지금까지 쓴 총 샘플 수 (= 다음 쓸 절대 번호)
        self._lock = threading.Lock()

    def write(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        n, cap = len(x), len(self.buf)
        if n >= cap:
            x, n = x[-cap:], cap
        with self._lock:
            i = self.count % cap
            k = min(n, cap - i)
            self.buf[i:i + k] = x[:k]
            if k < n:
                self.buf[:n - k] = x[k:]
            self.count += n

    def read(self, a, b):
        """절대 구간 [a, b) — 이미 덮어써진 부분은 잘라낸다. 반환: (시작 번호, 샘플)"""
        with self._lock:
            cap = len(self.buf)
            a = max(int(a), self.count - cap, 0)
            b = min(int(b), self.count)
            if b <= a:
                return a, np.zeros(0, dtype=np.float32)
            idx = np.arange(a, b) % cap
            return a, self.buf[idx].copy()


class RealtimeReceiver:
    """
    콜백 3개로 UI 와 이어진다 (모두 수신 스레드에서 불린다 — UI 쪽은 Qt 시그널로 받을 것):
        on_status(msg, is_error)
        on_marker(dict)   : {'kind','abs','df','rho'}  — 워터폴 표시용
        on_decoded(result, label)
    """

    SCAN_S = 1.5          # 한 번에 탐색하는 길이
    STEP_S = 0.25         # 탐색 간격
    EDGE_S = 0.15         # 탐색 구간 양 끝(변환 필터 가장자리)의 검출은 버린다
    PRE_S = 0.6           # 복조 구간을 A 앞으로 이만큼 넉넉히
    POST_S = 0.2          # B 끝 뒤로 이만큼 기다렸다 복조 (0.4 → 0.2: 원격 세션 3번, 복조 창 여유는 필터 꼬리 약 0.1 s)

    def __init__(self, engine, fs, on_status, on_marker, on_decoded, pc=None, sync=False):
        # sync=True : 시험용. 스레드를 띄우지 않고, 부르는 쪽이 STEP_S 분량 샘플마다 step() 을 부른다.
        #             복조도 그 자리에서 바로 한다. 수신 로직은 똑같고 '시계'만 샘플 수로 바뀐다.
        self.sync = sync
        self.engine = engine
        self.cfg = engine.cfg
        self.pc = pc or PreambleConfig()
        self.fs = float(fs)
        self.det = CostasDetector(self.pc, self.cfg.fs_base)    # 복조용과 따로 (스레드 안전)
        self.ring = RingBuffer(self.fs * (self.pc.max_message_s + 10.0))
        self.on_status, self.on_marker, self.on_decoded = on_status, on_marker, on_decoded
        self.n_costas = int(round(self.pc.symbol_ms * 1e-3 * self.pc.n_tones * self.fs))
        self._stop = threading.Event()
        self._thread = None
        self.enabled = True               # 단일 모드 운용: TNG44 가 아닌 모드면 False (탐색 안 함, 스레드도 안 띄움)
        self.ignore_before = 0            # 이 절대 번호 이전의 검출은 무시 (송신 일시정지용)
        self.paused = False
        self._reset_state()
        self.stats = {"A": 0, "B": 0, "decoded": 0, "ok": 0, "scans": 0}
        self.on_heartbeat = None          # 선택: 1분마다 동작 기록 (탐색 횟수, 최대 rho², 입력 레벨)
        self.B_TOL_S = 0.5                # 길이 필드를 알 때 코스타스 B 인정 범위 (예상 위치 ±)
        self.B_LLR_S = 0.6                # 길이 필드를 모를 때: B 뒤 이만큼의 평균 |LLR| 로 신호 소실 확인 뒤 B 인정
        self.B_LLR_T = 2.85               # = LiveStream.LOSS_LLR (잡음 0.6 s 평균 99.9% 분위 1.72 · 최대 1.86, 측정)
        self.B_STRONG = 0.35              # 이 ρ² 이상 B 는 B 뒤 |LLR| 확인 없이 바로 인정 (음성 가짜 B 최대 0.236 측정, results/remote/tng44_fa.log)
        self.stop_req = False             # 수동 중지 요청 (화면 스레드 → 수신 스레드)
        self.on_live = None               # 선택: 실시간 구간 사건 {'type': 'start'|'segments'|'final_segments'|'lost'|'b'|'b_only'|'format', ...}
        self.live = None
        self.expect_b_until = 0           # 길이 필드로 먼저 끝낸 메시지의 B 는 이 번호까지 조용히 무시
        self._hb = {"t": time.time(), "scans": 0, "max_rho": 0.0, "rms": 0.0}

    # ---------------------------------------------------------------- 제어
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rt-rx")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def feed(self, x):
        """입력 콜백에서 부른다 — 빠르게 링버퍼에 쓰기만 한다."""
        self.ring.write(x)

    def pause(self, on, holdoff_s=0.3):
        """송신 중 수신 일시정지. 재개할 때 송신 꼬리를 피하려고 holdoff 만큼 더 건너뛴다."""
        self.paused = bool(on)
        self._reset_state()
        self.ignore_before = self.ring.count + (int(holdoff_s * self.fs) if not on else 0)

    def _reset_state(self):
        self.state = "idle"
        self.live = None
        self.a = None                     # 진행 중인 메시지의 A 검출 {'abs','df','rho'}
        self.pending = None               # 예약된 복조 (구간 끝 번호, 시작 번호, 설명)
        self.seen = {"A": [], "B": []}    # 중복 검출 제거용 최근 절대 위치
        self.b_hold = None                # 신호 소실 확인 대기 중인 B (pos, d)
        self._suspect = []                # 음성 오검출 의심 (B 뒤 오디오 기다림)

    # ---------------------------------------------------------------- 수신 스레드
    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                if not self.paused and self.enabled:
                    self._step()
            except Exception as ex:          # 수신 스레드가 죽지 않도록 — 상태줄에만 알린다
                self.on_status("실시간 수신 오류 — {}: {}".format(type(ex).__name__, ex), True)
            self._stop.wait(max(0.0, self.STEP_S - (time.time() - t0)))

    BB_H = 240                 # 30부: resample_poly (1, 24) 필터 반길이 (481탭) — 입력 샘플

    def _mix(self, x, n0):
        """절대 입력 번호 n0 부터의 오디오를 중심 주파수로 하향 변환 (audio_to_baseband 와 같은 식, 위상은 절대 시각 기준)"""
        fs, fc = float(self.fs), float(self.cfg.audio_center_hz)
        n = np.arange(n0, n0 + len(x), dtype=np.float64)
        return 2.0 * np.asarray(x, np.float64) * np.exp(-2j * np.pi * np.mod(fc * n, fs) / fs)

    def _bb_window(self, raw, a0):
        """
        30부: 탐색 창 [a0, a0+len(raw)) 의 기저대역 = 예전 _to_baseband(raw) 와 같은 값 (반올림 차이만).
        가운데 = 새로 들어온 부분만 변환해 이어 붙인 기저대역 줄 (필터가 창 안에 다 들어가는 출력은 창과 무관하게 같다),
        양 끝 (필터가 창 밖 0 을 보는 출력, 각 약 10개) 은 예전처럼 창 기준으로 새로 계산. a0 는 down 의 배수 (탐색 격자 정렬)
        """
        down = int(round(self.fs / self.cfg.fs_base))
        H = self.BB_H
        e = int(np.ceil(H / down))                          # 가장자리 출력 수 (앞뒤)
        n_out = -(-len(raw) // down)
        fc = self.cfg.audio_center_hz
        st = self.__dict__.get("_bbs")
        if st is None or st["fc"] != fc or st["down"] != down:
            st = self._bbs = {"fc": fc, "down": down, "lo": None, "buf": np.zeros(0, np.complex128)}
        key0, now = a0 // down, a0 + len(raw)
        # 1) 줄 늘리기: 필터가 다 들어가는 출력 j (j*down + H < now) 까지
        hi_new = (now - H - 1) // down + 1                  # 계산 가능한 출력 끝 (배타)
        if st["lo"] is None or st["lo"] > key0 or st["lo"] + len(st["buf"]) < key0:
            st["lo"], st["buf"] = key0 + e, np.zeros(0, np.complex128)
        cur_hi = st["lo"] + len(st["buf"])
        if hi_new > cur_hi:
            p0 = cur_hi * down - H - down * (e + 1)         # 계산 청크 시작 (down 의 배수, 앞 여유)
            p0 = max(p0 - (p0 % down), a0)
            s0_, x_ = self.ring.read(p0, now)
            if s0_ == p0 and len(x_):
                y = sg.resample_poly(self._mix(x_, p0), 1, down)
                j = np.arange(len(y))
                ok = (j * down - H >= 0) & (j * down + H < len(x_))
                jj = j[ok] + p0 // down
                take = (jj >= cur_hi) & (jj < hi_new)
                if take.any():
                    st["buf"] = np.concatenate([st["buf"], y[ok][take]])
        # 오래된 앞부분 버림 (창 앞 1 s 여유)
        cut = key0 - int(self.cfg.fs_base) - st["lo"]
        if cut > 0:
            st["buf"], st["lo"] = st["buf"][cut:], st["lo"] + cut
        lo, buf = st["lo"], st["buf"]
        # 2) 창 조립: 가운데 = 줄 (창 기준 위상으로), 양 끝 = 창 기준 새 계산
        a, b = key0 + e, key0 + n_out - e                   # 줄에서 가져올 [a, b)
        if not (lo <= a and b <= lo + len(buf) and b > a):
            return self._to_baseband(raw)                  # 줄이 아직 없음 → 예전 방식
        ph = np.exp(2j * np.pi * np.mod(fc * float(a0), float(self.fs)) / float(self.fs))   # 절대 위상 → 창 기준 위상
        mid = buf[a - lo:b - lo] * ph
        L_ = min(len(raw), (e + 1) * down + H + down)
        left = sg.resample_poly(self._mix(raw[:L_], 0), 1, down)[:e]
        r0 = (n_out - e) * down - H - down
        r0 = max(r0 - (r0 % down), 0)
        right = sg.resample_poly(self._mix(raw[r0:], r0), 1, down)[(n_out - e) - r0 // down:]
        out = np.concatenate([left, mid, right])
        return out if len(out) == n_out else self._to_baseband(raw)

    def _to_baseband(self, x):
        fs = self.fs
        if fs % self.cfg.fs_base != 0:
            x = resample(x, fs, MODEM_FS)
            fs = MODEM_FS
        return audio_to_baseband(np.asarray(x, dtype=np.float64), fs, self.cfg)

    def step(self):
        """동기 모드에서 부르는 한 번의 탐색 (수신 스레드의 한 주기와 같다)"""
        if not self.paused and self.enabled:
            self._step()

    def _step(self):
        now = self.ring.count
        fs = self.fs

        # 1) 예약된 복조가 때가 됐으면 실행
        if self.pending is not None and now >= self.pending[0]:
            end, start, why = self.pending
            self.pending = None
            self._decode(start, end, why)

        # 1-1) 길이 모르는 메시지의 B: B 뒤 0.6 s |LLR| 이 잡음 수준이면 인정, 아니면 가짜 B (신호 계속)
        if self.b_hold is not None:
            self._check_b_hold()

        # 1-2) 실시간 구간 복조 (A 검출 뒤, B 전)
        lv = self.live
        if lv is not None and not lv.done and self.state == "rx":
            try:
                fmt0 = lv.format
                segs = lv.step(self.ring)
                if lv.format is not None and fmt0 is None:
                    self._live_event({"type": "format", "format": lv.format})
                if lv.done and lv.end_reason == "length":
                    # 길이 필드: 받을 프레임을 다 받음 → 가드 끝 + 0.35초에 B 없이 복조 (B 가 먼저 오면 B 경로)
                    self._live_event({"type": "final_segments", "segs": segs, "fed": lv.fed, "length": lv.length})
                    end = int(lv.end_abs() + 0.35 * self.fs)
                    self.pending = (end, self.a["abs"] - int(self.PRE_S * self.fs), "길이 필드")
                    self.state = "decoding"
                    self.expect_b_until = end + int(2.0 * self.fs)
                    self.on_status("길이 필드로 메시지 끝 확인 — 복조 중…", False)
                elif lv.done and lv.end_reason == "fake":
                    # 가짜 시작 (A 오검출): 복조하지 않고 폐기 → 다시 A 대기
                    self._live_event({"type": "fake", "fed": lv.fed, "guard_q": lv.guard_q,
                                      "llr": float(np.mean(lv.llr_lv)) if lv.llr_lv else None,
                                      "t_s": (self.ring.count - self.a["abs"]) / self.fs, "df": self.a["df"],
                                      "rho": self.a["rho"]})
                    self.on_status("오검출 폐기 — 코스타스 A 뒤 데이터 없음 (가드 불명확 · 첫 구간 실패)", True)
                    self._reset_state()
                elif lv.done and lv.end_reason == "loss":
                    if segs:
                        self._live_event({"type": "segments", "segs": segs, "fed": lv.fed})
                    self._live_event({"type": "lost", "fed": lv.fed})
                    self.on_status("신호 소실 — 받은 부분까지 복조", True)
                    self._decode(self.a["abs"] - int(self.PRE_S * self.fs), self.ring.count, "신호 소실")
                    self._reset_state()
                elif segs:
                    self._live_event({"type": "segments", "segs": segs, "fed": lv.fed})
            except Exception as ex:
                lv.done = True
                self.on_status("실시간 구간 복조 오류 — {}: {}".format(type(ex).__name__, ex), True)

        # 1-3) 수동 중지: 받은 데까지 복조하고 다시 A 대기
        if self.stop_req:
            self.stop_req = False
            if self.state == "rx" and self.a is not None:
                self._live_event({"type": "manual_stop"})
                self._decode(self.a["abs"] - int(self.PRE_S * self.fs), self.ring.count, "수동 중지")
                self._reset_state()

        # 2) 최근 구간에서 코스타스 탐색
        a0 = max(now - int(self.SCAN_S * fs), self.ignore_before)
        k = fs / self.cfg.fs_base                 # 기저대역 1샘플 = 입력 k 샘플
        g_ = int(round(k * self.pc.coarse_step))
        key0 = None
        if abs(k - round(k)) < 1e-9 and g_ > 0:   # 29부: 창 시작을 탐색 격자 (coarse_step 기저대역 샘플) 에 맞춤 (최대 2 ms 앞) → 앞 창 계산 재사용
            a0 = max((a0 // g_) * g_, 0)
            key0 = a0 // int(round(k))
        if now - a0 < int(0.6 * fs):
            return
        s0, raw = self.ring.read(a0, now)
        if np.max(np.abs(raw)) < 1e-7:
            return
        if s0 != a0:
            key0 = None                           # 링버퍼 앞이 잘림 → 이번엔 재사용 안 함
        bb = self._bb_window(raw, a0) if key0 is not None else self._to_baseband(raw)
        if key0 is not None and getattr(self.det, "_cache_fc", None) != self.cfg.audio_center_hz:
            self.det._cache.clear()               # 30부: 중심 주파수가 바뀌면 탐색 재사용 값도 버림
            self.det._zcache.clear()
            self.det._cache_fc = self.cfg.audio_center_hz
        edge = int(self.EDGE_S * self.cfg.fs_base)
        dets = self.det.detect(bb, key0=key0)
        self._heartbeat(raw)
        for d in dets:
            if d["start"] < edge or d["start"] + self.det.n > len(bb) - edge:
                continue                          # 변환 가장자리 — 다음 탐색에서 다시 본다
            pos = int(round(s0 + d["start"] * k))
            if pos < self.ignore_before or self._dup(d["kind"], pos):
                continue
            self.seen[d["kind"]].append(pos)
            v = self._plausible(d["kind"], pos, d, now)
            if v is None:
                self._suspect.append((d["kind"], pos, d))       # B 뒤 오디오가 아직 없음 → 다음 주기에 다시
                continue
            if not v:
                continue
            self._on_detect(d["kind"], pos, d, now)
        for kind, pos, d in list(self._suspect):
            v = self._plausible(kind, pos, d, now)
            if v is None and now - pos < int(3.0 * fs):
                continue
            self._suspect.remove((kind, pos, d))
            if v:
                self._on_detect(kind, pos, d, now)

        # 3) B 를 끝내 못 찾은 메시지: 최대 길이를 넘기면 A 위치부터 복조 시도
        if self.state == "rx" and self.pending is None and \
                now - self.a["abs"] > int((self.pc.max_message_s + 1.0) * fs):
            self._decode(self.a["abs"] - int(self.PRE_S * fs), now, "B 미검출 — 최대 길이 초과")
            self._reset_state()

    def _heartbeat(self, raw):
        """탐색이 실제로 돌고 있다는 기록 — 오검출 0회가 '아무것도 안 돌았다' 와 구별되도록."""
        self.stats["scans"] += 1
        hb = self._hb
        hb["scans"] += 1
        hb["max_rho"] = max(hb["max_rho"], getattr(self.det, "last_max", 0.0))
        hb["rms"] = max(hb["rms"], float(np.sqrt(np.mean(raw.astype(np.float64) ** 2))))
        if self.on_heartbeat is not None and time.time() - hb["t"] >= 60.0:
            self.on_heartbeat({"scans": hb["scans"], "max_rho": hb["max_rho"],
                               "rms_dbfs": 20 * np.log10(max(hb["rms"], 1e-12)),
                               "total_scans": self.stats["scans"], "A": self.stats["A"],
                               "B": self.stats["B"]})
            self._hb = {"t": time.time(), "scans": 0, "max_rho": 0.0, "rms": 0.0}

    # ---------------------------------------------------------------- 음성 오검출 방지 (수신 판단만, 송신 형식 그대로)
    # 측정 (results/remote/tng44_fa.log): 녹음 가짜 검출 17건은 모두 창 전력이 큰데 (s ≥ 26) ρ² 0.187~0.236,
    # 참 검출 453건 중 ρ² < 0.30 인 것은 저SNR (s 작음) 또는 Poor 페이딩 (ρ² 0.25~0.29). 그래서
    # '센 창 · 약한 상관' (s ≥ SUS_S, ρ² < SUS_RHO) 만 의심하고, 구조로 확인한다:
    #   A: 앞 300 ms 희생 톤이 같은 주파수 오차에 있어야 (톤 ρ² ≥ TONE_RHO)
    #   B: 송신 끝이므로 B 뒤 0.1~0.4 s 전력이 B 의 DROP 배 미만으로 떨어져야
    SUS_S = 8.0
    SUS_RHO = 0.30
    TONE_RHO = 0.35
    DROP = 0.35

    def _plausible(self, kind, pos, d, now):
        """True = 인정, False = 음성 오검출로 버림, None = 판단에 필요한 오디오가 아직 없음"""
        fs = self.fs
        n = self.n_costas
        if d["rho"] >= self.SUS_RHO:
            return True
        need_end = pos + n + int(0.45 * fs) if kind == "B" else pos + n
        if now < need_end:
            return None
        s0, raw = self.ring.read(pos - int(10.0 * fs), need_end)
        if len(raw) < int(1.0 * fs):
            return True
        bb = self.det._prefilter(np.asarray(self._to_baseband(raw), dtype=np.complex128))
        kk = fs / self.cfg.fs_base
        p = int(round((pos - s0) / kk))
        nb = self.det.n
        m = 100
        head = bb[:max(p - 700, 0)]
        L = len(head) // m * m
        if L < 5 * m:
            return True
        floor = float(np.quantile((np.abs(head[:L]) ** 2).reshape(-1, m).mean(1), 0.05)) / 0.84
        pw = float(np.mean(np.abs(bb[p:p + nb]) ** 2))
        s_hat = pw / max(floor, 1e-20) - 1.0
        if s_hat < self.SUS_S:
            return True
        if kind == "A":
            t = bb[max(p - 560, 0):max(p - 40, 0)]
            if len(t) < 200:
                return True
            w = np.exp(-2j * np.pi * d["df"] * np.arange(len(t)) / self.cfg.fs_base)
            r = abs(np.sum(t * w)) ** 2 / (len(t) * np.sum(np.abs(t) ** 2) + 1e-30)
            ok = r >= self.TONE_RHO
            why = "앞 희생 톤 없음 (톤 ρ² {:.2f})".format(r)
        else:
            a = bb[p + nb + 200:p + nb + 800]
            if len(a) < 300:
                return True
            drop = float(np.mean(np.abs(a) ** 2)) / max(pw, 1e-20)
            ok = drop < self.DROP
            why = "B 뒤 전력 안 떨어짐 ({:.2f})".format(drop)
        if not ok:
            self.stats["voice_drop"] = self.stats.get("voice_drop", 0) + 1
            self.on_status("코스타스 {} 무시 — 음성 오검출 의심: {} (ρ² {:.2f}, s {:.0f})".format(kind, why, d["rho"], s_hat), False)
        return ok

    def _dup(self, kind, pos):
        win = int(0.2 * self.fs)
        self.seen[kind] = [p for p in self.seen[kind] if abs(p - pos) < int(30 * self.fs)]
        return any(abs(p - pos) < win for p in self.seen[kind])

    def _on_detect(self, kind, pos, d, now):
        fs = self.fs
        self.stats[kind] += 1
        self.on_marker({"kind": kind, "abs": pos, "df": d["df"], "rho": d["rho"]})
        if kind == "A":
            if self.state == "rx" and self.pending is None:
                # 앞 메시지는 B 없이 끝났다 → A 위치 기준으로 복조 시도 (대체 경로)
                self._decode(self.a["abs"] - int(self.PRE_S * fs), pos - int(0.3 * fs),
                             "B 미검출 — 다음 신호 시작")
            self.state = "rx"
            self.a = {"abs": pos, "df": d["df"], "rho": d["rho"]}
            if self.on_live is not None:
                try:
                    from stream_ui import LiveStream
                    n_max = int(self.pc.max_message_s * self.cfg.fs_base / self.cfg.frame_len) + 2
                    self.live = LiveStream(self.engine, pos, d["df"], fs, n_max)
                except Exception as ex:
                    self.live = None
                    self.on_status("실시간 구간 복조 준비 실패 — {}".format(ex), True)
                self._live_event({"type": "start", "abs": pos, "df": d["df"], "rho": d["rho"], "fs": fs})
            self.on_status("신호 감지 (주파수 오차 {:+.1f} Hz, ρ² {:.2f})".format(d["df"], d["rho"]), False)
        else:
            self._on_b(pos, d)

    def _on_b(self, pos, d):
        """B 처리 (보류했던 B 도 여기로 다시 온다 — 검출 통계 · 표시는 _on_detect 에서 이미 했다)"""
        fs = self.fs
        lv = self.live
        if self.state == "rx" and lv is not None and lv.n_true is not None:
            # 길이 필드(v2)로 데이터 끝을 안다: B 는 예상 위치 (끝 가드 뒤) ±B_TOL_S 에서만 인정 (말소리 가짜 B 로 잘림 방지)
            exp_b = lv.end_abs()
            if abs(pos - exp_b) > int(self.B_TOL_S * fs) and not d.get("llr_ok"):
                if self.B_LLR_S > 0 and not lv.done and pos < exp_b:
                    self.b_hold = (pos, d)            # 예상보다 이른 B: 송신 중지일 수 있다 → B 뒤 신호 소실이면 인정
                    return
                self.on_status("코스타스 B 무시 — 길이 필드 예상 끝과 {:+.1f} s 차이 (ρ² {:.2f})".format(
                    (pos - exp_b) / fs, d["rho"]), True)
                self._live_event({"type": "b_ignored", "abs": pos, "rho": d["rho"], "off_s": (pos - exp_b) / fs})
                return
        if self.state == "rx" and lv is not None and lv.n_true is None and self.B_LLR_S > 0 and \
                d["rho"] < self.B_STRONG and \
                lv.format in ("stream1", "stream2") and not lv.done and not d.get("llr_ok") and \
                pos > self.a["abs"] + 3 * int(self.cfg.frame_len * fs / self.cfg.fs_base):
            self.b_hold = (pos, d)                # 길이 모름: B 뒤 신호 소실을 확인하고 인정
            return
        if self.state == "rx" and pos > self.a["abs"] + 3 * int(self.cfg.frame_len * fs / self.cfg.fs_base):
            end = pos + self.n_costas + int(self.POST_S * fs)
            self.pending = (end, self.a["abs"] - int(self.PRE_S * fs), "코스타스 B")
            self.state = "decoding"
            if self.live is not None:
                self.live.done = True
            self._live_event({"type": "b", "abs": pos, "df": d["df"], "rho": d["rho"], "fs": fs})
            self.on_status("메시지 끝 감지 — 복조 중…", False)
        elif self.state == "idle" and pos <= self.expect_b_until:
            pass                                  # 길이 필드로 이미 끝낸 메시지의 B
        elif self.state == "idle" and self.pending is None:
            # 역방향 동기: A 를 놓친 메시지 — 녹음 버퍼에서 B 앞 최대 길이를 되짚어 복조 (A 는 복조기가 B 기준으로 다시 찾음)
            end = pos + self.n_costas + int(self.POST_S * fs)
            self.pending = (end, max(0, pos - int((self.pc.max_message_s + 1.0) * fs)), "코스타스 B 역방향")
            self.on_status("메시지 끝(B)만 감지 — 시작(A)을 B 기준으로 되짚어 찾는 중…", False)
            self._live_event({"type": "b_back", "abs": pos, "df": d["df"], "rho": d["rho"], "fs": fs})

    def _b_llr(self, lv, pos):
        """B 끝 뒤 B_LLR_S 초 (프레임 단위) 의 평균 |LLR| (중앙 정렬 가설). 오디오가 아직 없으면 None"""
        from stream_ui import OFFSETS
        Lk = self.cfg.frame_len * lv.k
        f0 = int(np.ceil((pos + self.n_costas + 0.05 * self.fs - lv.f0_abs) / Lk))
        nf = max(1, int(round(self.B_LLR_S * self.cfg.fs_base / self.cfg.frame_len)))
        got = lv._windows(self.ring, f0, f0 + nf)
        if got is None:
            return None
        h = int(np.argmin(np.abs(np.asarray(OFFSETS) - lv.dec.center)))
        return float(np.abs(got[0][h]).mean())

    def _check_b_hold(self):
        pos, d = self.b_hold
        lv = self.live
        if self.state != "rx" or lv is None:
            self.b_hold = None                    # 그 사이 끝났다
            return
        if lv.n_true is not None and abs(pos - lv.end_abs()) <= int(self.B_TOL_S * self.fs):
            self.b_hold = None                    # 그 사이 길이를 알았고 예상 끝과 맞다 → 원래 규칙으로
            self._on_b(pos, dict(d, llr_ok=True))
            return
        m = self._b_llr(lv, pos)
        if m is None:
            return
        self.b_hold = None
        if m < self.B_LLR_T:
            self._on_b(pos, dict(d, llr_ok=True, b_llr=m))
        else:
            self.on_status("코스타스 B 무시 — B 뒤 신호 계속 (|LLR| {:.2f}, ρ² {:.2f})".format(m, d["rho"]), True)
            self._live_event({"type": "b_ignored", "abs": pos, "rho": d["rho"], "llr": m})

    def _live_event(self, ev):
        if self.on_live is not None:
            try:
                self.on_live(ev)
            except Exception:
                pass

    def _decode(self, start, end, why):
        s0, seg = self.ring.read(start, end)
        if len(seg) < int(1.0 * self.fs):
            return
        self.stats["decoded"] += 1
        if self.state == "decoding":
            self.state = "idle"
            self.a = None

        def work():
            try:
                r = self.engine.decode(seg, self.fs)
            except Exception as ex:
                from engine import DecodeResult
                r = DecodeResult(False, reason="{}: {}".format(type(ex).__name__, ex))
            if r.ok:
                self.stats["ok"] += 1
            self.on_decoded(r, "실시간 · {}".format(why))

        if self.sync:
            work()
        else:
            threading.Thread(target=work, daemon=True, name="rt-decode").start()
