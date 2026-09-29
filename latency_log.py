"""
검출 표시 지연 요약 기록 (항상 켜짐, --rxlog 없이도). 로그 폴더/latency_YYYYMMDD.csv

기준 = 입력 콜백 시각: 동기 구간 끝 샘플이 콜백으로 들어온 순간 (벽시계) → 사건 시각까지.
  stage  emit    수신 스레드가 표시 사건을 낸 순간
         ui      화면 스레드가 그 사건을 받은 순간 (워터폴에 그려지는 시점)
         decoded 복조 결과가 나온 순간 (기준 = 같은 모드의 마지막 B 끝, 없으면 A 끝)
  dev_s  장치 입력 버퍼 지연 (PortAudio currentTime − inputBufferAdcTime, 콜백마다). 실제 소리 → 콜백 사이 시간
합성 입력 (자가검사) 에서도 같은 방식으로 기록된다 (dev_s 는 빈칸).
"""
import os
import threading
import time
from collections import deque


class LatencyLog:
    def __init__(self, log_dir):
        self.dir = log_dir
        self.arrive = deque(maxlen=20000)          # (지금까지 받은 샘플 수, 벽시계)
        self.dev_s = None
        self.fs = None
        self.last_sync_end = {}                    # 모드 → (끝 샘플, 도착 벽시계)
        self._lock = threading.Lock()

    def block(self, count_end, fs, dev_s=None):
        self.fs = float(fs)
        self.arrive.append((count_end, time.perf_counter()))
        if dev_s is not None:
            self.dev_s = dev_s

    def _t_arrive(self, n):
        t_hit = None
        for c, t in reversed(self.arrive):         # 가장 최근부터 되짚어 — 끝 샘플 n 이 처음 들어온 블록
            if c < n:
                break
            t_hit = t
        return t_hit

    def event(self, m, stage):
        try:
            if m.get("remove") or m.get("tentative") or self.fs is None:
                return
            end = int(m["abs"] + float(m.get("dur", 0.224)) * self.fs)
            ta = self._t_arrive(end)
            if ta is None:
                return
            mode = m.get("mode", "TNG44")
            if stage == "emit":
                self.last_sync_end[mode] = (end, ta)
            self._write(mode, m["kind"], stage, time.perf_counter() - ta)
        except Exception:
            pass

    def decoded(self, mode):
        try:
            v = self.last_sync_end.get(mode)
            if v is not None:
                self._write(mode, "result", "decoded", time.perf_counter() - v[1])
        except Exception:
            pass

    def _write(self, mode, kind, stage, lat):
        if not self.dir:
            return
        os.makedirs(self.dir, exist_ok=True)
        p = os.path.join(self.dir, "latency_{}.csv".format(time.strftime("%Y%m%d")))
        new = not os.path.exists(p)
        with self._lock, open(p, "a", encoding="utf-8") as f:
            if new:
                f.write("local_time,mode,kind,stage,latency_s,device_buffer_s,fs\n")
            f.write("{},{},{},{},{:.3f},{},{:.0f}\n".format(time.strftime("%H:%M:%S"), mode, kind, stage, lat,
                                                          "" if self.dev_s is None else "{:.3f}".format(self.dev_s), self.fs or 0))
